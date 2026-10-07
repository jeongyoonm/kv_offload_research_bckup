# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched dataset additions with frozen tree recipes and episode cross-fitting."""

import argparse
import hashlib
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_dataset_coverage_20261007")
HIST = Path("/mnt/localnvme/glm53_predictor_overnight_20261006")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


arch = module(
    "coverage_architecture",
    SOURCE / "experiments/predictor_architecture_cpu_20261007/study.py",
)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def balanced(rows, limit):
    groups = defaultdict(list)
    for r in sorted(
        rows,
        key=lambda r: hashlib.sha256(
            f"{r['episode']}:{r['step']}".encode()
        ).hexdigest(),
    ):
        groups[r["workload"]].append(r)
    result = []
    names = sorted(groups)
    while len(result) < limit and any(groups.values()):
        for name in names:
            if groups[name] and len(result) < limit:
                result.append(groups[name].pop())
    return result


def predict(state, rows):
    model = arch.FrozenArchitecture(state)
    x = arch.runtime_features(rows)
    return np.array([model.predict_one(v) for v in x])


def scores(rows, pred):
    result = arch.metrics(arch.target(rows), pred, rows)
    errors = pred - arch.target(rows)
    result["deadline_errors"] = {
        str(t): dict(
            under_percent=float(np.mean(errors < -t) * 100),
            over_percent=float(np.mean(errors > t) * 100),
        )
        for t in [5, 10, 20]
    }
    result["by_block"] = {
        b: arch.metrics(
            arch.target([r for r in rows if r.get("block") == b]),
            pred[[i for i, r in enumerate(rows) if r.get("block") == b]],
            [r for r in rows if r.get("block") == b],
        )
        for b in sorted({r.get("block", "historical") for r in rows})
        if any(r.get("block") == b for r in rows)
    }
    return result


def exported(base, correction, provenance):
    return dict(
        base=arch.export_component(base, arch.surface.FEATURE_NAMES),
        correction=arch.export_component(
            correction,
            arch.surface.FEATURE_NAMES
            + arch.surface.FORECAST_NAMES
            + arch.surface.BASE_PREDICTION_NAMES,
        ),
        provenance=provenance,
    )


def serving(state):
    result = json.loads((HIST / "boosted_residual/model.json").read_text())
    base = state["base"]["state"]
    correction = state["correction"]
    result["state"]["latency_surface"] = base
    result["state"]["refinement_surface"] = dict(
        schema=3,
        target_schema="critical_end_after_all_rank_start.v1",
        base=base,
        residual=correction["state"],
        residual_shift_ms=correction.get("shift", 0),
        provenance=state["provenance"],
    )
    result["provenance"] = state["provenance"]
    return result


def data():
    hist_batches = read(HIST / "refreshed/data/batches.jsonl")
    hist_forecasts = read(HIST / "refreshed/data/forecasts.jsonl")
    fresh = read(ROOT / "data/forecasts.jsonl")
    new = [r for r in fresh if r["split"] == "train"]
    additions = {
        b: [r for r in new if r["block"] == b]
        for b in ["geometry", "composition", "transfer", "familiar"]
    }
    # One matching base row per forecast; base-only new rows cannot confound volume.
    limit = min(
        400,
        *(len(additions[b]) for b in ["geometry", "composition", "transfer"]),
        len(additions["familiar"]) // 3,
    )
    if limit < 30:
        raise RuntimeError(
            f"need at least 30 matched forecasts per block, have {limit}"
        )
    additions = {
        b: balanced(v, limit * (3 if b == "familiar" else 1))
        for b, v in additions.items()
    }
    development = [r for r in fresh if r["split"] == "development"]
    if not development:
        raise RuntimeError("development set is not collected")
    return hist_batches, hist_forecasts, additions, development, limit


def development():
    hb, hf, a, dev, limit = data()
    historical = []
    for phase in arch.PHASES:
        historical.extend(
            read(HIST / phase / "frozen_evaluation/forecasts/forecasts.jsonl")
        )
    variants = {
        "historical_refit": [],
        "familiar_small": a["familiar"][:limit],
        "familiar_full": a["familiar"],
        "geometry": a["geometry"],
        "composition": a["composition"],
        "transfer": a["transfer"],
        "combined": a["geometry"] + a["composition"] + a["transfer"],
    }
    for b in ["geometry", "composition", "transfer"]:
        variants["without_" + b] = [
            r
            for other in ["geometry", "composition", "transfer"]
            if other != b
            for r in a[other]
        ] + a["familiar"][:limit]
    for n in [100, 200]:
        if n >= limit:
            continue
        variants[f"combined_n{3 * n}"] = [
            r
            for b in ["geometry", "composition", "transfer"]
            for r in balanced(a[b], n)
        ]
        variants[f"familiar_n{3 * n}"] = balanced(a["familiar"], 3 * n)
    audit = dict(
        limit_per_block=limit,
        historical_base=len(hb),
        historical_forecasts=len(hf),
        counts={k: len(v) for k, v in variants.items()},
        extra_base_per_forecast=1,
        hashes={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / "matrix.json", ROOT / "data/forecasts.jsonl"]
        },
    )
    dump(ROOT / "study/dataset_audit.json", audit)
    outputs = {}
    for name, extra in variants.items():
        rows = hf + extra
        batches = hb + extra
        assert not {r["episode"] for r in rows + batches} & {r["episode"] for r in dev}
        for seed in [73, 74, 75]:
            directory = ROOT / "study/models" / name / f"seed{seed}"
            metrics_path = directory / "development.json"
            if metrics_path.exists():
                outputs[f"{name}:{seed}"] = json.loads(metrics_path.read_text())
                continue
            print(
                json.dumps(dict(event="fit", name=name, seed=seed, samples=len(rows))),
                flush=True,
            )
            base, correction, cross_audit = arch.full_fit(
                arch.FIXED_BASE, arch.FIXED_CORRECTION, batches, rows, seed
            )
            provenance = dict(
                variant=name,
                seed=seed,
                base_samples=len(batches),
                forecast_samples=len(rows),
                general_prefill=True,
                features_unchanged=True,
                cross_fit=cross_audit,
            )
            state = exported(base, correction, provenance)
            dev_pred = predict(state, dev)
            bx = arch.features(dev)
            bp = base.predict(bx)
            expected = np.maximum(
                1,
                bp
                + correction.predict(
                    np.column_stack((arch.runtime_features(dev), bp)).astype(np.float32)
                ),
            )
            assert np.max(np.abs(dev_pred - expected)) < 0.00001
            report = dict(
                development=scores(dev, dev_pred),
                historical=scores(historical, predict(state, historical)),
            )
            dump(directory / "architecture.json", state)
            dump(directory / "serving.json", serving(state))
            dump(metrics_path, report)
            outputs[f"{name}:{seed}"] = report
    summary = {
        name: dict(
            development_macro_mae=float(
                np.mean(
                    [
                        outputs[f"{name}:{s}"]["development"]["workload_macro_mae_ms"]
                        for s in [73, 74, 75]
                    ]
                )
            ),
            historical_mae=float(
                np.mean(
                    [
                        outputs[f"{name}:{s}"]["historical"]["mae_ms"]
                        for s in [73, 74, 75]
                    ]
                )
            ),
        )
        for name in variants
    }
    threshold = summary["historical_refit"]["historical_mae"] * 1.1
    eligible = [
        name
        for name in ["geometry", "composition", "transfer", "combined"]
        if summary[name]["historical_mae"] <= threshold
    ]
    winner = (
        min(eligible, key=lambda n: summary[n]["development_macro_mae"])
        if eligible
        else "historical_refit"
    )
    selection = dict(
        winner=winner,
        seed=73,
        variants=summary,
        development_only=True,
        checkpoint=str(ROOT / "study/models" / winner / "seed73/serving.json"),
        historical_limit_ms=threshold,
    )
    dump(ROOT / "study/selection.json", selection)
    print(json.dumps(selection), flush=True)


def final():
    selection = json.loads((ROOT / "study/selection.json").read_text())
    rows = [r for r in read(ROOT / "data/forecasts.jsonl") if r["split"] == "final"]
    if not rows:
        raise RuntimeError("no fresh final forecasts")
    results = {}
    predictions = {}
    for name in selection["variants"]:
        p = []
        for seed in [73, 74, 75]:
            state = json.loads(
                (
                    ROOT / "study/models" / name / f"seed{seed}/architecture.json"
                ).read_text()
            )
            p.append(predict(state, rows))
        predictions[name] = np.mean(p, axis=0)
        results[name] = dict(
            seed_scores=[scores(rows, x) for x in p],
            mean_prediction=scores(rows, predictions[name]),
        )
    y = arch.target(rows)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["episode"]].append(i)
    keys = list(groups)
    rng = np.random.default_rng(820)
    intervals = {}
    for name, reference in [
        ("geometry", "familiar_small"),
        ("composition", "familiar_small"),
        ("transfer", "familiar_small"),
        ("combined", "familiar_full"),
    ]:
        differences = np.abs(predictions[name] - y) - np.abs(predictions[reference] - y)
        samples = []
        for _ in range(2000):
            indices = [
                i
                for key in rng.choice(keys, size=len(keys), replace=True)
                for i in groups[key]
            ]
            by_workload = defaultdict(list)
            for i in indices:
                by_workload[rows[i]["workload"]].append(differences[i])
            samples.append(np.mean([np.mean(v) for v in by_workload.values()]))
        intervals[name] = dict(
            reference=reference,
            macro_mae_difference_ci95_ms=np.quantile(samples, [0.025, 0.975]).tolist(),
        )
    dump(
        ROOT / "study/final.json",
        dict(
            samples=len(rows),
            episodes=len(groups),
            selection=selection,
            results=results,
            paired_intervals=intervals,
        ),
    )
    (ROOT / "study/final_predictions.jsonl").write_text(
        "".join(
            json.dumps(
                dict(
                    episode=r["episode"],
                    workload=r["workload"],
                    block=r["block"],
                    actual_ms=float(y[i]),
                    predictions={name: float(p[i]) for name, p in predictions.items()},
                )
            )
            + "\n"
            for i, r in enumerate(rows)
        )
    )
    print(
        json.dumps(
            dict(event="final_complete", samples=len(rows), paired_intervals=intervals)
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with threadpool_limits(limits=1):
        final() if args.final else development()
