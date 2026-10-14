# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refit the original offset tree recipe using current worker traces."""

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "retrain_arch", SOURCE / "experiments/predictor_architecture_cpu_20261007/study.py"
)
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def collect(roots):
    batches, forecasts, audit = [], [], []
    for root in roots:
        plan = json.loads((root / "plan.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        assert summary["complete"] and summary["stopped"]
        assert (
            plan["connector"]["kv_connector_extra_config"]["async_h2d_chunk_bytes"]
            == 8 * 1024**2
        )
        assert all(p["copy_pipeline_depth"] == 2 for p in plan["policies"].values())
        counts = Counter()
        for path in sorted((root / "live").glob("*/*/result.json")):
            result = json.loads(path.read_text())
            if result["repeat"] == "warmup":
                continue
            assert result["accepted"] and result["frozen_unchanged"]
            group = f"{root.name}:{result['workload']}:{result['seed']}"
            events = [
                json.loads(line)
                for line in (path.parent / "engine.jsonl").read_text().splitlines()
            ]
            done = {}
            for event in events:
                if (
                    event["event"] != "step_completion"
                    or not event.get("valid")
                    or not event.get("selected")
                    or not event.get("prefill_only")
                ):
                    continue
                row = dict(
                    event,
                    episode=group,
                    workload=result["workload"],
                    _source=root.name,
                    source_file=str(path.parent / "engine.jsonl"),
                )
                if not arch.valid(row):
                    counts["invalid_gpu_start_bracket"] += 1
                    continue
                if not all(r[5] == 1 and r[6] > 0 for r in row["execution_ranks"]):
                    continue
                batches.append(row)
                done[row["step"]] = row
                counts["batches"] += 1
            late = json.loads((path.parent / "late.json").read_text())["records"]
            seen = set()
            for point in late:
                if point["event"] != "seal_prediction" or not point.get(
                    "predictor_input"
                ):
                    continue
                step = point["predecessor_step"]
                if step not in done or step in seen:
                    continue
                row = dict(
                    done[step],
                    forecast_time_ns=point["forecast_ns"],
                    forecast_features=point["predictor_input"][11:23],
                )
                if not arch.valid(row, True):
                    counts["noncausal_forecasts"] += 1
                    continue
                rebuilt = arch.runtime_features([row])[0]
                assert np.allclose(rebuilt, point["predictor_input"][:23])
                forecasts.append(row)
                seen.add(step)
                counts["forecasts"] += 1
        audit.append(dict(root=str(root), counts=dict(counts)))
    return batches, forecasts, audit


original_weights = arch.weights


def weights(rows):
    values = np.zeros(len(rows))
    for source in sorted({r["_source"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["_source"] == source]
        w = original_weights([rows[i] for i in indices])
        values[indices] = w / w.sum()
    return values * len(rows) / values.sum()


arch.weights = weights


def matrix(rows):
    x = arch.runtime_features(rows)
    x[:, 21] = 0
    return x


def fit(batches, forecasts, seed):
    base, _ = arch.fit(
        arch.FIXED_BASE, arch.features(batches), arch.target(batches), batches, seed
    )
    oof, audit = arch.cross_fit(arch.FIXED_BASE, batches, forecasts, seed)
    assert all(a["episode_overlap"] == 0 for a in audit)
    correction, _ = arch.fit(
        arch.FIXED_CORRECTION,
        np.column_stack((matrix(forecasts), oof)),
        arch.target(forecasts) - oof,
        forecasts,
        seed,
    )
    model = dict(
        base=arch.export_component(base, arch.surface.FEATURE_NAMES),
        correction=arch.export_component(
            correction,
            arch.surface.FEATURE_NAMES
            + arch.surface.FORECAST_NAMES
            + arch.surface.BASE_PREDICTION_NAMES,
        ),
    )
    x = matrix(forecasts)
    indices = [0, 1, 4, 5, 6, 7]
    state = dict(
        models=[model, model],
        weights=[1.0, 0.0],
        excluded_feature_indices=[21],
        support_guard=dict(
            indices=indices,
            minimum=x[:, indices].min(axis=0).tolist(),
            maximum=x[:, indices].max(axis=0).tolist(),
            fallback_model_index=1,
        ),
    )
    frozen = arch.FrozenArchitecture(model)
    exported = np.array([frozen.predict_one(xi) for xi in x])
    bp = arch.predict(base, arch.features(forecasts))
    fitted = np.maximum(1, bp + arch.predict(correction, np.column_stack((x, bp))))
    difference = np.abs(exported - fitted)
    if difference.max() > 2e-6:
        worst = int(difference.argmax())
        raise AssertionError(
            dict(
                max_error_ms=float(difference.max()),
                row=worst,
                exported=float(exported[worst]),
                fitted=float(fitted[worst]),
                base_sklearn=float(bp[worst]),
                base_exported=frozen.base.predict_one(x[worst][:11]),
            )
        )
    return state, audit


def score(state, rows):
    predictor = arch.FrozenArchitecture(state["models"][0])
    predicted = np.array([predictor.predict_one(x) for x in matrix(rows)])
    truth = arch.target(rows)
    return arch.metrics(truth, predicted, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-state", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "retrained.json").exists()
    batches, forecasts, audit = collect(args.roots)
    save(args.output / "collection_audit.json", audit)
    pool = dict(batches=batches, forecasts=forecasts)
    data = json.dumps(pool, separators=(",", ":")).encode()
    (args.output / "current_pool.json.gz").write_bytes(gzip.compress(data, mtime=0))
    save(
        args.output / "protocol.json",
        dict(
            base=arch.FIXED_BASE,
            correction=arch.FIXED_CORRECTION,
            seed=73,
            target="critical_end_after_all_rank_start.v1",
            elapsed_feature_mask=21,
            correction_training="three-fold purged out-of-fold base residual",
            source_weighting="equal allocation mass; workload weights per source",
            grouping="allocation, workload, seed; all policy variants grouped together",
            pool_sha256=hashlib.sha256(data).hexdigest(),
            sklearn=sklearn.__version__,
            batches=len(batches),
            forecasts=len(forecasts),
            external_shape_calibration=False,
            collection="Existing current-chunk serving traces; "
            "no old-worker rows mixed in",
            scope="Current observed shape coverage; "
            "fresh serving seeds are blind evaluation",
        ),
    )
    held = args.roots[-1].name
    train_b = [r for r in batches if r["_source"] != held]
    train_f = [r for r in forecasts if r["_source"] != held]
    test = [r for r in forecasts if r["_source"] == held]
    with threadpool_limits(limits=1):
        validation, folds = fit(train_b, train_f, 73)
        save(
            args.output / "held_allocation.json",
            dict(source=held, score=score(validation, test), folds=folds),
        )
        state, folds = fit(batches, forecasts, 73)
    old = json.loads(args.old_state.read_text())
    new = copy.deepcopy(old)
    new["latency_surface"]["checkpoint"] = state
    new["latency_surface"]["provenance"] = dict(
        general_prefill=True,
        seed=73,
        fitting="current_chunk_episode_purged_refit",
        excluded_features=["elapsed_execution_ms"],
    )
    save(args.output / "retrained.json", new)
    save(args.output / "cross_fit_audit.json", folds)
    print(json.dumps(dict(batches=len(batches), forecasts=len(forecasts), held=held)))


if __name__ == "__main__":
    main()
