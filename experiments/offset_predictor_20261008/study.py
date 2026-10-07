# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare duration estimators with elapsed time masked, using episode splits."""

import copy
import hashlib
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_offset_predictor_20261008")
OLD = Path("/mnt/localnvme/glm53_live_predictors_20261008")
RESEARCH = Path("/mnt/localnvme/vllm_glm53_predictor_expanded_20261007")
spec = importlib.util.spec_from_file_location(
    "offset_arch", RESEARCH / "experiments/predictor_architecture_cpu_20261007/study.py"
)
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)
original_weights = arch.weights
MLP = dict(kind="mlp", layers=[64, 64], input="log1p", solver="lbfgs", alpha=0.01)


def weights(rows):
    result = np.zeros(len(rows))
    # Equal source mass; identical for the paired feature ablations.
    sources = np.array([r["_source"] for r in rows])
    for source in sorted(set(sources)):
        mask = sources == source
        values = original_weights([r for r, keep in zip(rows, mask) if keep])
        result[mask] = values / values.sum()
    return result * len(rows) / result.sum()


arch.weights = weights


def save(name, data):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def extra(rows):
    values = []
    for row in rows:
        prefix = causal = sparse = 0
        for q, c in row["request_shapes"]:
            prefix += q * c
            causal += q * (q + 1) / 2
            n = min(q, max(0, 2048 - c))
            sparse += n * c + n * (n + 1) / 2 + (q - n) * 2048
        values.append([prefix, causal, sparse])
    return np.asarray(values)


def matrix(rows, offset=False, sparse=False):
    x = arch.runtime_features(rows)
    if offset:
        x[:, 21] = 0
    return np.column_stack((x, extra(rows))) if sparse else x


def live_rows():
    batches, forecasts = [], []
    for path in sorted((OLD / "live").glob("*/r[0-3]_*/result.json")):
        result = json.loads(path.read_text())
        assert result["accepted"]
        # Purge all predictor variants with the same workload and prompt seed.
        episode = f"live:{result['workload']}:{result['repeat']}"
        events = [
            json.loads(line)
            for line in (path.parent / "engine.jsonl").read_text().splitlines()
        ]
        done = {
            e["step"]: e
            for e in events
            if e["event"] == "step_completion"
            and e.get("valid")
            and e.get("selected")
            and e.get("prefill_only")
        }
        for data in done.values():
            ranks = data["execution_ranks"]
            if len(ranks) != 8 or not all(r[5] == 1 and r[6] > 0 for r in ranks):
                continue
            row = dict(
                data,
                episode=episode,
                workload=result["workload"],
                _source="live",
                _phase="live",
                _repeat=result["repeat"],
                source_file=str(path.parent / "engine.jsonl"),
            )
            batches.append(row)
        late = json.loads((path.parent / "late.json").read_text())["records"]
        for point in late:
            if (
                point["event"] != "seal_prediction"
                or point.get("predictor_input") is None
            ):
                continue
            data = done[point["predecessor_step"]]
            row = dict(
                data,
                episode=episode,
                workload=result["workload"],
                _source="live",
                _phase="live",
                _repeat=result["repeat"],
                forecast_time_ns=point["forecast_ns"],
                forecast_features=point["predictor_input"][11:23],
                source_file=str(path.parent / "engine.jsonl"),
            )
            assert arch.valid(row, True)
            forecasts.append(row)
    assert len(forecasts) == 215
    return batches, forecasts


def tree_fit(batches, forecasts, seed):
    base, _ = arch.fit(
        arch.FIXED_BASE, arch.features(batches), arch.target(batches), batches, seed
    )
    oof, audit = arch.cross_fit(arch.FIXED_BASE, batches, forecasts, seed)
    assert all(a["episode_overlap"] == 0 for a in audit)
    return base, oof


def variants(batches, forecasts, seed):
    base, oof = tree_fit(batches, forecasts, seed)
    states = {}
    names = arch.surface.FEATURE_NAMES + arch.surface.FORECAST_NAMES
    for offset in [False, True]:
        x = matrix(forecasts, offset)
        correction, _ = arch.fit(
            arch.FIXED_CORRECTION,
            np.column_stack((x, oof)),
            arch.target(forecasts) - oof,
            forecasts,
            seed,
        )
        tree = dict(
            base=arch.export_component(base, arch.surface.FEATURE_NAMES),
            correction=arch.export_component(
                correction, names + arch.surface.BASE_PREDICTION_NAMES
            ),
        )
        suffix = "offset" if offset else "elapsed"
        guard_indices = [0, 1, 4, 5, 6, 7]
        bounds = dict(
            indices=guard_indices,
            minimum=x[:, guard_indices].min(axis=0).tolist(),
            maximum=x[:, guard_indices].max(axis=0).tolist(),
            fallback_model_index=1,
        )
        checkpoint = dict(
            models=[tree, tree],
            weights=[1.0, 0.0],
            support_guard=bounds,
        )
        if offset:
            checkpoint["excluded_feature_indices"] = [21]
        states[f"trees_{suffix}"] = checkpoint
        neural_x = matrix(forecasts, offset, True)
        mlp, _ = arch.fit(MLP, neural_x, arch.target(forecasts), forecasts, seed)
        mlp_state = dict(
            direct=arch.export_component(mlp, names + ["prefix", "causal", "sparse"]),
            input_indices=list(range(23)) + [28, 29, 30],
        )
        neural_checkpoint = copy.deepcopy(checkpoint)
        neural_checkpoint["models"] = [mlp_state, tree]
        states[f"mlp_{suffix}"] = neural_checkpoint
    return states


def predict(checkpoint, rows):
    offset = checkpoint.get("excluded_feature_indices") == [21]
    x = matrix(rows, offset, True)
    # The exported fallback consumes the ordinary 23-column geometry/runtime input.
    result = []
    models = [arch.FrozenArchitecture(m) for m in checkpoint["models"]]
    guard = checkpoint["support_guard"]
    for value in x:
        outside = any(
            value[i] < lo or value[i] > hi
            for i, lo, hi in zip(guard["indices"], guard["minimum"], guard["maximum"])
        )
        index = guard["fallback_model_index"] if outside else 0
        model_input = value if checkpoint["models"][index].get("direct") else value[:23]
        result.append(models[index].predict_one(model_input))
    return np.array(result)


def main():
    ROOT.mkdir(exist_ok=True)
    pool = joblib.load("/mnt/localnvme/glm53_predictor_expanded_20261007/pool.joblib")
    live_batches, live_forecasts = live_rows()
    save(
        "protocol.json",
        dict(
            sources=dict(Counter(r["_source"] for r in pool["forecasts"])),
            old_live_forecasts=len(live_forecasts),
            old_live_batches=len(live_batches),
            paired_feature_ablation=True,
            held_groups="All predictor variants with same workload and prompt seed",
            training_sources=["historical", "fresh", "live"],
            source_weighting="Equal total mass for each source",
            elapsed_column=21,
            elapsed_mask=0,
            target="critical_end_after_all_rank_start.v1",
            seed=73,
            blind_live_seeds=[2301, 2302, 2303, 2304],
            original_final_benchmark_used_for_training=False,
            fitting_configurations=dict(
                base=arch.FIXED_BASE, correction=arch.FIXED_CORRECTION, mlp=MLP
            ),
        ),
    )
    cv = []
    with threadpool_limits(limits=1):
        for source in ["original", "live_aware"]:
            for repeat in range(4):
                output = ROOT / "folds" / source / f"r{repeat}"
                completed = output / "scores.json"
                if completed.exists():
                    cv.extend(json.loads(completed.read_text()))
                    continue
                test = [r for r in live_forecasts if r["_repeat"] == repeat]
                train = list(pool["forecasts"])
                batches = list(pool["batches"])
                if source == "live_aware":
                    train += [r for r in live_forecasts if r["_repeat"] != repeat]
                    batches += [r for r in live_batches if r["_repeat"] != repeat]
                assert not {r["episode"] for r in test}.intersection(
                    r["episode"] for r in train + batches
                )
                begin = time.monotonic()
                states = variants(batches, train, 73)
                scores = []
                for name, state in states.items():
                    prediction = predict(state, test)
                    scores.append(
                        dict(
                            source=source,
                            repeat=repeat,
                            name=name,
                            metrics=arch.metrics(arch.target(test), prediction, test),
                            predictions=prediction.tolist(),
                            actual=arch.target(test).tolist(),
                            workloads=[r["workload"] for r in test],
                            test_episode_overlap=0,
                        )
                    )
                    save(str(output.relative_to(ROOT) / f"{name}.json"), state)
                save(str(completed.relative_to(ROOT)), scores)
                cv.extend(scores)
                print(
                    json.dumps(
                        dict(
                            source=source,
                            repeat=repeat,
                            seconds=time.monotonic() - begin,
                        )
                    ),
                    flush=True,
                )
        aggregate = []
        for source in ["original", "live_aware"]:
            for name in ["trees_elapsed", "trees_offset", "mlp_elapsed", "mlp_offset"]:
                selected = [
                    r for r in cv if r["source"] == source and r["name"] == name
                ]
                y = np.concatenate([r["actual"] for r in selected])
                p = np.concatenate([r["predictions"] for r in selected])
                ws = sum([r["workloads"] for r in selected], [])
                error = p - y
                per_workload = {
                    w: float(np.mean(np.abs(error[np.array(ws) == w])))
                    for w in sorted(set(ws))
                }
                aggregate.append(
                    dict(
                        source=source,
                        name=name,
                        mae_ms=float(np.mean(np.abs(error))),
                        macro_mae_ms=float(np.mean(list(per_workload.values()))),
                        bias_ms=float(np.mean(error)),
                        p95_ms=float(np.quantile(np.abs(error), 0.95)),
                        per_workload_mae_ms=per_workload,
                    )
                )
        save("cpu_comparison.json", aggregate)
        candidates = [
            r
            for r in aggregate
            if r["source"] == "live_aware" and r["name"].endswith("_offset")
        ]
        winner = min(candidates, key=lambda r: r["macro_mae_ms"])
        family = winner["name"].split("_")[0]
        save(
            "selection.json",
            dict(
                selected_family=family,
                selected=winner,
                criterion="Live-aware grouped out-of-fold workload-macro MAE",
                matched_control=f"{family}_elapsed",
            ),
        )
        train = pool["forecasts"] + live_forecasts
        batches = pool["batches"] + live_batches
        states = variants(batches, train, 73)
        tree = json.loads((OLD / "models/trees.json").read_text())
        for suffix in ["elapsed", "offset"]:
            state = copy.deepcopy(tree)
            checkpoint = states[f"{family}_{suffix}"]
            state["latency_surface"] = dict(
                schema=4,
                architecture="guarded_sparse_mlp_offset"
                if suffix == "offset"
                else "guarded_sparse_mlp",
                checkpoint=checkpoint,
                provenance=dict(
                    general_prefill=True,
                    seed=73,
                    fitting="original_plus_old_live",
                    excluded_features=["elapsed_execution_ms"]
                    if suffix == "offset"
                    else [],
                ),
                target_schema="critical_end_after_all_rank_start.v1",
            )
            save(f"models/matched_{suffix}.json", state)
        save("models/trees.json", tree)
        save(
            "freeze.json",
            dict(
                models={
                    p.stem: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT / "models").glob("*.json")
                },
                blind_seeds=[2301, 2302, 2303, 2304],
                selection=winner,
            ),
        )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
