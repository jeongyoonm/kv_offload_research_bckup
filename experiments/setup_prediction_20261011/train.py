# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped CPU setup study; freeze selection before opening the final split."""

import argparse
import gzip
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "setup_runtime", SOURCE / "vllm/v1/kv_offload/setup_predictor.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def metrics(rows, prediction, target):
    y = np.array([r[target] for r in rows])
    weight = np.array([r["weight"] for r in rows])
    error = prediction - y
    result = dict(
        mae_ms=float(np.average(abs(error), weights=weight)),
        bias_ms=float(np.average(error, weights=weight)),
        covered=float(np.average(error >= 0, weights=weight)),
        shortfall_ms=float(np.average(np.maximum(-error, 0), weights=weight)),
        excess_ms=float(np.average(np.maximum(error, 0), weights=weight)),
    )
    for position in ("first", "last"):
        selected = np.array([r[position] for r in rows])
        result[position + "_mae_ms"] = float(np.mean(abs(error[selected])))
    return result


def export(model, name, target, config):
    if isinstance(model, ExtraTreesRegressor):
        trees = [
            [
                [int(left), int(right), int(f), float(t), float(v)]
                for left, right, f, t, v in zip(
                    estimator.tree_.children_left,
                    estimator.tree_.children_right,
                    estimator.tree_.feature,
                    estimator.tree_.threshold,
                    estimator.tree_.value[:, 0, 0],
                )
            ]
            for estimator in model.estimators_
        ]
        shift = 0.0
    else:
        count = len(model._predictors)
        shift = 1000.0
        baseline = float(model._baseline_prediction[0, 0])
        trees = []
        for predictors in model._predictors:
            tree = []
            for node in predictors[0].nodes:
                leaf = bool(node["is_leaf"])
                tree.append(
                    [
                        -1 if leaf else int(node["left"]),
                        -1 if leaf else int(node["right"]),
                        -2 if leaf else int(node["feature_idx"]),
                        float(node["num_threshold"]),
                        shift + baseline + count * float(node["value"])
                        if leaf
                        else 0.0,
                    ]
                )
            trees.append(tree)
    return dict(
        schema=1,
        feature_names=runtime.FEATURE_NAMES,
        model_id=name,
        target=target,
        trees=trees,
        shift_ms=shift,
        config=config,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal-snapshot-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "selection.json").exists()
    rows = json.loads(gzip.decompress(args.data.read_bytes()))
    snapshot_audit = {}
    if args.seal_snapshot_only:
        kept = []
        for row in rows:
            if not row["last"]:
                continue
            actual = runtime.setup_features(
                row["final_shapes"], row["features"][7], np.zeros(28)
            )
            matches = actual is not None and np.array_equal(
                actual[:7], np.asarray(row["features"][:7], np.float32)
            )
            if not matches:
                snapshot_audit["geometry_mismatch"] = (
                    snapshot_audit.get("geometry_mismatch", 0) + 1
                )
                continue
            kept.append(dict(row, weight=1.0, first=True))
        snapshot_audit["retained_batches"] = len(kept)
        snapshot_audit["discarded_earlier_frames"] = sum(not r["last"] for r in rows)
        rows = kept
        (args.output / "snapshot_audit.json").write_text(
            json.dumps(snapshot_audit, indent=2) + "\n"
        )
    splits = {
        name: [r for r in rows if r["split"] == name]
        for name in ("train", "dev", "final")
    }
    assert all(splits.values())
    groups = {name: {r["group"] for r in part} for name, part in splits.items()}
    assert not groups["train"] & groups["dev"]
    assert not groups["train"] & groups["final"]
    x = {
        name: np.array([r["features"] for r in part], np.float32)
        for name, part in splits.items()
    }
    weights = np.array([r["weight"] for r in splits["train"]])
    candidates, models = {}, {}
    configs = {
        "extra80": dict(n_estimators=80, min_samples_leaf=20, max_depth=12),
        "extra80_small": dict(n_estimators=80, min_samples_leaf=50, max_depth=10),
        "boost120": dict(max_iter=120, max_leaf_nodes=15, min_samples_leaf=30),
        "boost80_small": dict(max_iter=80, max_leaf_nodes=7, min_samples_leaf=50),
    }
    if args.seal_snapshot_only:
        configs.update(
            extra80_leaf5=dict(n_estimators=80, min_samples_leaf=5, max_depth=12),
            extra80_leaf10=dict(n_estimators=80, min_samples_leaf=10, max_depth=12),
        )
    with threadpool_limits(limits=4):
        for target in ("target_cpu_ms", "target_ready_ms"):
            for name, config in configs.items():
                key = target + "_" + name
                estimator = (
                    ExtraTreesRegressor(random_state=171, n_jobs=4, **config)
                    if name.startswith("extra")
                    else HistGradientBoostingRegressor(
                        random_state=171, early_stopping=False, **config
                    )
                )
                begin = time.monotonic()
                estimator.fit(
                    x["train"],
                    [r[target] for r in splits["train"]],
                    sample_weight=weights,
                )
                score = metrics(splits["dev"], estimator.predict(x["dev"]), target)
                candidates[key] = dict(
                    dev=score, train_seconds=time.monotonic() - begin
                )
                models[key] = estimator
                print(key, score, flush=True)
        best = min(
            (k for k in models if k.startswith("target_ready_ms")),
            key=lambda k: candidates[k]["dev"]["mae_ms"],
        )
        # This second proposal learns a coverage target rather than adding a fixed lead.
        quantile = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.9,
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            random_state=171,
            early_stopping=False,
        ).fit(
            x["train"],
            [r["target_ready_ms"] for r in splits["train"]],
            sample_weight=weights,
        )
        models["ready_q90"] = quantile
        candidates["ready_q90"] = dict(
            dev=metrics(splits["dev"], quantile.predict(x["dev"]), "target_ready_ms")
        )
        selection = dict(
            seal_snapshot_only=args.seal_snapshot_only,
            snapshot_audit=snapshot_audit,
            mean=best,
            uncertainty="ready_q90",
            candidates=candidates,
            groups={k: sorted(v) for k, v in groups.items()},
        )
        (args.output / "selection.json").write_text(json.dumps(selection, indent=2))
        # Final labels are inspected only after selection has been written.
        final = {}
        for arm, key in (("mean", best), ("uncertainty", "ready_q90")):
            model = models[key]
            final[arm] = metrics(
                splits["final"], model.predict(x["final"]), "target_ready_ms"
            )
            model_id = key + ("_seal_snapshot" if args.seal_snapshot_only else "")
            state = export(model, model_id, "target_ready_ms", model.get_params())
            indices = [3, 4, 10, 11]
            state["support"] = dict(
                indices=indices,
                minimum=[0, 0, 0, 0],
                maximum=x["train"][:, indices].max(axis=0).tolist(),
            )
            exported = runtime.SetupPredictor(state)
            actual = model.predict(x["final"])
            reference = np.array(
                [exported.predict_ms(v, native=False) for v in x["final"]]
            )
            native = np.array([exported.predict_ms(v) for v in x["final"]])
            assert np.max(abs(reference - actual)) < 2e-6
            assert np.max(abs(native - reference)) < 2e-6
            begin = time.perf_counter_ns()
            for v in x["final"]:
                exported.predict_ms(v)
            final[arm]["runtime_us"] = (
                (time.perf_counter_ns() - begin) / len(x["final"]) / 1000
            )
            final[arm]["runtime_native"] = exported.native is not None
            (args.output / (arm + ".json")).write_text(json.dumps(state))
        for split in ("dev", "final"):
            final["fixed50_" + split] = metrics(
                splits[split], np.full(len(splits[split]), 50.0), "target_ready_ms"
            )
        (args.output / "final.json").write_text(json.dumps(final, indent=2))
        print(json.dumps(dict(selection=selection, final=final), indent=2), flush=True)


if __name__ == "__main__":
    main()
