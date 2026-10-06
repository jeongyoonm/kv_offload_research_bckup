# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persist a wide cached base and a causal residual trained on held-out groups."""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from evaluate import arrays, features, read, target, tree, weights
from sklearn.model_selection import GroupKFold

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "latency_surface", SOURCE / "vllm/v1/kv_offload/latency_surface.py"
)
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)


def export(model, names, provenance):
    trees = []
    for estimator in model.estimators_:
        t = estimator.tree_
        trees.append(
            [
                [
                    int(t.children_left[i]),
                    int(t.children_right[i]),
                    int(t.feature[i]),
                    float(t.threshold[i]),
                    float(t.value[i, 0, 0]),
                ]
                for i in range(t.node_count)
            ]
        )
    return dict(
        schema=2,
        feature_names=names,
        trees=trees,
        provenance=provenance,
        target_schema="critical_end_after_all_rank_start.v1",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batches, rows = read(args.batches), read(args.forecasts)
    bx = np.stack([features(r) for r in batches])
    by = target(batches)
    fx, cx = arrays(rows)
    fy = target(rows)
    episodes = np.array([r["episode"] for r in rows])
    batch_episodes = np.array([r["episode"] for r in batches])
    oof = np.full(len(rows), np.nan)
    for _, test in GroupKFold(n_splits=3).split(fx, groups=episodes):
        train = np.flatnonzero(~np.isin(batch_episodes, episodes[test]))
        model = tree().fit(
            bx[train], by[train], sample_weight=weights([batches[i] for i in train])
        )
        oof[test] = model.predict(fx[test])
    assert np.all(np.isfinite(oof))
    base = tree().fit(bx, by, sample_weight=weights(batches))
    residual_y = fy - oof
    shift = max(1024.0, 1 - float(min(residual_y)))
    residual = tree(10).fit(
        np.column_stack((cx, oof)), residual_y + shift, sample_weight=weights(rows)
    )
    provenance = dict(
        training_samples=len(rows),
        cached_base_samples=len(batches),
        cross_fit="three episode folds",
        repeated_training_required=False,
        corpora={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.batches, args.forecasts)
        },
    )
    state = dict(
        schema=3,
        target_schema="critical_end_after_all_rank_start.v1",
        base=export(base, production.FEATURE_NAMES, provenance),
        residual=export(
            residual,
            production.FEATURE_NAMES
            + production.FORECAST_NAMES
            + production.BASE_PREDICTION_NAMES,
            provenance,
        ),
        residual_shift_ms=shift,
        provenance=provenance,
    )
    restored = production.LatencySurface(state)
    base_ms = base.predict(fx)
    expected = base_ms + residual.predict(np.column_stack((cx, base_ms))) - shift
    errors, timings = [], []
    for r, prediction in zip(rows, expected):
        data = dict(
            r,
            execution_snapshot=r["execution_ranks"],
            forecast_ns=r["forecast_time_ns"],
            forecast_load_features=r["forecast_features"][:10],
        )
        begin = time.perf_counter_ns()
        actual = restored.predict(data)
        timings.append(time.perf_counter_ns() - begin)
        errors.append(abs(actual - max(1, prediction) * 1e6))
    assert max(errors) < 3, max(errors)
    payload = json.loads(args.base_model.read_text())
    payload["state"].update(schema=2, latency_surface=state)
    payload["provenance"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload) + "\n")
    report = dict(
        max_equivalence_error_ns=max(errors),
        samples=len(rows),
        cached_base_samples=len(batches),
        mean_inference_us=float(np.mean(timings) / 1000),
        p95_inference_us=float(np.quantile(timings, 0.95) / 1000),
        bytes=args.output.stat().st_size,
    )
    args.output.with_suffix(".verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
