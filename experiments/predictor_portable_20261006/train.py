# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fit once, export JSON trees, and verify production inference equivalence."""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "experiments/predictor_corpus_20261006"))
from corpus import features  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "latency_surface", SOURCE / "vllm/v1/kv_offload/latency_surface.py"
)
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["shape", "dispatch"], default="dispatch")
    parser.add_argument("--leaf", type=int, default=5)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    x = np.stack([features(r) for r in rows])
    if args.mode == "shape":
        x = x[:, :9]
    y = np.array([r["duration_ms"] for r in rows])
    counts = Counter(r["workload"] for r in rows)
    weight = np.array([1 / max(counts[r["workload"]], 50) for r in rows])
    weight *= len(weight) / sum(weight)
    model = ExtraTreesRegressor(
        n_estimators=160,
        min_samples_leaf=args.leaf,
        max_depth=18,
        random_state=17,
        n_jobs=4,
    ).fit(x, y, sample_weight=weight)
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        trees.append(
            [
                [
                    int(tree.children_left[i]),
                    int(tree.children_right[i]),
                    int(tree.feature[i]),
                    float(tree.threshold[i]),
                    float(tree.value[i, 0, 0]),
                ]
                for i in range(tree.node_count)
            ]
        )
    surface = dict(
        schema=1,
        feature_names=production.FEATURE_NAMES[: x.shape[1]],
        trees=trees,
        provenance=dict(
            corpus=str(args.corpus),
            corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
            training_samples=len(rows),
            workloads=dict(counts),
            sklearn_version=sklearn.__version__,
            seed=17,
            min_samples_leaf=args.leaf,
            features_available_at="dispatch",
            repeated_training_required=False,
            target="slowest rank CUDA event span after input preparation",
        ),
    )
    restored = production.LatencySurface(json.loads(json.dumps(surface)))
    timings = []
    errors = []
    reference = model.predict(x)
    for row, expected in zip(rows, reference):
        begin = time.perf_counter_ns()
        actual = restored.predict(row)
        timings.append(time.perf_counter_ns() - begin)
        errors.append(abs(actual - expected * 1e6))
    assert max(errors) <= 2, f"Production prediction differs by {max(errors)}ns"
    base = json.loads(args.base_model.read_text())
    base["state"].update(schema=2, latency_surface=surface)
    base["provenance"] = dict(
        training_source=str(args.corpus),
        mode=args.mode,
        frozen_model=True,
        base_snapshot=str(args.base_model),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(base) + "\n")
    report = dict(
        mode=args.mode,
        model=str(args.output),
        samples=len(rows),
        max_equivalence_error_ns=max(errors),
        latency_us=dict(
            mean=float(np.mean(timings) / 1000),
            median=float(np.median(timings) / 1000),
            p95=float(np.quantile(timings, 0.95) / 1000),
            max=float(max(timings) / 1000),
        ),
        total_nodes=sum(len(t) for t in trees),
        json_bytes=args.output.stat().st_size,
    )
    args.output.with_suffix(".verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
