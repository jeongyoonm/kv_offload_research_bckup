# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ablate causal transfer state and robust loss on identical forecast splits."""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "experiments/predictor_corpus_20261006"))
from corpus import features, metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    shape = np.stack([features(r) for r in rows])
    runtime = np.array([r["forecast_features"] for r in rows])
    y = np.array([r["duration_ms"] for r in rows])
    episodes = np.array([r["episode"] for r in rows])
    workloads = np.array([r["workload"] for r in rows])
    datasets = {
        "shape": shape[:, :9],
        "dispatch_loads": shape,
        "current_loads": np.column_stack((shape, runtime[:, :2])),
        "current_load_age": np.column_stack((shape, runtime[:, :10])),
        "causal_full": np.column_stack((shape, runtime)),
    }
    splits = []
    for seed in [17, 29, 43]:
        train, test = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                shape, groups=episodes
            )
        )
        splits.append((f"episode_seed{seed}", train, test))
    for w in sorted(set(workloads)):
        splits.append(
            (
                f"workload_{w}",
                np.flatnonzero(workloads != w),
                np.flatnonzero(workloads == w),
            )
        )
    old = np.array(["20261006" not in r["source"] for r in rows])
    splits.append(("temporal_oct6", np.flatnonzero(old), np.flatnonzero(~old)))
    results = []
    for name, train, test in splits:
        counts = Counter(workloads[train])
        weight = np.array([1 / counts[w] for w in workloads[train]])
        weight *= len(weight) / sum(weight)
        prior = np.array([rows[i]["prior_prediction_ms"] for i in test])
        results.append(
            dict(split=name, model="issued_prior", metrics=metrics(prior - y[test]))
        )
        for variant, x in datasets.items():
            for kind in ["extra_trees", "boost_mean", "boost_median"]:
                if kind == "extra_trees":
                    model = ExtraTreesRegressor(
                        n_estimators=160,
                        min_samples_leaf=5,
                        max_depth=18,
                        random_state=17,
                        n_jobs=4,
                    )
                else:
                    model = HistGradientBoostingRegressor(
                        max_iter=220,
                        max_leaf_nodes=15,
                        min_samples_leaf=15,
                        l2_regularization=10,
                        random_state=17,
                        loss="quantile" if kind == "boost_median" else "squared_error",
                        quantile=0.5 if kind == "boost_median" else None,
                    )
                begin = time.monotonic()
                model.fit(x[train], y[train], sample_weight=weight)
                prediction = np.maximum(1, model.predict(x[test]))
                result = dict(
                    split=name,
                    model=f"{variant}_{kind}",
                    metrics=metrics(prediction - y[test]),
                    elapsed_seconds=time.monotonic() - begin,
                    per_workload={
                        w: metrics((prediction - y[test])[workloads[test] == w])
                        for w in sorted(set(workloads[test]))
                    },
                )
                results.append(result)
                print(name, result["model"], result["metrics"], flush=True)
            (args.output / "reports.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
