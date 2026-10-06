# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare latency surfaces using episode, workload, and temporal holdouts."""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "experiments/predictor_corpus_20261006"))
from corpus import FEATURE_NAMES, features, metrics  # noqa: E402


def physics(x):
    return np.column_stack(
        (np.ones(len(x)), x[:, 0] / 8192, x[:, 2] / 1e8, x[:, 1] / 8)
    )


class PhysicalModel:
    def fit(self, x, y, sample_weight=None):
        p = physics(x)
        w = np.sqrt(sample_weight) if sample_weight is not None else np.ones(len(y))
        self.coefficients = nnls(p * w[:, None], y * w)[0]
        return self

    def predict(self, x):
        return np.maximum(1, physics(x) @ self.coefficients)


class RatioSurface:
    def __init__(self, runtime=False, neighbors=32):
        self.runtime = runtime
        self.neighbors = neighbors

    def transform(self, x):
        indices = [0, 1, 2, 5, 7] + ([9, 10] if self.runtime else [])
        return np.log1p(x[:, indices])

    def fit(self, x, y, sample_weight=None):
        self.base = PhysicalModel().fit(x, y, sample_weight)
        self.surface = KNeighborsRegressor(
            n_neighbors=min(self.neighbors, len(y)), weights="distance", n_jobs=4
        )
        self.surface.fit(self.transform(x), np.log(y / self.base.predict(x)))
        return self

    def predict(self, x):
        query = self.transform(x)
        distance = self.surface.kneighbors(query, n_neighbors=1)[0][:, 0]
        ratio = self.surface.predict(query)
        # Taper correction outside the measured shape neighborhood.
        factor = np.maximum(0, 1 - distance / 1.5)
        return self.base.predict(x) * np.exp(ratio * factor)


class ResidualTrees:
    def __init__(self, runtime=False):
        self.runtime = runtime

    def fit(self, x, y, sample_weight=None):
        self.base = PhysicalModel().fit(x, y, sample_weight)
        self.trees = ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=5,
            max_depth=18,
            random_state=17,
            n_jobs=4,
        )
        self.trees.fit(
            x if self.runtime else x[:, :9],
            np.log(y / self.base.predict(x)),
            sample_weight=sample_weight,
        )
        return self

    def predict(self, x):
        return self.base.predict(x) * np.exp(
            self.trees.predict(x if self.runtime else x[:, :9])
        )


def models():
    return {
        "physical_nnls": PhysicalModel(),
        "shape_knn32": RatioSurface(),
        "shape_load_knn32": RatioSurface(runtime=True),
        "shape_residual_trees": ResidualTrees(),
        "shape_load_residual_trees": ResidualTrees(runtime=True),
        "shape_trees": ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=5,
            max_depth=18,
            random_state=17,
            n_jobs=4,
        ),
        "shape_load_trees": ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=5,
            max_depth=18,
            random_state=17,
            n_jobs=4,
        ),
        "shape_load_boosting": HistGradientBoostingRegressor(
            max_iter=220,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=10,
            random_state=17,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    x, y = (
        np.stack([features(r) for r in rows]),
        np.array([r["duration_ms"] for r in rows]),
    )
    workloads = np.array([r["workload"] for r in rows])
    episodes = np.array([r["episode"] for r in rows])
    sources = np.array([r["source"] for r in rows])
    splits = []
    for seed in [17, 29, 43]:
        train, test = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                x, groups=episodes
            )
        )
        splits.append((f"episode_seed{seed}", train, test))
    for workload in sorted(set(workloads)):
        splits.append(
            (
                f"workload_{workload}",
                np.flatnonzero(workloads != workload),
                np.flatnonzero(workloads == workload),
            )
        )
    old = np.array(["20261006" not in r["source"] for r in rows])
    splits.append(("temporal_oct6", np.flatnonzero(old), np.flatnonzero(~old)))
    reports, predictions = [], []
    for name, train, test in splits:
        counts = Counter(workloads[train])
        weight = np.array([1 / counts[w] for w in workloads[train]])
        weight *= len(weight) / sum(weight)
        for model_name, model in models().items():
            begin = time.monotonic()
            shape_only = model_name == "shape_trees"
            train_x = x[train, :9] if shape_only else x[train]
            test_x = x[test, :9] if shape_only else x[test]
            if isinstance(model, RatioSurface):
                model.fit(train_x, y[train], sample_weight=weight)
            else:
                model.fit(train_x, y[train], sample_weight=weight)
            prediction = np.maximum(1, model.predict(test_x))
            error = prediction - y[test]
            report = dict(
                split=name,
                model=model_name,
                train_samples=len(train),
                train_episodes=len(set(episodes[train])),
                metrics=metrics(error),
                elapsed_seconds=time.monotonic() - begin,
                per_workload={},
                per_source={},
            )
            for key, group in [("per_workload", workloads), ("per_source", sources)]:
                for value in sorted(set(group[test])):
                    mask = group[test] == value
                    report[key][value] = metrics(error[mask])
            reports.append(report)
            predictions.extend(
                dict(
                    split=name,
                    model=model_name,
                    index=int(i),
                    prediction_ms=float(p),
                    actual_ms=float(y[i]),
                )
                for i, p in zip(test, prediction)
            )
            (args.output / "reports.json").write_text(
                json.dumps(reports, indent=2) + "\n"
            )
            print(name, model_name, report["metrics"], flush=True)
    with (args.output / "predictions.jsonl").open("w") as stream:
        for p in predictions:
            stream.write(json.dumps(p) + "\n")
    noise = defaultdict(list)
    for row in rows:
        signature = tuple(tuple(s) for s in row["request_shapes"])
        noise[row["source"], signature, tuple(row["load_state"])].append(
            row["duration_ms"]
        )
    repeated = [np.array(v) for v in noise.values() if len(v) >= 5]
    (args.output / "noise.json").write_text(
        json.dumps(
            dict(
                feature_names=FEATURE_NAMES,
                repeated_signatures=len(repeated),
                repeated_samples=sum(map(len, repeated)),
                retrospective_median_oracle=metrics(
                    np.concatenate([np.median(v) - v for v in repeated])
                ),
                note="Within-signature median is retrospective, not a forecast.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
