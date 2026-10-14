# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Broaden a causal forecast using a group-cross-fitted cached execution base."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "experiments/predictor_corpus_20261006"))
from corpus import features, metrics  # noqa: E402


def read(path):
    return [
        r
        for line in path.read_text().splitlines()
        if (r := json.loads(line))
        and all(
            0 < e[2] <= e[3] and e[3] - e[2] <= 2_000_000 for e in r["execution_ranks"]
        )
    ]


def target(rows):
    return np.array(
        [
            (
                max(e[3] + e[6] for e in r["execution_ranks"])
                - max(e[3] for e in r["execution_ranks"])
            )
            / 1e6
            for r in rows
        ]
    )


def weights(rows):
    counts = Counter(r["workload"] for r in rows)
    result = np.array([1 / max(50, counts[r["workload"]]) for r in rows])
    return result * len(result) / sum(result)


def tree(leaf=5):
    return ExtraTreesRegressor(
        n_estimators=160, min_samples_leaf=leaf, max_depth=18, random_state=17, n_jobs=4
    )


def arrays(rows):
    base = np.stack([features(r) for r in rows])
    runtime = np.array([r["forecast_features"] for r in rows])
    return base, np.column_stack((base, runtime))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    batches, forecasts, fresh = map(read, (args.batches, args.forecasts, args.fresh))
    bx = np.stack([features(r) for r in batches])
    by = target(batches)
    fx, cx = arrays(forecasts)
    fresh_x, fresh_c = arrays(fresh)
    fy = target(forecasts)
    episodes = np.array([r["episode"] for r in forecasts])
    batch_episodes = np.array([r["episode"] for r in batches])
    workloads = np.array([r["workload"] for r in forecasts])
    splits = []
    for seed in [17, 29, 43]:
        train, test = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                fx, groups=episodes
            )
        )
        splits.append((f"episode_{seed}", train, test, False))
    old = np.array(["20261006" not in r["source"] for r in forecasts])
    splits.append(("temporal", np.flatnonzero(old), np.flatnonzero(~old), False))
    for w in sorted(set(workloads)):
        if sum(workloads == w) >= 100:
            splits.append(
                (
                    f"workload_{w}",
                    np.flatnonzero(workloads != w),
                    np.flatnonzero(workloads == w),
                    False,
                )
            )
    splits.append(("fresh", np.arange(len(forecasts)), None, True))
    reports = []
    for name, train, test, is_fresh in splits:
        held = set() if is_fresh else set(episodes[test])
        if name.startswith("workload_"):
            banned = name[len("workload_") :]
            bt = np.array([i for i, r in enumerate(batches) if r["workload"] != banned])
        elif name == "temporal":
            bt = np.array(
                [i for i, r in enumerate(batches) if "20261006" not in r["source"]]
            )
        else:
            bt = np.flatnonzero(~np.isin(batch_episodes, list(held)))
        train_rows = [forecasts[i] for i in train]
        tx, tc, ty = (
            (fresh_x, fresh_c, target(fresh))
            if is_fresh
            else (fx[test], cx[test], fy[test])
        )
        test_rows = fresh if is_fresh else [forecasts[i] for i in test]
        base = tree().fit(
            bx[bt], by[bt], sample_weight=weights([batches[i] for i in bt])
        )
        base_prediction = base.predict(tx)
        oof = np.full(len(train), np.nan)
        for inner_train, inner_test in GroupKFold(n_splits=3).split(
            fx[train], groups=episodes[train]
        ):
            inner_held = set(episodes[train[inner_test]])
            ib = bt[~np.isin(batch_episodes[bt], list(inner_held))]
            model = tree().fit(
                bx[ib], by[ib], sample_weight=weights([batches[i] for i in ib])
            )
            oof[inner_test] = model.predict(fx[train[inner_test]])
        assert np.all(np.isfinite(oof))
        candidates = {"wide_base": base_prediction}
        direct = tree().fit(cx[train], fy[train], sample_weight=weights(train_rows))
        candidates["direct_causal"] = direct.predict(tc)
        for kind in ["forest", "boost_mean", "boost_median"]:
            residual = (
                tree(10)
                if kind == "forest"
                else HistGradientBoostingRegressor(
                    max_iter=180,
                    max_leaf_nodes=15,
                    min_samples_leaf=15,
                    l2_regularization=10,
                    random_state=17,
                    loss="quantile" if kind == "boost_median" else "squared_error",
                    quantile=0.5 if kind == "boost_median" else None,
                )
            )
            residual.fit(
                np.column_stack((cx[train], oof)),
                fy[train] - oof,
                sample_weight=weights(train_rows),
            )
            candidates[f"base_plus_{kind}"] = base_prediction + residual.predict(
                np.column_stack((tc, base_prediction))
            )
        for label, prediction in candidates.items():
            prediction = np.maximum(1, prediction)
            error = prediction - ty
            report = dict(
                split=name,
                model=label,
                metrics=metrics(error),
                per_workload={
                    w: metrics(error[np.array([r["workload"] == w for r in test_rows])])
                    for w in sorted({r["workload"] for r in test_rows})
                },
            )
            reports.append(report)
            print(name, label, report["metrics"], flush=True)
        (args.output / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
