# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Training-support fallback for launch-envelope suffix extrapolation."""

import argparse
import hashlib
import json
from pathlib import Path

import envelope as e
import envelope_evaluate as evaluate
import joblib
import numpy as np

ROOT = Path("/mnt/localnvme/glm53_envelope_guard_20261007")
POLICIES = {"suffix_guard": [5], "geometry_guard": [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12]}
NAMES = [f"{policy}_{name}" for policy in POLICIES for name in e.NAMES]


def bounds(rows, indices):
    x = e.features(rows)[:, indices]
    return dict(indices=indices, minimum=x.min(0).tolist(), maximum=x.max(0).tolist())


def outside(x, support):
    values = x[:, support["indices"]]
    return ((values < support["minimum"]) | (values > support["maximum"])).any(1)


class Frozen:
    def __init__(self, state, native=False):
        self.support = state["support"]
        self.envelope = e.Frozen(state["envelope"], native)
        self.fallback = e.ARCH.FrozenArchitecture(state["fallback"], native)

    def predict_one(self, x):
        if outside(np.asarray(x)[None, :], self.support)[0]:
            return self.fallback.predict_one(x[:23])
        return self.envelope.predict_one(x)


def apply(task):
    pool = joblib.load(task["dataset"])
    train, test = ([pool["forecasts"][i] for i in task[k]] for k in ["train", "test"])
    assert not {r["episode"] for r in train} & {r["episode"] for r in test}
    x = e.features(test)
    with np.load(e.ROOT / "folds" / task["name"] / "predictions.npz") as f:
        data = {k: f[k] for k in f.files}
    with np.load(e.DATA / "folds" / task["name"] / "predictions.npz") as f:
        np.testing.assert_array_equal(data["y"], f["y"])
        fallback = f["trees_fixed"].copy()
    audit = {}
    for policy, indices in POLICIES.items():
        support = bounds(train, indices)
        mask = outside(x, support)
        audit[policy] = dict(
            support=support,
            fallback_rows=int(mask.sum()),
            test_rows=len(test),
            training_episode_overlap=0,
        )
        for name in e.NAMES:
            data[f"{policy}_{name}"] = np.where(mask, fallback, data[name])
    directory = ROOT / "folds" / task["name"]
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "predictions.npz", **data)
    e.ARCH.dump(directory / "audit.json", audit)


def summarize(tasks):
    summary = {}
    for group in sorted(
        {f"{t['range_scope'] or 'episode'}:{t['regime']}" for t in tasks}
    ):
        members = [
            t
            for t in tasks
            if f"{t['range_scope'] or 'episode'}:{t['regime']}" == group
        ]
        data = evaluate.arrays(members)
        pieces = []
        for task in members:
            with np.load(ROOT / "folds" / task["name"] / "predictions.npz") as f:
                pieces.append({n: f[n] for n in NAMES})
        data.update({n: np.concatenate([p[n] for p in pieces]) for n in NAMES})
        old = e.NAMES
        try:
            e.NAMES = old + NAMES
            summary[group] = evaluate.scores(data)
        finally:
            e.NAMES = old
    cv = summary["episode:wide"]
    limit = cv["historical"]["trees_fixed"]["mae_ms"] * 1.1
    scopes = ["long_context", "long_suffix", "large_batch", "heavy_transfers"]

    def range_score(name):
        return float(
            np.mean(
                [
                    summary[f"{scope}:wide"]["fresh"][name]["mae_ms"]
                    / summary[f"{scope}:wide"]["fresh"]["trees_fixed"]["mae_ms"]
                    for scope in scopes
                ]
            )
        )

    eligible = [n for n in e.NAMES + NAMES if cv["historical"][n]["mae_ms"] <= limit]
    winner = min(
        eligible,
        key=lambda n: (cv["fresh"][n]["workload_macro_mae_ms"], range_score(n)),
    )
    e.ARCH.dump(ROOT / "validation_summary.json", summary)
    e.ARCH.dump(
        ROOT / "selection.json",
        dict(
            winner=winner,
            benchmark_used_for_selection=False,
            benchmark_is_reused=True,
            criterion=(
                "fresh episode macro MAE; exact ties by mean range MAE "
                "normalized to fixed trees; <=10% historical regression"
            ),
            range_score=range_score(winner),
        ),
    )
    print("selected", winner, "range_score", range_score(winner), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(
        (
            e.DATA / ("benchmark_tasks.json" if args.benchmark else "tasks.json")
        ).read_text()
    )
    checksum = (
        hashlib.sha256((ROOT / "selection.json").read_bytes()).hexdigest()
        if args.benchmark
        else None
    )
    for task in tasks:
        apply(task)
    if args.benchmark:
        assert (
            hashlib.sha256((ROOT / "selection.json").read_bytes()).hexdigest()
            == checksum
        )
        e.ARCH.dump(
            ROOT / "benchmark_opened.json",
            dict(selection_sha256=checksum, reused_benchmark=True),
        )
    else:
        summarize(tasks)


if __name__ == "__main__":
    main()
