# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed equal blends of complementary, support-guarded predictors."""

import argparse
import hashlib
import json
from pathlib import Path

import envelope as e
import envelope_evaluate as evaluate
import envelope_guard as guard
import numpy as np
import report

ROOT = Path("/mnt/localnvme/glm53_envelope_blend_20261007")
SOURCES = {
    "envelope_sparse_mlp": "guarded_sparse_mlp",
    "envelope_sparse_blend": "guarded_sparse_blend",
}


class Frozen:
    def __init__(self, state, native=False):
        self.envelope = guard.Frozen(state["envelope"], native)
        self.neural = e.improve.FrozenCandidate(state["neural"], native)

    def predict_one(self, x):
        # The neural models use sparse features at positions 28:31; neither
        # requires the five unused per-rank enqueue features at positions 23:28.
        neural_x = np.r_[x[:23], np.zeros(5), x[23:26]]
        return (self.envelope.predict_one(x) + self.neural.predict_one(neural_x)) / 2


def arrays(tasks):
    data = evaluate.arrays(tasks)
    envelope = []
    for task in tasks:
        with np.load(guard.ROOT / "folds" / task["name"] / "predictions.npz") as f:
            envelope.append(f["suffix_guard_envelope_boost"].copy())
    data["suffix_guard_envelope_boost"] = np.concatenate(envelope)
    for name, source in SOURCES.items():
        data[name] = (data["suffix_guard_envelope_boost"] + data[source]) / 2
    return data


def summarize(tasks):
    result = {}
    for group in sorted(
        {f"{t['range_scope'] or 'episode'}:{t['regime']}" for t in tasks}
    ):
        members = [
            t
            for t in tasks
            if f"{t['range_scope'] or 'episode'}:{t['regime']}" == group
        ]
        data = arrays(members)
        old = e.NAMES
        try:
            e.NAMES = old + list(SOURCES) + ["suffix_guard_envelope_boost"]
            metrics = evaluate.scores(data)
        finally:
            e.NAMES = old
        paired = {}
        for baseline in ["trees_fixed", "guarded_sparse_mlp", "guarded_sparse_blend"]:
            reference = dict(data, trees_fixed=data[baseline])
            paired[baseline] = {
                n: report.paired_interval(
                    reference, n, data["phase"] == "fresh", macro=True
                )
                for n in SOURCES
            }
        result[group] = dict(scores=metrics, paired=paired)
    return result


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
    result = summarize(tasks)
    if args.benchmark:
        assert (
            hashlib.sha256((ROOT / "selection.json").read_bytes()).hexdigest()
            == checksum
        )
        e.ARCH.dump(
            ROOT / "benchmark_opened.json",
            dict(selection_sha256=checksum, benchmark_is_reused=True),
        )
        e.ARCH.dump(ROOT / "benchmark_summary.json", result)
    else:
        cv = result["episode:wide"]["scores"]
        eligible = [
            n
            for n in SOURCES
            if cv["historical"][n]["mae_ms"]
            <= cv["historical"]["trees_fixed"]["mae_ms"] * 1.1
        ]
        winner = min(eligible, key=lambda n: cv["fresh"][n]["workload_macro_mae_ms"])
        e.ARCH.dump(
            ROOT / "selection.json",
            dict(
                winner=winner,
                criterion="fresh episode macro MAE; <=10% historical regression",
                weights=[0.5, 0.5],
                benchmark_used_for_selection=False,
                benchmark_is_reused=True,
            ),
        )
        e.ARCH.dump(ROOT / "validation_summary.json", result)
        for n in SOURCES:
            print(
                n,
                "fresh",
                round(cv["fresh"][n]["mae_ms"], 3),
                "macro",
                round(cv["fresh"][n]["workload_macro_mae_ms"], 3),
                "historical",
                round(cv["historical"][n]["mae_ms"], 3),
                flush=True,
            )
        print("selected", winner, flush=True)


if __name__ == "__main__":
    main()
