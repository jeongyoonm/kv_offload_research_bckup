# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize matched C/D serving results and measured compute gaps."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def union(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return merged


def diagnostics(directory):
    events = [
        json.loads(line)
        for line in (directory / "engine.jsonl").read_text().splitlines()
    ]
    completed = [
        e for e in events if e["event"] == "step_completion" and e.get("selected")
    ]
    assert all(e.get("valid") and len(e["execution_ranks"]) == 8 for e in completed)
    intervals = union(
        [
            (
                max(r[3] for r in e["execution_ranks"]),
                max(r[3] + r[6] for r in e["execution_ranks"]),
            )
            for e in completed
        ]
    )
    gap_ms = sum(b[0] - a[1] for a, b in zip(intervals, intervals[1:])) / 1e6
    seals = [
        e
        for e in json.loads((directory / "late.json").read_text())["records"]
        if e["event"] == "membership_sealed"
    ]
    transports = [e["transport"] for e in seals]
    waits = union(
        [
            (t["base_ready_ns"], t["seal_ns"])
            for t in transports
            if t.get("base_ready_ns") is not None
        ]
    )
    exposed = 0
    for begin, end in waits:
        overlap = sum(max(0, min(end, b) - max(begin, a)) for a, b in intervals)
        exposed += end - begin - overlap
    return dict(
        inter_compute_gap_ms=gap_ms,
        hold_without_compute_ms=exposed / 1e6,
        seal_reasons=dict(
            Counter(t.get("seal_reason", "ordinary_or_fallback") for t in transports)
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"]
    raw = summary["results"]
    for result in raw:
        assert result["accepted"] and result["frozen_unchanged"]
        directory = (
            root
            / "live"
            / result["workload"]
            / f"r{result['repeat']}_{result['label']}"
        )
        result.update(diagnostics(directory))
        if result["label"] == "C":
            assert result["late_admissions"] == 0
    metrics = (
        "throughput_rps",
        "mean_ttft_ms",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "compute_rounds",
        "late_admissions",
        "kv_ready_during_compute",
        "recovered_compute_opportunities",
        "inter_compute_gap_ms",
        "hold_without_compute_ms",
    )
    levels, comparisons = [], []
    for workload in ("short16k", "familiar60k", "heterogeneous"):
        arms = {
            arm: sorted(
                [r for r in raw if r["workload"] == workload and r["label"] == arm],
                key=lambda r: r["repeat"],
            )
            for arm in ("C", "D")
        }
        assert [r["seed"] for r in arms["C"]] == [r["seed"] for r in arms["D"]]
        for arm in ("C", "D"):
            levels.append(
                dict(
                    workload=workload,
                    arm=arm,
                    repeats=len(arms[arm]),
                    **{m: statistics.mean(r[m] for r in arms[arm]) for m in metrics},
                )
            )
        pairs = []
        for c, d in zip(arms["C"], arms["D"]):
            pairs.append(
                dict(
                    repeat=c["repeat"],
                    **{
                        m: 100 * (d[m] / c[m] - 1)
                        for m in ("throughput_rps", "mean_ttft_ms", "p95_ttft_ms")
                    },
                )
            )
        comparisons.append(
            dict(
                workload=workload,
                pairs=pairs,
                **{
                    m + "_change_percent": 100
                    * (
                        math.exp(
                            statistics.mean(
                                math.log(d[m] / c[m])
                                for c, d in zip(arms["C"], arms["D"])
                            )
                        )
                        - 1
                    )
                    for m in ("throughput_rps", "mean_ttft_ms", "p95_ttft_ms")
                },
            )
        )
    (root / "comparison.json").write_text(
        json.dumps(
            dict(
                levels=levels,
                comparisons=comparisons,
                episodes=raw,
                note="Absolute levels are arithmetic episode means; changes use "
                "paired geometric ratios. p95 is mean episode p95, not pooled. "
                "Inter-compute gaps exclude initial/final idle. Hold without compute "
                "is base-ready-to-seal wall time outside measured compute envelopes; "
                "it includes load-only waiting and is not an independently "
                "attributable throughput cost.",
            ),
            indent=2,
        )
        + "\n"
    )
    with (root / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(levels[0]))
        writer.writeheader()
        writer.writerows(levels)
    print(json.dumps(dict(levels=levels, comparisons=comparisons), indent=2))


if __name__ == "__main__":
    main()
