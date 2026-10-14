# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare three policies on matched 80-request, three-repeat TP8 cells."""

import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    cells = json.loads((args.root / "cells.json").read_text())
    pairs = {}
    for cell in cells:
        assert cell["workload"] == "p61440_f95_h100"
        assert cell["accepted"] and not cell["gpu_hits"] and not cell["preemptions"]
        assert not cell["missing_execution_timing_steps"]
        assert sum(cell["cached_distribution"].values()) == 80
        repeat = Path(cell["path"]).name.split("_", 1)[0]
        pair = pairs.setdefault(repeat, {})
        assert cell["mode"] not in pair
        pair[cell["mode"]] = cell
    assert len(pairs) == 3
    for pair in pairs.values():
        assert set(pair) == {"current", "cached_pinned", "compact_pinned"}
        current = pair["current"]
        for cell in pair.values():
            assert cell["h2d_bytes"] == current["h2d_bytes"]
            assert cell["workload_manifest"] == current["workload_manifest"]

    def interval(values):
        logs = [math.log(v) for v in values]
        mean = statistics.mean(logs)
        # Two-sided Student t 95% interval, three paired runs, two df.
        radius = 4.302652729749462 * statistics.stdev(logs) / math.sqrt(3)
        return dict(
            geometric_mean_ratio=math.exp(mean),
            lower_95=math.exp(mean - radius),
            upper_95=math.exp(mean + radius),
        )

    comparisons = []
    for reference, candidate in (
        ("current", "cached_pinned"),
        ("current", "compact_pinned"),
        ("cached_pinned", "compact_pinned"),
    ):
        rows = []
        for repeat, pair in sorted(pairs.items()):
            base, new = pair[reference], pair[candidate]
            rows.append(
                dict(
                    repeat=repeat,
                    throughput_ratio=new["throughput"] / base["throughput"],
                    mean_ttft_ratio=new["mean_ttft_ms"] / base["mean_ttft_ms"],
                    p95_ttft_ratio=new["p95_ttft_ms"] / base["p95_ttft_ms"],
                    reference_req_s=base["throughput"],
                    candidate_req_s=new["throughput"],
                    reference_compute_calls=base["compute_calls"],
                    candidate_compute_calls=new["compute_calls"],
                )
            )
        comparisons.append(
            dict(
                reference=reference,
                candidate=candidate,
                paired_runs=rows,
                throughput=interval([r["throughput_ratio"] for r in rows]),
                mean_ttft=interval([r["mean_ttft_ratio"] for r in rows]),
                p95_ttft=interval([r["p95_ttft_ratio"] for r in rows]),
            )
        )
    result = dict(
        comparisons=comparisons,
        scope="Three warmed alternating policy-order runs on one workload and "
        "one server with a common frozen predictor. Current is the existing "
        "full prototype, not vanilla vLLM. Confidence intervals describe "
        "run variation assuming independent approximately normal log ratios.",
    )
    (args.root / "compact_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
