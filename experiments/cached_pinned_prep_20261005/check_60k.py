# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report a predeclared six-pair throughput noninferiority check."""

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
        assert cell["accepted"] and not cell["preemptions"] and not cell["gpu_hits"]
        assert not cell["missing_execution_timing_steps"]
        repeat = Path(cell["path"]).name.split("_", 1)[0]
        modes = pairs.setdefault(repeat, {})
        assert cell["mode"] not in modes, (repeat, cell["mode"])
        modes[cell["mode"]] = cell
    assert len(pairs) == 6
    rows = []
    for repeat, pair in sorted(pairs.items()):
        assert set(pair) == {"current", "cached_pinned"}
        base, new = pair["current"], pair["cached_pinned"]
        assert base["h2d_bytes"] == new["h2d_bytes"]
        assert base["workload_manifest"] == new["workload_manifest"]
        assert sum(base["cached_distribution"].values()) == 80
        assert sum(new["cached_distribution"].values()) == 80
        rows.append(
            dict(
                repeat=repeat,
                current_req_s=base["throughput"],
                cached_pinned_req_s=new["throughput"],
                throughput_ratio=new["throughput"] / base["throughput"],
                mean_ttft_ratio=new["mean_ttft_ms"] / base["mean_ttft_ms"],
                current_compute_calls=base["compute_calls"],
                cached_pinned_compute_calls=new["compute_calls"],
            )
        )

    # Two-sided 95% Student t interval, six paired log ratios, five df.
    # The interval assumes independent approximately normal run-level log ratios.
    def interval(key):
        logs = [math.log(row[key]) for row in rows]
        mean = statistics.mean(logs)
        radius = 2.570581835636314 * statistics.stdev(logs) / math.sqrt(6)
        return dict(
            geometric_mean_ratio=math.exp(mean),
            lower_95=math.exp(mean - radius),
            upper_95=math.exp(mean + radius),
        )

    throughput = interval("throughput_ratio")
    report = dict(
        paired_runs=rows,
        throughput=throughput,
        mean_ttft=interval("mean_ttft_ratio"),
        noninferiority_margin_percent=2,
        excludes_more_than_two_percent_slowdown=throughput["lower_95"] > 0.98,
        interval_excludes_any_slowdown=throughput["lower_95"] >= 1,
        interval_shows_slowdown=throughput["upper_95"] < 1,
        scope="One 60K workload, concurrency 32, same server, fixed predictor; "
        "confidence interval describes run variation, "
        "not all workloads or deployments.",
    )
    (args.root / "noninferiority.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
