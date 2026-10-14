# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize calibration and matched small serving validation separately."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from experiments.h2d_interference_20261009.analyze import paired_gain


def calibration_summary(data):
    cells = data["cells"]
    index = {
        (r["mode"], r["shape"], r["profile"], r["chunk_mib"], r["repeat"]): r
        for r in cells
    }
    concurrent = [r for r in cells if r["mode"] == "concurrent"]
    geometries = sorted({(r["shape"], r["profile"]) for r in concurrent})
    repeats = sorted({r["repeat"] for r in concurrent})
    configs = sorted({r["chunk_mib"] for r in concurrent})
    summaries = []
    for chunk in configs:
        record = dict(chunk_mib=chunk, depth=2)
        for metric in ("joint_finish_ms", "forward_enqueue_ms", "first_ready_ms"):
            # Average nine log ratios within each repeat before estimating its
            # interval, keeping shared all-rank/shape observations together.
            ratios = [
                math.exp(
                    statistics.mean(
                        math.log(
                            index["concurrent", s, p, 128, repeat][metric]
                            / index["concurrent", s, p, chunk, repeat][metric]
                        )
                        for s, p in geometries
                    )
                )
                for repeat in repeats
            ]
            record[metric] = paired_gain([(r, 1) for r in ratios])
        summaries.append(record)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    confirmation = json.loads((root / "confirmation/comparison.json").read_text())
    calibration = calibration_summary(confirmation)
    serving = []
    for chunk in (128, 8, 16):
        stage = root / f"serving_{chunk}mib"
        if not (stage / "summary.json").exists():
            continue
        summary = json.loads((stage / "summary.json").read_text())
        assert summary["complete"] and summary["stopped"]
        assert all(r["accepted"] for r in summary["results"])
        serving.extend(dict(chunk_mib=chunk, **r) for r in summary["results"])
    serving_rows = []
    for workload in sorted({r["workload"] for r in serving}):
        baseline = {
            r["repeat"]: r
            for r in serving
            if r["workload"] == workload and r["chunk_mib"] == 128
        }
        for chunk in (128, 8, 16):
            rows = [
                r
                for r in serving
                if r["workload"] == workload and r["chunk_mib"] == chunk
            ]
            if not rows:
                continue
            record = dict(
                workload=workload,
                chunk_mib=chunk,
                depth=2,
                repeats=len(rows),
                requests=sum(r["requests"] for r in rows),
            )
            for metric in (
                "throughput_rps",
                "mean_ttft_ms",
                "p50_ttft_ms",
                "p95_ttft_ms",
                "compute_rounds",
                "late_admissions",
            ):
                record[metric] = statistics.mean(r[metric] for r in rows)
                if baseline:
                    record[metric + "_change_percent"] = (
                        100
                        * math.expm1(
                            statistics.mean(
                                math.log(r[metric] / baseline[r["repeat"]][metric])
                                for r in rows
                                if r[metric] > 0 and baseline[r["repeat"]][metric] > 0
                            )
                        )
                        if all(
                            r[metric] > 0 and baseline[r["repeat"]][metric] > 0
                            for r in rows
                        )
                        else None
                    )
            serving_rows.append(record)
    out = dict(
        calibration=calibration,
        serving=serving_rows,
        scope="Calibration: equal-weight nine fixed geometries, eight paired repeats. "
        "Serving: two matched reset/primed 32-request episodes/workload/config; "
        "separate allocations, not a continuous 300-request evaluation.",
        uncertainty="Calibration intervals are nominal, unadjusted for multiple "
        "comparisons. Serving has only two repeats and allocation/order confounds; "
        "descriptive gains are not statistical equivalence or universal speedups.",
    )
    (root / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    if serving_rows:
        with (root / "serving_comparison.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(serving_rows[0]))
            writer.writeheader()
            writer.writerows(serving_rows)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
