# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize past-only forecasts and observed preparation; no performance inference."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def median(values):
    return statistics.median(values) if values else None


def main(root):
    rows = []
    for path in sorted(root.glob("*/p*/*/analysis.json")):
        data = json.loads(path.read_text())
        samples = data.get("prediction_samples", [])
        if not data["accepted"] or not samples:
            continue
        prepared = [p for p in samples if p["actual_b_preparation_ms"] is not None]
        row = dict(
            phase=path.parents[2].name,
            workload=path.parents[1].name,
            policy=path.parent.name.split("_", 1)[1],
            compute_samples=len(samples),
            compute_median_ratio=median([p["ratio"] for p in samples]),
            compute_median_absolute_error_ms=median(
                [abs(p["predicted_ms"] - p["actual_ms"]) for p in samples]
            ),
            compute_median_absolute_percentage_error=median(
                [100 * abs(p["ratio"] - 1) for p in samples]
            ),
            preparation_samples=len(prepared),
            preparation_reserve_median_ms=median(
                [p["preparation_reserve_ms"] for p in prepared]
            ),
            preparation_actual_median_ms=median(
                [p["actual_b_preparation_ms"] for p in prepared]
            ),
            preparation_median_absolute_error_ms=median(
                [
                    abs(p["preparation_reserve_ms"] - p["actual_b_preparation_ms"])
                    for p in prepared
                ]
            ),
            preparation_underestimate_percent=(
                100
                * sum(
                    p["preparation_reserve_ms"] < p["actual_b_preparation_ms"]
                    for p in prepared
                )
                / len(prepared)
                if prepared
                else None
            ),
            missed_acked_candidates_ready_before_a_end_lower=data[
                "missed_candidates_ready_before_a_end_lower"
            ],
            exposure_lower_ms=data["exposed_lower_ms"].get("total", 0),
            adaptive_poll_cpu_ms=data["adaptive_poll_cpu_ms"],
            throughput=data["throughput"],
            mean_ttft_ms=data["mean_ttft_ms"],
        )
        rows.append(row)
    out = root / "summary"
    out.mkdir(exist_ok=True)
    (out / "prediction_diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (out / "prediction_diagnostics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(cells=len(rows))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
