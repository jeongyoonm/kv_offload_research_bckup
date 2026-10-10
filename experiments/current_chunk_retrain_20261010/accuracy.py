# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate prospective live forecasts from predictions after GPU completion."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def metrics(rows):
    if not rows:
        return dict(samples=0)
    error = np.array([r["error_ms"] for r in rows])
    truth = np.array([r["truth_ms"] for r in rows])
    return dict(
        samples=len(rows),
        mae_ms=float(np.mean(np.abs(error))),
        mape_percent=float(np.mean(np.abs(error) / truth) * 100),
        bias_ms=float(np.mean(error)),
        p95_absolute_ms=float(np.quantile(np.abs(error), 0.95)),
        under_gt5_percent=float(np.mean(error < -5) * 100),
        over_gt5_percent=float(np.mean(error > 5) * 100),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"]
    rows, rejected = [], Counter()
    for result in summary["results"]:
        directory = (
            args.root
            / "live"
            / result["workload"]
            / f"r{result['repeat']}_{result['label']}"
        )
        events = [
            json.loads(line)
            for line in (directory / "engine.jsonl").read_text().splitlines()
        ]
        computes = {
            e["step"]: e
            for e in events
            if e["event"] == "step_completion" and e.get("selected")
        }
        steps = sorted(computes)
        following = dict(zip(steps, steps[1:]))
        records = json.loads((directory / "late.json").read_text())["records"]
        seen = set()
        for point in records:
            if point["event"] != "seal_prediction":
                continue
            step = point["predecessor_step"]
            if step in seen or step not in computes:
                continue
            seen.add(step)
            done = computes[step]
            ranks = done.get("execution_ranks", [])
            if (
                not done.get("valid")
                or len(ranks) != 8
                or any(
                    not 0 < r[2] <= r[3] or r[3] - r[2] > 2000000 or r[6] <= 0
                    for r in ranks
                )
            ):
                rejected["invalid_clock_or_duration"] += 1
                continue
            start = max(r[3] for r in ranks)
            end = max(r[3] + r[6] for r in ranks)
            when = point.get("forecast_ns", point["time_ns"])
            if when < start:
                rejected["forecast_before_all_rank_start"] += 1
                continue
            truth = (end - start) / 1e6
            predicted = point["predicted_duration_ns"] / 1e6
            rows.append(
                dict(
                    workload=result["workload"],
                    arm=result["label"],
                    repeat=result["repeat"],
                    seed=result["seed"],
                    predecessor=step,
                    target_step=point["step"],
                    truth_ms=truth,
                    prediction_ms=predicted,
                    error_ms=predicted - truth,
                    prospective=when < end,
                    elapsed_ms=(when - start) / 1e6,
                    actual_remaining_ms=(end - when) / 1e6,
                    adjacent_compute=following.get(step) == point["step"],
                    snapshot_start_difference_ms=(point["gpu_start_upper_ns"] - start)
                    / 1e6,
                    features=point["predictor_input"],
                    shapes=done["request_shapes"],
                )
            )
    groups = defaultdict(list)
    for row in rows:
        groups[(row["workload"], row["arm"])].append(row)
    levels = []
    for (workload, arm), selected in sorted(groups.items()):
        prospective = [r for r in selected if r["prospective"]]
        levels.append(
            dict(
                workload=workload,
                arm=arm,
                all=metrics(selected),
                prospective=metrics(prospective),
                prospective_adjacent=metrics(
                    [r for r in prospective if r["adjacent_compute"]]
                ),
                after_gpu_completion=metrics(
                    [r for r in selected if not r["prospective"]]
                ),
                max_snapshot_start_difference_ms=max(
                    abs(r["snapshot_start_difference_ms"]) for r in selected
                ),
            )
        )
    (args.root / "live_accuracy.json").write_text(
        json.dumps(
            dict(
                levels=levels,
                rejected=dict(rejected),
                definition="First forecast per preceding compute; target is critical "
                "end after all-rank GPU start. Prospective requires forecast before "
                "the observed GPU execution end. Serving correction included.",
            ),
            indent=2,
        )
        + "\n"
    )
    (args.root / "live_accuracy_rows.json.gz").write_bytes(
        gzip.compress(
            (json.dumps(rows, separators=(",", ":")) + "\n").encode(), mtime=0
        )
    )
    print(json.dumps(levels, indent=2), flush=True)


if __name__ == "__main__":
    main()
