# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconstruct only KV state observable when each historical forecast issued."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    batches = defaultdict(dict)
    for line in args.corpus.read_text().splitlines():
        row = json.loads(line)
        batches[row["source_file"]][row["step"]] = row
    output, rejected = [], Counter()
    for name, by_step in batches.items():
        path = Path(name)
        late_path = path.parent / "late.json"
        if not late_path.exists():
            continue
        late = json.loads(late_path.read_text())
        if late.get("overflow"):
            rejected["late_overflow"] += 1
            continue
        events = [json.loads(line) for line in path.read_text().splitlines()]
        timeline = []
        for e in events:
            if e["event"] == "kv_load_dispatched":
                timeline.append((e["dispatch_ns"], 0, e))
            elif e["event"] == "kv_all_rank_completion":
                # Engine receipt time, rather than an earlier device completion.
                timeline.append((e["time_ns"], 1, e))
        for e in late["records"]:
            if e["event"] == "seal_prediction":
                timeline.append((e["time_ns"], 2, e))
        jobs, dispatched, completed, seen = {}, {}, {}, set()
        for now, kind, event in sorted(timeline, key=lambda e: e[:2]):
            if kind == 0:
                job = (now, event["external_tokens"])
                jobs[event["job_id"]] = job
                dispatched[event["job_id"]] = job
                continue
            if kind == 1:
                jobs.pop(event["job_id"], None)
                completed[event["job_id"]] = now
                continue
            step = event["predecessor_step"]
            if step not in by_step or step in seen:
                rejected["not_cached_or_duplicate"] += 1
                continue
            if "gpu_start_upper_ns" not in event:
                rejected["older_forecast_target_semantics"] += 1
                continue
            seen.add(step)
            row = dict(by_step[step])
            if now < event["gpu_start_upper_ns"]:
                rejected["negative_forecast_elapsed"] += 1
                continue
            weights = np.array([j[1] for j in jobs.values()], dtype=float)
            ages = np.array([(now - j[0]) / 1e6 for j in jobs.values()], dtype=float)
            old = [j for j in jobs.values() if j[0] <= row["dispatch_ns"]]
            new = [j for j in jobs.values() if j[0] > row["dispatch_ns"]]
            completed_tokens = sum(
                tokens
                for job_id, (when, tokens) in dispatched.items()
                if when <= row["dispatch_ns"]
                and completed.get(job_id, 0) > row["dispatch_ns"]
            )
            row["forecast_features"] = [
                len(jobs),
                float(sum(weights)),
                float(np.min(ages)) if len(ages) else 0.0,
                float(np.max(ages)) if len(ages) else 0.0,
                float(np.average(ages, weights=weights)) if len(ages) else 0.0,
                len(old),
                sum(j[1] for j in old),
                len(new),
                sum(j[1] for j in new),
                completed_tokens,
                (now - event["gpu_start_upper_ns"]) / 1e6,
                (event["gpu_start_upper_ns"] - row["dispatch_ns"]) / 1e6,
            ]
            row["forecast_time_ns"] = now
            row["prior_prediction_ms"] = event["predicted_duration_ns"] / 1e6
            output.append(row)
    with (args.output / "forecasts.jsonl").open("w") as stream:
        for row in output:
            stream.write(json.dumps(row) + "\n")
    summary = dict(
        samples=len(output),
        episodes=len({r["episode"] for r in output}),
        workloads=dict(Counter(r["workload"] for r in output)),
        rejected=dict(rejected),
        features=[
            "current_load_count",
            "current_load_tokens",
            "load_min_age_ms",
            "load_max_age_ms",
            "load_weighted_age_ms",
            "old_remaining_count",
            "old_remaining_tokens",
            "new_load_count",
            "new_load_tokens",
            "completed_old_tokens",
            "elapsed_execution_ms",
            "queue_to_gpu_start_ms",
        ],
        availability="No completion observed after forecast_time_ns is used.",
    )
    (args.output / "inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
