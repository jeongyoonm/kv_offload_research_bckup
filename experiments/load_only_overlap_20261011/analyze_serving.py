# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize serving cells while retaining workload/repeat identity."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def uncovered(begin, end, intervals):
    cursor, idle = begin, 0
    for start, finish in intervals:
        if finish <= cursor or start >= end:
            continue
        idle += max(0, min(start, end) - cursor)
        cursor = max(cursor, min(finish, end))
    return idle + max(0, end - cursor)


def summarize(path):
    result = json.loads(path.read_text())
    engine = [json.loads(line) for line in (path.parent / "engine.jsonl").open()]
    records = json.loads((path.parent / "late.json").read_text())["records"]
    computed = [
        e
        for e in engine
        if e["event"] == "step_completion" and e.get("selected") and e.get("valid")
    ]
    intervals = sorted(
        (
            max(r[3] for r in e["execution_ranks"]),
            max(r[3] + r[6] for r in e["execution_ranks"]),
        )
        for e in computed
    )
    seals = [e["transport"] for e in records if e["event"] == "seal_eligibility"]
    holds = [t for t in seals if t.get("base_ready_ns") and t.get("seal_ns")]
    load_only = [t for t in holds if not t.get("has_compute", False)]
    by_step = {e["step"]: e for e in computed}
    prediction_errors, seal_slacks, seal_to_gpu, late_seals = [], [], [], 0
    for event in records:
        if event["event"] != "seal_eligibility":
            continue
        transport = event["transport"]
        prediction = transport.get("prediction") or {}
        predecessor = by_step.get(prediction.get("predecessor_step"))
        upcoming = by_step.get(event.get("step"))
        if predecessor and prediction.get("predicted_duration_ns"):
            end = max(r[3] + r[6] for r in predecessor["execution_ranks"])
            estimated = (
                prediction["gpu_start_upper_ns"] + prediction["predicted_duration_ns"]
            )
            prediction_errors.append((estimated - end) / 1e6)
            seal = transport["seal_ns"]
            seal_slacks.append((end - seal) / 1e6)
            late_seals += seal > end
        if upcoming and transport.get("seal_ns"):
            start = max(r[3] for r in upcoming["execution_ranks"])
            seal_to_gpu.append((start - transport["seal_ns"]) / 1e6)
    return dict(
        prediction_count=len(prediction_errors),
        prediction_end_mae_ms=statistics.mean(map(abs, prediction_errors))
        if prediction_errors
        else None,
        prediction_end_bias_ms=statistics.mean(prediction_errors)
        if prediction_errors
        else None,
        prediction_seal_slack_ms=seal_slacks,
        prediction_late_seals=late_seals,
        seal_to_B_gpu_start_ms=seal_to_gpu,
        path=str(path.parent),
        arm=result["label"],
        workload=result["workload"],
        repeat=result["repeat"],
        accepted=result["accepted"],
        rps=result["throughput_rps"],
        mean_ttft_ms=result["mean_ttft_ms"],
        p95_ttft_ms=result["p95_ttft_ms"],
        rounds=result["compute_rounds"],
        admissions=result["late_admissions"],
        kv_ready_during_compute=result["kv_ready_during_compute"],
        recovered_compute_opportunities=result["recovered_compute_opportunities"],
        gpu_envelope_ms=sum(b - a for a, b in intervals) / 1e6,
        inter_compute_gaps_ms=sum(
            max(0, b[0] - a[1]) for a, b in zip(intervals, intervals[1:])
        )
        / 1e6,
        load_only_seals=len(load_only),
        load_only_hold_ms=sum(t.get("seal_hold_actual_ns", 0) for t in load_only) / 1e6,
        all_exposed_hold_ms=sum(
            uncovered(t["base_ready_ns"], t["seal_ns"], intervals) for t in holds
        )
        / 1e6,
        load_only_exposed_hold_ms=sum(
            uncovered(t["base_ready_ns"], t["seal_ns"], intervals) for t in load_only
        )
        / 1e6,
        seal_reasons=dict(
            Counter(t.get("seal_reason", t.get("path", "unknown")) for t in seals)
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [
        summarize(path)
        for path in sorted((args.root / "live").glob("*/*/result.json"))
        if "warmup" not in path.parent.name
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[row["workload"], row["arm"]].append(row)
    summaries = []
    for (workload, arm), cells in groups.items():
        summaries.append(
            dict(
                workload=workload,
                arm=arm,
                repeats=len(cells),
                all_accepted=all(c["accepted"] for c in cells),
                means={
                    key: statistics.mean(c[key] for c in cells)
                    for key in cells[0]
                    if isinstance(cells[0][key], (int, float))
                    and key not in ("repeat", "accepted")
                },
            )
        )
    args.output.write_text(json.dumps(dict(rows=rows, groups=summaries), indent=2))
    for summary in summaries:
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
