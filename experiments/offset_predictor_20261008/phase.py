# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Describe per-cell phase differences without changing live execution."""

import argparse
import json
from pathlib import Path

import numpy as np
from analyze import diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.root / "live").glob("*/r[0-3]_*/result.json")):
        result = json.loads(path.read_text())
        d = diagnostics(path.parent)
        late = json.loads((path.parent / "late.json").read_text())["records"]
        no_compute_steps = {
            e["step"]
            for e in late
            if e["event"] == "seal_repeat" and not e["has_compute"]
        }
        no_compute_holds = []
        for event in late:
            if event["event"] == "membership_sealed":
                transport = event.get("transport", {})
                if event["step"] in no_compute_steps:
                    no_compute_holds.append(
                        (transport["seal_ns"] - transport["start_ns"]) / 1e6
                    )
        events = [
            json.loads(line)
            for line in (path.parent / "engine.jsonl").read_text().splitlines()
        ]
        dispatched = {
            e["job_id"]: e["dispatch_ns"]
            for e in events
            if e["event"] == "kv_load_dispatched"
        }
        load_times = [
            (e["completion_ns"] - dispatched[e["job_id"]]) / 1e6
            for e in events
            if e["event"] == "kv_all_rank_completion" and e["job_id"] in dispatched
        ]
        rows.append(
            dict(
                workload=result["workload"],
                repeat=result["repeat"],
                predictor=result["label"],
                throughput_rps=result["throughput_rps"],
                first_request_ns=min(r["start_ns"] for r in result["responses"]),
                compute_rounds=d["compute_rounds"],
                batch_requests=d["batch_requests"],
                compute_span_ms=d["gpu_compute_span_ms"],
                gpu_gap_ms=d["gpu_gap_ms"],
                native_predictions=d["point_predictions"],
                no_compute_repeated_hold_ms=no_compute_holds,
                no_compute_repeated_hold_sum_ms=sum(no_compute_holds),
                load_turnaround_median_ms=float(np.median(load_times))
                if load_times
                else None,
                load_turnaround_p95_ms=float(np.quantile(load_times, 0.95))
                if load_times
                else None,
                aggregate_load_bytes=result["metrics_after"][
                    "vllm:kv_offload_load_bytes_total"
                ]
                - result["metrics_before"]["vllm:kv_offload_load_bytes_total"],
            )
        )
    rows.sort(key=lambda r: r["first_request_ns"])
    previous = None
    position = 0
    for row in rows:
        group = (row["workload"], row["repeat"])
        position = position + 1 if group == previous else 0
        row["position_within_workload_repeat"] = position
        previous = group
    output = dict(
        measured_cells=len(rows),
        cells=rows,
        scope=(
            "Repeated holds classified by a recorded has_compute=False poll. "
            "Hold sums are not disjoint physical GPU-gap attribution. KV load "
            "turnaround includes queueing and transfer. This audit is descriptive."
        ),
    )
    (args.root / "phase_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(measured_cells=len(rows))))


if __name__ == "__main__":
    main()
