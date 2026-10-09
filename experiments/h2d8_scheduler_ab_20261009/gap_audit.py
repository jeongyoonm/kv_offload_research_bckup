# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Partition existing 60K gaps using recorded scheduler and worker boundaries."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

STAGES = (
    "worker_entry",
    "request_state_updated",
    "input_copy_submission",
    "input_preparation_return",
    "attention_metadata_ready",
    "forward_submission_begin",
)
PHASES = (
    "before_schedule",
    "scheduler",
    "dispatch_to_worker",
    "request_state",
    "prepare_to_copy",
    "input_copy_to_return",
    "attention_metadata",
    "pre_forward",
    "forward_to_gpu_start",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    stages = defaultdict(dict)
    loads = {}
    for path in (root / "server/timing/queue_stages").glob("*.json"):
        export = json.loads(path.read_text())
        rank = export.get("rank", -1)
        if rank < 0:
            continue
        assert not export["overflow"], path
        for event in export["events"]:
            if event["event"] in STAGES:
                stages[event["step"], rank][event["event"]] = event["time_ns"]
            elif event["event"] == "load_submit":
                loads[event["job_id"], rank] = event["time_ns"]
    rows, episodes = [], []
    for arm in ("C", "D"):
        for repeat in range(4):
            directory = root / "live/familiar60k" / f"r{repeat}_{arm}"
            events = list(
                map(json.loads, (directory / "engine.jsonl").read_text().splitlines())
            )
            seals = {
                e["step"]: e["transport"]
                for e in json.loads((directory / "late.json").read_text())["records"]
                if e["event"] == "membership_sealed"
            }
            computes = sorted(
                [
                    e
                    for e in events
                    if e["event"] == "step_completion" and e.get("selected")
                ],
                key=lambda e: e["step"],
            )
            totals = defaultdict(float)
            for previous, current in zip(computes, computes[1:]):
                step = current["step"]
                end = max(r[3] + r[6] for r in previous["execution_ranks"])
                start = max(r[3] for r in current["execution_ranks"])
                assert start >= end
                rank_stages = {str(rank): stages[step, rank] for rank in range(8)}
                points = [
                    max(stages[step, rank][s] for rank in range(8)) for s in STAGES
                ]
                boundaries = [
                    end,
                    current["schedule_begin_ns"],
                    current["dispatch_ns"],
                    *points,
                    start,
                ]
                clipped = [max(end, min(start, t)) for t in boundaries]
                assert sorted(clipped) == clipped, (arm, repeat, step)
                phases = {
                    name: (b - a) / 1e6
                    for name, a, b in zip(PHASES, clipped, clipped[1:])
                }
                gap_ms = (start - end) / 1e6
                assert abs(sum(phases.values()) - gap_ms) < 1e-6
                for name, value in phases.items():
                    totals[name] += value
                meta, forward = points[-2:]
                totals["full_pre_forward_ms"] += (forward - meta) / 1e6
                totals["pre_forward_hidden_by_A_ms"] += (
                    max(0, min(forward, end) - meta) / 1e6
                )
                totals["compute_transitions"] += 1
                totals["dispatched_before_A_end"] += current["dispatch_ns"] < end
                totals["forward_submitted_before_A_end"] += forward < end
                job_ids = [
                    e["job_id"]
                    for e in events
                    if e["event"] == "kv_load_dispatched" and e["step"] == step
                ]
                transport = seals.get(step, {})
                deadline = transport.get("compute_seal_deadline_ns")
                if deadline:
                    assert (
                        transport["prediction"]["predecessor_step"] == previous["step"]
                    )
                rows.append(
                    dict(
                        arm=arm,
                        repeat=repeat,
                        step=step,
                        preceding_step=previous["step"],
                        requests=len(current["selected"]),
                        gap_ms=gap_ms,
                        phases_ms=phases,
                        A_end_ns=end,
                        B_start_upper_ns=start,
                        schedule_begin_ns=current["schedule_begin_ns"],
                        dispatch_ns=current["dispatch_ns"],
                        worker_stages=rank_stages,
                        new_loads={
                            str(job): {
                                str(rank): loads.get((job, rank)) for rank in range(8)
                            }
                            for job in job_ids
                        },
                        seal_reason=transport.get("seal_reason"),
                        deadline_ns=deadline,
                        deadline_minus_A_end_ms=(deadline - end) / 1e6
                        if deadline
                        else None,
                    )
                )
            episodes.append(dict(arm=arm, repeat=repeat, **totals))
    means = {
        arm: {
            key: statistics.mean(e[key] for e in episodes if e["arm"] == arm)
            for key in episodes[0]
            if key not in ("arm", "repeat")
        }
        for arm in ("C", "D")
    }
    result = dict(
        means=means,
        episodes=episodes,
        gaps=rows,
        definition="Conservative all-rank execution gap bounds, clipped at each "
        "host-stage boundary. Sums partition the previously reported gaps. "
        "Pre-forward includes connector handling and other work after attention "
        "metadata. Store submission and store-completion waits are not separately "
        "instrumented; their individual contributions cannot be established here.",
    )
    errors = [
        r["deadline_minus_A_end_ms"]
        for r in rows
        if r["deadline_minus_A_end_ms"] is not None
    ]
    result["deadline_audit"] = dict(
        observations=len(errors),
        deadlines_after_A_end=sum(e > 0 for e in errors),
        mean_deadline_minus_A_end_ms=statistics.mean(errors),
        median_deadline_minus_A_end_ms=statistics.median(errors),
    )
    (root / "gap_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(means, indent=2))
    print(json.dumps(result["deadline_audit"], indent=2))


if __name__ == "__main__":
    main()
