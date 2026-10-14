# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded retrospective opportunities; never an execution counterfactual."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def analyze(events, timing, candidate_cap=4):
    loads = {e["request"]: e for e in timing if e["event"] == "kv_load_dispatched"}
    completions = {
        e["job_id"]: e for e in timing if e["event"] == "kv_all_rank_completion"
    }
    known_ready = {
        e["request"]: e["time_ns"] for e in timing if e["event"] == "kv_ready_observed"
    }
    selected = {}
    for event in timing:
        if event["event"] == "dispatch":
            for rid in event["selected"]:
                selected[rid] = min(selected.get(rid, float("inf")), event["time_ns"])
    arrivals = {
        e["request"]: e["time_ns"]
        for e in events
        if e["event"] == "engine_request_visible"
    }
    stages = defaultdict(lambda: defaultdict(dict))
    for e in events:
        if "step" in e and e["rank"] >= 0:
            stages[e["step"]][e["event"]][e["rank"]] = e["time_ns"]
    rows = []
    for batch in (e for e in timing if e["event"] == "dispatch"):
        opened = batch["dispatch_ns"]
        step = batch["step"]
        # Only metadata dispatched before opening is eligible for reservation.
        # Future readiness is not used to choose the bounded candidate set.
        candidates = sorted(
            (
                rid
                for rid, load in loads.items()
                if load["dispatch_ns"] <= opened
                and rid not in batch["selected"]
                and known_ready.get(rid, float("inf")) > opened
                and selected.get(rid, float("inf")) > opened
            ),
            key=lambda rid: (arrivals.get(rid, loads[rid]["dispatch_ns"]), rid),
        )[:candidate_cap]
        row = dict(
            step=step,
            opened_ns=opened,
            base_requests=len(batch["selected"]),
            base_tokens=sum(batch["selected"].values()),
            candidates=candidates,
            boundaries={},
        )
        for name in (
            "worker_entry",
            "request_state_updated",
            "input_copy_submission",
            "forward_submission_begin",
        ):
            ranks = stages[step][name]
            boundary = dict(
                rank_coverage=len(ranks),
                candidates=len(candidates),
                ready_at_open=[],
                became_ready=[],
                missing_completion=[],
            )
            if set(ranks) == set(range(8)):
                # Earliest rank is the conservative no-rank-waits boundary;
                # latest rank is an upper bound requiring coordination.
                boundary["earliest_ns"], boundary["latest_ns"] = (
                    min(ranks.values()),
                    max(ranks.values()),
                )
                for rid in candidates:
                    completion = completions.get(loads[rid]["job_id"])
                    stamps = (
                        completion.get("rank_completion_ns", []) if completion else []
                    )
                    if len(stamps) != 8 or any(t <= 0 for t in stamps):
                        boundary["missing_completion"].append(rid)
                        continue
                    ready = max(stamps)
                    if ready <= opened:
                        boundary["ready_at_open"].append(rid)
                    elif ready <= boundary["earliest_ns"]:
                        boundary["became_ready"].append(rid)
            row["boundaries"][name] = boundary
        rows.append(row)
    summary = {}
    for name in (
        "worker_entry",
        "request_state_updated",
        "input_copy_submission",
        "forward_submission_begin",
    ):
        observed = [r for r in rows if r["boundaries"][name]["rank_coverage"] == 8]
        useful = [r for r in observed if r["boundaries"][name]["became_ready"]]
        summary[name] = dict(
            batches_observed=len(observed),
            batches_total=len(rows),
            batches_with_new_ready=len(useful),
            zero_base_batches_with_new_ready=sum(
                r["base_requests"] == 0 for r in useful
            ),
            candidate_observations=sum(len(r["candidates"]) for r in observed),
            new_ready_observations=sum(
                len(r["boundaries"][name]["became_ready"]) for r in observed
            ),
            unique_new_ready=len(
                {rid for r in observed for rid in r["boundaries"][name]["became_ready"]}
            ),
            missing_completion=sum(
                len(r["boundaries"][name]["missing_completion"]) for r in observed
            ),
        )
    return dict(
        candidate_cap=candidate_cap,
        summary=summary,
        batches=rows,
        limitation=(
            "Instrumented observational upper bound: no resource feasibility, "
            "reservation contention, cancellation, or coordination cost inferred. "
            "Candidate observations can repeat across batches. Future selection "
            "only excludes already-selected requests; it does not rank candidates."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = {}
    for cell in sorted(args.source.glob("*/calibration/*/b*_credit0_markers*")):
        if not (cell / "diagnostic_validation.json").exists():
            continue
        validation = json.loads((cell / "diagnostic_validation.json").read_text())
        if not validation["passed"]:
            continue
        events = json.loads((cell / "queue_stage_events.json").read_text())
        timing = json.loads((cell / "timing_events.json").read_text())
        results[str(cell.relative_to(args.source))] = analyze(events, timing)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for name, result in results.items():
        print(name, json.dumps(result["summary"]["request_state_updated"]))


if __name__ == "__main__":
    main()
