# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit frozen prediction correction and candidates missed at compute deadlines."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def audit(root):
    comparison = json.loads((root / "comparison.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"]
    output = {}
    for workload in ("short16k", "familiar60k"):
        errors = {}
        transitions = [
            g
            for g in comparison["gaps"]
            if g["workload"] == workload and g["arm"] == "D"
        ]
        for key in (
            "raw_prediction_error_ms",
            "corrected_prediction_error_ms",
            "deadline_minus_A_end_ms",
        ):
            values = [g[key] for g in transitions if g[key] is not None]
            errors[key] = dict(
                count=len(values),
                mean=statistics.mean(values),
                mae=statistics.mean(abs(v) for v in values),
                minimum=min(values),
                maximum=max(values),
            )
        totals = Counter()
        rows = []
        episodes = [
            r
            for r in comparison["episodes"]
            if r["workload"] == workload and r["label"] == "D"
        ]
        for episode in episodes:
            repeat = episode["repeat"]
            directory = root / "live" / workload / f"r{repeat}_D"
            events = list(
                map(json.loads, (directory / "engine.jsonl").read_text().splitlines())
            )
            late = json.loads((directory / "late.json").read_text())["records"]
            completions = {
                e["job_id"]: max(e["rank_completion_ns"])
                for e in events
                if e["event"] == "kv_all_rank_completion"
            }
            ready = {
                e["request"]: completions[e["job_id"]]
                for e in events
                if e["event"] == "kv_load_dispatched"
            }
            computes = {
                e["step"]: e
                for e in events
                if e["event"] == "step_completion" and e.get("selected")
            }
            intervals = [
                (
                    max(r[3] for r in e["execution_ranks"]),
                    max(r[3] + r[6] for r in e["execution_ranks"]),
                )
                for e in computes.values()
            ]
            opportunities = {
                rid
                for rid, t in ready.items()
                if any(a <= t <= b for a, b in intervals)
            }
            admitted = {
                rid
                for e in late
                if e["event"] == "membership_sealed"
                for rid in e["activated"]
            }
            missed = opportunities - admitted
            assert len(opportunities) == episode["kv_ready_during_compute"]
            assert (
                len(opportunities & admitted)
                == episode["recovered_compute_opportunities"]
            )
            totals["opportunities"] += len(opportunities)
            totals["recovered"] += len(opportunities & admitted)
            totals["not_late_admitted"] += len(missed)
            eligibility = {
                e["step"]: e for e in late if e["event"] == "seal_eligibility"
            }
            window, acked = set(), set()
            for event in late:
                if event["event"] != "membership_sealed":
                    continue
                transport = event["transport"]
                if transport.get("seal_reason") != "compute_seal_deadline":
                    continue
                prediction = transport.get("prediction")
                predecessor = (
                    computes.get(prediction["predecessor_step"]) if prediction else None
                )
                if predecessor is None:
                    continue
                totals["deadline_seals"] += 1
                end = max(r[3] + r[6] for r in predecessor["execution_ranks"])
                seal = transport["seal_ns"]
                eligible = eligibility[event["step"]]
                for slot, rid in enumerate(eligible["candidates"]):
                    if rid in transport["cancelled"] or rid in event["activated"]:
                        continue
                    if rid not in ready or not seal < ready[rid] <= end:
                        continue
                    prepared = rid in eligible["staged"]
                    statuses = transport["statuses"][8 * (slot + 1) : 8 * (slot + 2)]
                    assert prepared == (
                        len(statuses) == 8 and all(s == 1 for s in statuses)
                    )
                    window.add(rid)
                    if prepared:
                        acked.add(rid)
                    rows.append(
                        dict(
                            repeat=repeat,
                            step=event["step"],
                            predecessor_step=predecessor["step"],
                            request=rid,
                            all_rank_prepared_at_seal=prepared,
                            seal_ns=seal,
                            kv_all_rank_ready_ns=ready[rid],
                            A_end_upper_ns=end,
                            ready_after_seal_ms=(ready[rid] - seal) / 1e6,
                            never_late_admitted=rid in missed,
                        )
                    )
            totals["unique_deadline_window_candidates"] += len(window)
            totals["unique_prepared_deadline_window_candidates"] += len(acked)
            totals["deadline_window_and_never_late_admitted"] += len(window & missed)
            totals["prepared_deadline_window_and_never_late_admitted"] += len(
                acked & missed
            )
        output[workload] = dict(
            prediction_errors=errors, counts=dict(totals), rows=rows
        )
    return dict(
        workloads=output,
        definition="Deadline window: published, not cancelled/admitted at a "
        "compute-deadline seal; actual all-rank KV completion after seal and "
        "before preceding compute's conservative all-rank end. Prepared means "
        "all eight successful acknowledgments in the seal snapshot. Unique "
        "requests are counted within episodes and summed. Prediction errors "
        "cover logged adjacent compute transitions, not every request. "
        "These are retrospective timing opportunities, not guaranteed "
        "counterfactual admissions or throughput gains.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve existing audit outputs"
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                w: {k: v for k, v in data.items() if k != "rows"}
                for w, data in result["workloads"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
