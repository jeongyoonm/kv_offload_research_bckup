# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Estimate deadline admission losses from unchanged serving traces."""

import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def analyze_cell(path, leads):
    events = [
        json.loads(line) for line in (path / "engine.jsonl").read_text().splitlines()
    ]
    records = read(path / "late.json")["records"]
    dispatch = {e["step"]: e for e in events if e["event"] == "dispatch"}
    done = {e["step"]: e for e in events if e["event"] == "step_completion"}
    eligibility = {r["step"]: r for r in records if r["event"] == "seal_eligibility"}
    physical_ready = {
        e["early_promoted_request"]: e["completion_ns"]
        for e in events
        if e["event"] == "kv_all_rank_completion" and e.get("early_promoted_request")
    }
    rows = []
    for seal in records:
        if seal["event"] != "membership_sealed":
            continue
        step = seal["step"]
        transport = seal.get("transport", {})
        prediction = transport.get("prediction")
        if not prediction:
            continue
        first = next(
            r for r in records if r["event"] == "seal_prediction" and r["step"] == step
        )
        a = prediction["predecessor_step"]
        a_ranks = done[a]["execution_ranks"]
        end_lower = max(r[7] for r in a_ranks)
        end_upper = max(r[8] for r in a_ranks)
        forecast = (
            prediction["gpu_start_upper_ns"] + prediction["predicted_duration_ns"]
        )
        actual_seal = transport["seal_ns"]
        activated = set(seal.get("activated", {}))
        candidate_ids = eligibility[step]["candidates"]
        for lead in leads:
            deadline = min(
                forecast - lead * 1_000_000,
                transport["base_ready_ns"] + 2_000_000_000,
            )
            cut = min(actual_seal, max(deadline, first["time_ns"]))
            ready = set()
            acked = set()
            for r in records:
                if r.get("step") != step or r["time_ns"] > cut:
                    continue
                if r["event"] == "candidate_kv_transition":
                    ready = set(r["ready"])
                elif r["event"] == "preparation_ack_transition":
                    assert len(candidate_ids) == len(r["candidates"])
                    acked = {
                        req
                        for req, statuses in zip(candidate_ids, r["candidates"])
                        if all(s == 1 for s in statuses)
                    }
            # An unchanged seal uses its authoritative snapshot, including the
            # progress poll that immediately preceded finalization.
            if cut == actual_seal:
                ready = set(transport.get("kv_ready", []))
                statuses = transport["statuses"]
                acked = {
                    req
                    for slot, req in enumerate(candidate_ids)
                    if all(s == 1 for s in statuses[8 * (slot + 1) : 8 * (slot + 2)])
                }
            b = dispatch[step]
            intermediates = [
                d for s, d in dispatch.items() if a < s < step and not d.get("selected")
            ]
            eligible_saving = not intermediates and b.get("selected")
            new_enqueue = b["dispatch_ns"] - (actual_seal - cut)
            lower = (
                (max(0, b["dispatch_ns"] - end_upper) - max(0, new_enqueue - end_upper))
                / 1e6
                if eligible_saving
                else 0
            )
            upper = (
                (max(0, b["dispatch_ns"] - end_lower) - max(0, new_enqueue - end_lower))
                / 1e6
                if eligible_saving
                else 0
            )
            rows.append(
                dict(
                    cell=path.name,
                    step=step,
                    predecessor_step=a,
                    lead_ms=lead,
                    deadline_ns=deadline,
                    hypothetical_seal_ns=cut,
                    actual_seal_ns=actual_seal,
                    forecast_error_lower_ms=(forecast - end_upper) / 1e6,
                    forecast_error_upper_ms=(forecast - end_lower) / 1e6,
                    earlier=cut < actual_seal,
                    actual_admissions=len(activated),
                    lost_physically_unready=sorted(
                        req
                        for req in activated
                        if physical_ready.get(req, 10**30) > cut
                    ),
                    lost_in_observed_snapshot=sorted(activated - (ready & acked)),
                    ideal_pre_enqueue_saving_lower_ms=lower,
                    ideal_pre_enqueue_saving_upper_ms=upper,
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    leads = [0, 5, 10, 20]
    paths = [
        Path(c["path"])
        for c in read(args.source / "cells.json")
        if c["mode"] == "compact_pinned"
    ]
    rows = [row for path in paths for row in analyze_cell(path, leads)]
    summary = []
    for lead in leads:
        selected = [r for r in rows if r["lead_ms"] == lead]
        admissions = sum(r["actual_admissions"] for r in selected)
        losses = sum(len(r["lost_physically_unready"]) for r in selected)
        summary.append(
            dict(
                lead_ms=lead,
                cells=len(paths),
                predicted_windows=len(selected),
                earlier_seals=sum(r["earlier"] for r in selected),
                actual_admissions=admissions,
                lost_physically_unready=losses,
                lost_physical_pct=100 * losses / admissions,
                lost_in_observed_snapshot=sum(
                    len(r["lost_in_observed_snapshot"]) for r in selected
                ),
                ideal_pre_enqueue_saving_lower_ms_per_cell=sum(
                    r["ideal_pre_enqueue_saving_lower_ms"] for r in selected
                )
                / len(paths),
                ideal_pre_enqueue_saving_upper_ms_per_cell=sum(
                    r["ideal_pre_enqueue_saving_upper_ms"] for r in selected
                )
                / len(paths),
                forecast_error_median_ms=statistics.median(
                    (r["forecast_error_lower_ms"] + r["forecast_error_upper_ms"]) / 2
                    for r in selected
                ),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                source=str(args.source),
                summary=summary,
                windows=rows,
                limitations=(
                    "Static replay does not model feedback from changed batches. "
                    "Observed snapshots may lag physical readiness. Admission "
                    "losses are existing actual admissions unavailable at the "
                    "hypothetical seal, not requests permanently lost. Savings "
                    "assume unchanged batch shapes and no worker delay; they "
                    "are not predicted throughput gains."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
