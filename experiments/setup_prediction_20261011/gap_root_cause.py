# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join empty-dispatch holds with candidate readiness and all-rank load submission."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def analyze(directory, stages, comparison):
    events = [
        json.loads(line)
        for line in (directory / "engine.jsonl").read_text().splitlines()
    ]
    records = read(directory / "late.json")["records"]
    result = read(directory / "result.json")
    computes = sorted(
        (e for e in events if e["event"] == "step_completion" and e["selected"]),
        key=lambda e: e["step"],
    )
    preceding = {b["step"]: a for a, b in zip(computes, computes[1:])}
    seals = {r["step"]: r for r in records if r["event"] == "membership_sealed"}
    by_step = defaultdict(list)
    for record in records:
        if "step" in record:
            by_step[record["step"]].append(record)
    dispatches = {e["step"]: e for e in events if e["event"] == "dispatch"}
    jobs = {e["job_id"]: e for e in events if e["event"] == "kv_load_dispatched"}
    completion = {
        e["job_id"]: e for e in events if e["event"] == "kv_all_rank_completion"
    }
    zero = min(d["dispatch_ns"] for d in dispatches.values())
    key = result["workload"], result["label"], result["repeat"]
    gaps = {
        g["step"]: g
        for g in comparison["gaps"]
        if (g["workload"], g["arm"], g["repeat"]) == key
    }
    load_only = []
    for b in computes:
        step = b["step"]
        if (
            step not in preceding
            or seals[step]["transport"].get("has_compute") is not False
        ):
            continue
        a = preceding[step]
        transport = seals[step]["transport"]
        eligibility = next(r for r in by_step[step] if r["event"] == "seal_eligibility")
        candidates = eligibility["candidates"]
        base_ns = transport["base_ready_ns"]
        ack, kv = {}, {}
        for r in by_step[step]:
            if r["event"] == "candidate_kv_transition":
                for req in r["ready"]:
                    kv.setdefault(req, r["time_ns"])
            if r["event"] == "preparation_ack_transition":
                for req, status in zip(candidates, r["candidates"]):
                    if all(s == 1 for s in status):
                        ack.setdefault(req, r["time_ns"])
        ready = sorted(
            max(base_ns, ack[req], kv[req])
            for req in candidates
            if req in ack and req in kv
        )
        assert len(ready) >= transport["ready_target"]
        a_end = max(r[3] + r[6] for r in a["execution_ranks"])
        thresholds = {str(n): ready[n - 1] for n in [1, 4, 8] if len(ready) >= n}
        load_only.append(
            dict(
                step=step,
                previous_compute_step=a["step"],
                gap_ms=gaps[step]["gap_ms"],
                phases_ms=gaps[step]["phases_ms"],
                preceding_compute_end_ms=(a_end - zero) / 1e6,
                seal_ms=(transport["seal_ns"] - zero) / 1e6,
                base_ready_ms=(base_ns - zero) / 1e6,
                ready_target=transport["ready_target"],
                published_candidates=len(candidates),
                selected_requests=len(b["selected"]),
                compute_deadline_ns=transport.get("compute_seal_deadline_ns"),
                setup_prediction=transport.get("setup_prediction"),
                ready_threshold_after_base_ms={
                    k: (v - base_ns) / 1e6 for k, v in thresholds.items()
                },
                wait_after_first_eligible_ms=(transport["seal_ns"] - ready[0]) / 1e6,
                wait_after_four_eligible_ms=(transport["seal_ns"] - thresholds["4"])
                / 1e6
                if "4" in thresholds
                else None,
                target_observation_to_seal_ms=(
                    transport["seal_ns"] - ready[transport["ready_target"] - 1]
                )
                / 1e6,
                hold_ms=transport["seal_hold_actual_ns"] / 1e6,
                candidates=[
                    dict(request=req, ack_ns=ack.get(req), kv_observed_ns=kv.get(req))
                    for req in candidates
                ],
            )
        )
    loads = []
    visible = {
        e["request"]: e["time_ns"]
        for e in stages
        if e["event"] == "engine_request_visible"
    }
    client_started = {
        "cmpl-" + r["request"]: r["start_ns"] for r in result["responses"]
    }
    for job, dispatch in jobs.items():
        submission = [
            e for e in stages if e["event"] == "load_submit" and e["job_id"] == job
        ]
        assert len(submission) == 8, (job, len(submission))
        request = dispatch["request"]
        client_ns = client_started[request.rsplit("-", 2)[0]]
        row = dict(
            job_id=job,
            request=dispatch["request"],
            step=dispatch["step"],
            external_tokens=dispatch["external_tokens"],
            dispatch_ms=(dispatch["dispatch_ns"] - zero) / 1e6,
            client_start_ms=(client_ns - zero) / 1e6,
            engine_visible_ms=(visible[request] - zero) / 1e6,
            client_to_visible_ms=(visible[request] - client_ns) / 1e6,
            visible_to_dispatch_ms=(dispatch["dispatch_ns"] - visible[request]) / 1e6,
            all_rank_submit_ms=(max(s["time_ns"] for s in submission) - zero) / 1e6,
            first_rank_submit_ms=(min(s["time_ns"] for s in submission) - zero) / 1e6,
            all_rank_complete_ms=(completion[job]["completion_ns"] - zero) / 1e6,
            bytes_per_rank=sorted({s["num_bytes"] for s in submission}),
            dispatched_in_load_only_rpc=not bool(
                dispatches[dispatch["step"]]["selected"]
            ),
        )
        loads.append(row)
    second = computes[1]
    previous_step = max(s for s in dispatches if s < second["step"])
    previous_jobs = {job for job, d in jobs.items() if d["step"] == previous_step}
    submissions = [
        e
        for e in stages
        if e["event"] == "load_submit" and e["job_id"] in previous_jobs
    ]
    rpc = next(
        e["rpc"]
        for e in stages
        if e["event"] == "rpc_enqueue"
        and e.get("step") == previous_step
        and e["method"] == "execute_model"
    )
    outputs = [
        e["time_ns"]
        for e in stages
        if e["event"] == "worker_output_ready" and e["rpc"] == rpc
    ]
    assert len(outputs) == 8
    second_records = by_step[second["step"]]
    base = next((e for e in second_records if e["event"] == "all_base_ready"), None)
    first_ack = next(
        (e for e in second_records if e["event"] == "preparation_ack_transition"),
        None,
    )
    initial_gate = dict(
        second_compute_step=second["step"],
        previous_empty_step=previous_step,
        previous_load_count=len(previous_jobs),
        previous_last_rank_submit_ms=(max(s["time_ns"] for s in submissions) - zero)
        / 1e6,
        previous_first_output_ms=(min(outputs) - zero) / 1e6,
        previous_last_output_ms=(max(outputs) - zero) / 1e6,
        B_first_ack_snapshot_ms=(first_ack["time_ns"] - zero) / 1e6
        if first_ack
        else None,
        B_all_base_ready_ms=(base["observed_ns"] - zero) / 1e6 if base else None,
        B_transport=seals[second["step"]]["transport"],
        setup_forecasts_before_second_dispatch=sum(
            e["event"] == "setup_prediction" and e["time_ns"] < second["dispatch_ns"]
            for e in records
        ),
    )
    return dict(
        arm=result["label"],
        repeat=result["repeat"],
        throughput_rps=result["throughput_rps"],
        load_only_gaps=load_only,
        loads=loads,
        initial_gate=initial_gate,
        fallback_steps=[
            dict(
                step=s,
                selected_requests=len(dispatches[s]["selected"]),
                transport=r["transport"],
            )
            for s, r in seals.items()
            if r["transport"].get("predecessor_ready") is True
            and "base_ready_ns" not in r["transport"]
        ],
        initial_computes=[
            dict(
                step=b["step"],
                requests=len(b["selected"]),
                shapes=b["request_shapes"],
                seal_reason=seals[b["step"]]["transport"].get("seal_reason"),
            )
            for b in computes[:8]
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stages = [
        e
        for p in (args.root / "server/timing/queue_stages").glob("*.json")
        for e in read(p)["events"]
    ]
    comparison = read(args.root / "comparison.json")
    episodes = [
        analyze(directory, stages, comparison)
        for directory in sorted((args.root / "live/mixed_supported").glob("r[01]_*"))
    ]
    args.output.write_text(
        json.dumps(
            dict(
                root=str(args.root),
                episodes=episodes,
                caveat="Ready-threshold differences are fixed-trace timing "
                "opportunities, not counterfactual throughput. Worker load_submit "
                "is host submission, not first DMA.",
            ),
            indent=2,
        )
        + "\n"
    )
    for e in episodes:
        print(
            e["arm"],
            e["repeat"],
            "load-only gaps",
            [
                {
                    k: g[k]
                    for k in [
                        "step",
                        "gap_ms",
                        "ready_target",
                        "wait_after_first_eligible_ms",
                        "wait_after_four_eligible_ms",
                        "target_observation_to_seal_ms",
                    ]
                }
                for g in e["load_only_gaps"]
            ],
        )


if __name__ == "__main__":
    main()
