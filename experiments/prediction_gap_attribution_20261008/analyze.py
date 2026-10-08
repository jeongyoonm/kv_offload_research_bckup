# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join saved forecasts, seals, dispatches and consecutive GPU spans."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

NS_MS = 1_000_000


def mean(values):
    return statistics.mean(values) if values else None


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def span(event):
    ranks = event["execution_ranks"]
    assert len(ranks) == 8
    assert all(r[5] == 1 and r[6] > 0 for r in ranks)
    return dict(
        start=max(r[3] for r in ranks),
        end=max(r[3] + r[6] for r in ranks),
        start_lower=max(r[2] for r in ranks),
        end_lower=max(r[7] for r in ranks),
        end_upper=max(r[8] for r in ranks),
    )


def analyze_cell(path, study, forecasts):
    result = json.loads((path / "result.json").read_text())
    assert result["accepted"]
    engine = read_lines(path / "engine.jsonl")
    late = json.loads((path / "late.json").read_text())["records"]
    completed = {
        e["step"]: e
        for e in engine
        if e["event"] == "step_completion" and e.get("valid") and e.get("selected")
    }
    seals = {e["step"]: e for e in late if e["event"] == "membership_sealed"}
    native = {
        e["step"]: e
        for e in late
        if e["event"] == "seal_prediction" and e.get("predictor_input") is not None
    }
    eligibility = {e["step"]: e for e in late if e["event"] == "seal_eligibility"}
    requests_by_job = {
        e["job_id"]: e["request"] for e in engine if e["event"] == "kv_load_dispatched"
    }
    kv_completion = {
        requests_by_job[e["job_id"]]: e["completion_ns"]
        for e in engine
        if e["event"] == "kv_all_rank_completion" and e["job_id"] in requests_by_job
    }
    first_prediction = min((e["time_ns"] for e in native.values()), default=None)
    ordered = sorted(completed.values(), key=lambda e: span(e)["start"])
    rows = []
    for a, b in zip(ordered, ordered[1:]):
        a_span, b_span = span(a), span(b)
        sealed = seals[b["step"]]
        transport = sealed["transport"]
        snapshot_ns = transport["seal_ns"]
        published_ns = sealed["time_ns"]
        dispatch_ns = b["dispatch_ns"]
        assert snapshot_ns <= published_ns <= dispatch_ns <= b_span["start"]
        gap_ns = max(0, b_span["start"] - a_span["end"])

        def clip(timestamp, lower=a_span["end"], upper=b_span["start"]):
            return min(max(lower, upper), max(lower, timestamp))

        cutpoints = [
            a_span["end"],
            clip(snapshot_ns),
            clip(published_ns),
            clip(dispatch_ns),
            max(a_span["end"], b_span["start"]),
        ]
        parts = [
            (right - left) / NS_MS for left, right in zip(cutpoints, cutpoints[1:])
        ]
        assert all(value >= 0 for value in parts)
        assert abs(sum(parts) - gap_ns / NS_MS) < 1e-7
        prediction = native.get(b["step"])
        direct = prediction is not None and prediction["predecessor_step"] == a["step"]
        reason = transport.get("seal_reason", transport.get("path", "unknown"))
        category = (
            "prediction_deadline"
            if direct and reason == "compute_seal_deadline"
            else "prediction_present_other_exit"
            if direct
            else "load_only"
            if transport.get("has_compute") is False
            else "no_direct_prediction"
        )
        row = dict(
            study=study,
            workload=result["workload"],
            predictor=result["label"],
            repeat=result["repeat"],
            cell=str(path),
            predecessor_step=a["step"],
            successor_step=b["step"],
            predecessor_requests=len(a["selected"]),
            successor_requests=len(b["selected"]),
            admitted_requests=len(sealed["activated"]),
            predecessor_start_ns=a_span["start"],
            predecessor_end_ns=a_span["end"],
            snapshot_ns=snapshot_ns,
            membership_record_ns=published_ns,
            dispatch_ns=dispatch_ns,
            successor_start_ns=b_span["start"],
            category=category,
            seal_reason=reason,
            direct_native_prediction=direct,
            gap_ms=gap_ns / NS_MS,
            estimated_span_overlap_ms=max(0, a_span["end"] - b_span["start"]) / NS_MS,
            gap_lower_ms=max(0, b_span["start_lower"] - a_span["end_upper"]) / NS_MS,
            gap_upper_ms=max(0, b_span["start"] - a_span["end_lower"]) / NS_MS,
            before_snapshot_ms=parts[0],
            snapshot_to_membership_record_ms=parts[1],
            membership_record_to_dispatch_ms=parts[2],
            dispatch_to_gpu_start_ms=parts[3],
            seal_lead_ms=(a_span["end"] - snapshot_ns) / NS_MS,
            membership_before_first_prediction=(
                first_prediction is None or published_ns < first_prediction
            ),
            no_prediction_in_cell=first_prediction is None,
        )
        if category == "load_only":
            completion_times = sorted(
                kv_completion[req] for req in b["selected"] if req in kv_completion
            )
            row["successor_kv_completion_count"] = len(completion_times)
            if len(completion_times) == len(b["selected"]):
                assert completion_times[-1] <= snapshot_ns
                row.update(
                    successor_ready_at_predecessor_end=sum(
                        timestamp <= a_span["end"] for timestamp in completion_times
                    ),
                    first_successor_kv_after_predecessor_ms=(
                        completion_times[0] - a_span["end"]
                    )
                    / NS_MS,
                    eighth_successor_kv_after_predecessor_ms=(
                        completion_times[min(7, len(completion_times) - 1)]
                        - a_span["end"]
                    )
                    / NS_MS,
                    last_successor_kv_after_predecessor_ms=(
                        completion_times[-1] - a_span["end"]
                    )
                    / NS_MS,
                    all_successor_kv_to_snapshot_ms=(snapshot_ns - completion_times[-1])
                    / NS_MS,
                )
        if direct:
            predicted = prediction["predicted_duration_ns"]
            actual = a_span["end"] - a_span["start"]
            deadline = transport["compute_seal_deadline_ns"]
            lead = transport["compute_seal_deadline_lead_us"] * 1000
            assert deadline == a_span["start"] + predicted - lead
            row.update(
                prediction_error_ms=(predicted - actual) / NS_MS,
                prediction_ns=prediction["time_ns"],
                deadline_ns=deadline,
                deadline_after_predecessor_end_ms=(deadline - a_span["end"]) / NS_MS,
                deadline_overshoot_ms=(snapshot_ns - deadline) / NS_MS,
            )
            if reason == "compute_seal_deadline":
                assert snapshot_ns >= deadline
                identity = (
                    lead / NS_MS
                    - row["prediction_error_ms"]
                    - row["deadline_overshoot_ms"]
                )
                assert abs(identity - row["seal_lead_ms"]) < 1e-7
            forecast = forecasts[(str(path), a["step"])]
            replay = {}
            for model, duration_ms in forecast["predictions_ms"].items():
                alternative_deadline = (
                    a_span["start"] + round(duration_ms * NS_MS) - lead
                )
                # Earliest possible decision using this already-captured forecast.
                earliest_seal = max(
                    alternative_deadline,
                    transport["compute_seal_deadline_armed_ns"],
                    transport["base_ready_ns"],
                )
                ready_events = [
                    e
                    for e in late
                    if e.get("step") == b["step"]
                    and e["event"] == "candidate_kv_transition"
                    and e["time_ns"] <= earliest_seal
                ]
                ack_events = [
                    e
                    for e in late
                    if e.get("step") == b["step"]
                    and e["event"] == "preparation_ack_transition"
                    and e["time_ns"] <= earliest_seal
                ]
                ready = set(ready_events[-1]["ready"]) if ready_events else set()
                candidate_ids = eligibility[b["step"]]["candidates"]
                statuses = ack_events[-1]["candidates"] if ack_events else []
                acked = {
                    req
                    for req, status in zip(candidate_ids, statuses)
                    if len(status) == 8 and all(s == 1 for s in status)
                }
                replay[model] = dict(
                    error_ms=duration_ms - actual / NS_MS,
                    deadline_after_predecessor_end_ms=(
                        alternative_deadline - a_span["end"]
                    )
                    / NS_MS,
                    earliest_timer_before_actual_seal_ms=(snapshot_ns - earliest_seal)
                    / NS_MS,
                    observed_ready_acked_at_earliest_timer=(
                        len(ready & acked) if earliest_seal <= snapshot_ns else None
                    ),
                    readiness_trace_covers_alternative_timer=earliest_seal
                    <= snapshot_ns,
                )
            row["fixed_trace_deadline_replay"] = replay
        rows.append(row)
    return rows, dict(
        study=study,
        workload=result["workload"],
        predictor=result["label"],
        repeat=result["repeat"],
        seconds=result["seconds"],
        throughput_rps=result["throughput_rps"],
        requests=result["requests"],
        compute_rounds=len(ordered),
        native_forecasts=len(native),
    )


def summarize(rows, cells):
    summaries = []
    for key in sorted({(r["study"], r["workload"], r["predictor"]) for r in cells}):
        selected = [
            r for r in rows if (r["study"], r["workload"], r["predictor"]) == key
        ]
        episodes = [
            r for r in cells if (r["study"], r["workload"], r["predictor"]) == key
        ]
        count = len(episodes)
        grouped = defaultdict(list)
        for row in selected:
            grouped[row["category"]].append(row)
        direct = [r for r in selected if r["direct_native_prediction"]]
        summary = dict(
            study=key[0],
            workload=key[1],
            predictor=key[2],
            episodes=count,
            throughput_rps=sum(c["requests"] for c in episodes)
            / sum(c["seconds"] for c in episodes),
            transitions=len(selected),
            native_forecasts=sum(c["native_forecasts"] for c in episodes),
            reasons=dict(Counter(r["seal_reason"] for r in selected)),
            before_first_prediction_gap_ms_per_episode=sum(
                r["gap_ms"] for r in selected if r["membership_before_first_prediction"]
            )
            / count,
            direct_prediction_error_mean_ms=mean(
                [r["prediction_error_ms"] for r in direct]
            ),
            direct_prediction_mae_ms=mean(
                [abs(r["prediction_error_ms"]) for r in direct]
            ),
            direct_prediction_positive_error_count=sum(
                r["prediction_error_ms"] > 0 for r in direct
            ),
            categories={
                name: dict(
                    transitions=len(group),
                    gap_ms_per_episode=sum(r["gap_ms"] for r in group) / count,
                    before_snapshot_ms_per_episode=sum(
                        r["before_snapshot_ms"] for r in group
                    )
                    / count,
                    seal_after_predecessor_count=sum(
                        r["seal_lead_ms"] < 0 for r in group
                    ),
                    mean_seal_lead_ms=mean([r["seal_lead_ms"] for r in group]),
                )
                for name, group in grouped.items()
            },
        )
        for name in [
            "gap_ms",
            "gap_lower_ms",
            "gap_upper_ms",
            "before_snapshot_ms",
            "snapshot_to_membership_record_ms",
            "membership_record_to_dispatch_ms",
            "dispatch_to_gpu_start_ms",
        ]:
            summary[name + "_per_episode"] = sum(r[name] for r in selected) / count
        load_only = [
            r for r in selected if "last_successor_kv_after_predecessor_ms" in r
        ]
        summary["load_only_kv_timing"] = {
            name: mean([r[name] for r in load_only])
            for name in [
                "first_successor_kv_after_predecessor_ms",
                "eighth_successor_kv_after_predecessor_ms",
                "last_successor_kv_after_predecessor_ms",
                "all_successor_kv_to_snapshot_ms",
            ]
        }
        summary["load_only_kv_timing"]["transitions"] = len(load_only)
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, cells = [], []
    for study, root in [
        ("main", Path("/mnt/localnvme/glm53_offset_predictor_20261008")),
        ("conditioned", Path("/mnt/localnvme/glm53_offset_conditioned_20261008")),
    ]:
        forecasts = {
            (r["cell"], r["predecessor_step"]): r
            for r in read_lines(root / "audit_live" / "forecasts.jsonl")
        }
        for result in sorted((root / "live").glob("*/r[0-3]_*/result.json")):
            transitions, cell = analyze_cell(result.parent, study, forecasts)
            rows.extend(transitions)
            cells.append(cell)
    report = dict(
        measured_cells=len(cells),
        transitions=len(rows),
        direct_forecast_transitions=sum(r["direct_native_prediction"] for r in rows),
        summary=summarize(rows, cells),
        cells=cells,
        limits=[
            "Gap phases are disjoint clock accounting, not independent causes.",
            "Critical execution spans include CPU submission stalls and output joins.",
            "Snapshot precedes membership accounting; both timestamps are retained.",
            (
                "Counterfactual deadlines hold future arrival/KV/ack traces fixed; "
                "changed membership would change that future."
            ),
            (
                "Observed readiness is the last logged snapshot, "
                "not instantaneous readiness."
            ),
            (
                "Candidate readiness after the actual seal is unobserved; "
                "hypothetical later timers report no readiness count."
            ),
            (
                "Changing prediction can change wakeups and future batches "
                "even when readiness exits first."
            ),
            (
                "No forecast or GPU execution is modified; this retrospective audit "
                "does not identify a causal throughput effect."
            ),
        ],
    )
    (args.output / "attribution.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "transitions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    for r in report["summary"]:
        print(json.dumps(r))
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "measured_cells",
                    "transitions",
                    "direct_forecast_transitions",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
