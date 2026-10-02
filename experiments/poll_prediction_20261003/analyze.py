# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize measured replay cells, admissions, and bounded exposed hold."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def stats(values):
    if not values:
        return dict(count=0)
    values = sorted(values)
    return dict(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        min=min(values),
        max=max(values),
        total=sum(values),
        p95=values[int(0.95 * (len(values) - 1))],
    )


def analyze(directory):
    result = read(directory / "result.json")
    usage = read(directory / "usage.json")
    ids = {row["request_id"] for row in usage}

    def matches(req):
        return any(request in req for request in ids)

    engine = [
        json.loads(line)
        for line in (directory / "engine.jsonl").read_text().splitlines()
    ]
    late_path = directory / "late.json"
    late = read(late_path)["records"] if late_path.exists() else []
    dispatch = {row["step"]: row for row in engine if row["event"] == "dispatch"}
    completed = {
        row["step"]: row for row in engine if row["event"] == "step_completion"
    }
    compute = sorted(step for step, row in dispatch.items() if row.get("selected"))
    replay_compute = [
        row
        for step, row in dispatch.items()
        if row.get("selected") and any(matches(req) for req in row["selected"])
    ]
    activations = {
        req
        for row in late
        if row["event"] == "membership_sealed"
        for req in row.get("activated", {})
        if matches(req)
    }
    durations = []
    worker_to_gpu = []
    between_compute_lower = []
    between_compute_upper = []
    previous_execution = None
    for row in replay_compute:
        ranks = completed.get(row["step"], {}).get("execution_ranks", [])
        if len(ranks) == 8 and all(r and r[6] for r in ranks):
            durations.append(max(r[6] for r in ranks) / 1e6)
            if all(r[2] and r[3] and r[7] and r[8] for r in ranks):
                if previous_execution is not None:
                    between_compute_lower.append(
                        max(0, min(r[2] for r in ranks) - previous_execution[1]) / 1e6
                    )
                    between_compute_upper.append(
                        max(0, min(r[3] for r in ranks) - previous_execution[0]) / 1e6
                    )
                previous_execution = (
                    max(r[7] for r in ranks),
                    max(r[8] for r in ranks),
                )
            primary = completed.get(row["step"], {}).get("ranks", [])
            if len(primary) == 8 and all(r and r[1] for r in primary):
                worker_to_gpu.append(
                    (max(r[3] for r in ranks) - max(r[1] for r in primary)) / 1e6
                )
    holds, exposure_lower, exposure_upper, windows = [], [], [], []
    reasons = Counter()
    predictions = []
    polling = []
    seal_to_entry = []
    seal_to_gpu = []
    for row in late:
        if row["event"] != "membership_sealed":
            continue
        transport = row.get("transport", {})
        if transport.get("adaptive_sleeps"):
            polling.append(
                {
                    k: transport.get(k, 0)
                    for k in (
                        "adaptive_poll_us",
                        "progress_polls",
                        "adaptive_sleeps",
                        "adaptive_poll_cpu_ns",
                        "adaptive_sleep_ns",
                        "adaptive_requested_sleep_ns",
                        "hybrid_coarse_sleeps",
                    )
                }
            )
        base, seal = transport.get("base_ready_ns"), transport.get("seal_ns")
        if not base or not seal:
            continue
        # Include empty load dispatches during replay; they can expose holds too.
        preceding = [step for step in compute if step < row["step"]]
        a_step = preceding[-1] if preceding else None
        a_data = completed.get(a_step, {})
        reasons[transport.get("seal_reason", "ordinary")] += 1
        held = (seal - base) / 1e6
        holds.append(held)
        ranks = a_data.get("execution_ranks", [])
        window = dict(
            step=row["step"],
            a_step=a_step,
            hold_ms=held,
            base_ns=base,
            seal_ns=seal,
            reason=transport.get("seal_reason"),
        )
        b = completed.get(row["step"], {})
        b_primary = b.get("ranks", [])
        b_execution = b.get("execution_ranks", [])
        if len(b_primary) == 8 and all(r and r[1] for r in b_primary):
            entry = max(r[1] for r in b_primary)
            seal_to_entry.append((entry - seal) / 1e6)
            window["b_entry_ns"] = entry
        if len(b_execution) == 8 and all(r and r[3] for r in b_execution):
            gpu = max(r[3] for r in b_execution)
            seal_to_gpu.append((gpu - seal) / 1e6)
            window["b_gpu_start_ns"] = gpu
        if len(ranks) == 8 and all(r and r[7] and r[8] for r in ranks):
            end_lower = max(r[7] for r in ranks)
            end_upper = max(r[8] for r in ranks)
            lo = max(0, seal - max(base, end_upper)) / 1e6
            hi = max(0, seal - max(base, end_lower)) / 1e6
            exposure_lower.append(lo)
            exposure_upper.append(hi)
            window.update(
                exposed_lower_ms=lo,
                exposed_upper_ms=hi,
                a_end_lower_ns=end_lower,
                a_end_upper_ns=end_upper,
                no_compute_at_base_ready=end_upper <= base,
                a_start_lower_ns=max(r[2] for r in ranks),
                a_start_upper_ns=max(r[3] for r in ranks),
                base_relative_to_a_start_ms=(base - max(r[3] for r in ranks)) / 1e6,
                seal_relative_to_a_start_ms=(seal - max(r[3] for r in ranks)) / 1e6,
                seal_relative_to_a_end_ms=(seal - end_upper) / 1e6,
            )
        elif a_step is None:
            exposure_lower.append(held)
            exposure_upper.append(held)
            window.update(
                exposed_lower_ms=held, exposed_upper_ms=held, no_preceding_compute=True
            )
        if transport.get("prediction"):
            prediction = transport["prediction"]
            actual_ranks = completed.get(prediction["predecessor_step"], {}).get(
                "execution_ranks", []
            )
            if len(actual_ranks) == 8 and all(r and r[6] for r in actual_ranks):
                actual = max(r[6] for r in actual_ranks)
                predictions.append(
                    dict(
                        predicted_ms=prediction["predicted_duration_ns"] / 1e6,
                        actual_ms=actual / 1e6,
                        ratio=prediction["predicted_duration_ns"] / actual,
                        step=row["step"],
                        preparation_conditioned=prediction.get(
                            "preparation_conditioned", False
                        ),
                        preparation_reserve_ms=prediction.get(
                            "preparation_reserve_ns", 0
                        )
                        / 1e6,
                        actual_b_preparation_ms=(
                            max(r[2] - r[1] for r in b_primary) / 1e6
                            if len(b_primary) == 8
                            and all(r and r[1] and r[2] >= r[1] for r in b_primary)
                            else None
                        ),
                    )
                )
        windows.append(window)
    ready_times = {}
    for row in engine:
        if row["event"] in ("kv_all_rank_completion", "kv_ready_observed"):
            req = row.get("request", row.get("early_promoted_request"))
            if req and matches(req):
                ready_times[req] = row.get("completion_ns", row.get("time_ns"))
    eligibility = {r["step"]: r for r in late if r["event"] == "seal_eligibility"}
    membership = {r["step"]: r for r in late if r["event"] == "membership_sealed"}
    published = {req for r in eligibility.values() for req in r.get("candidates", [])}
    missed_at_entry = set()
    missed_before_a_end_lower = set()
    missed_before_a_end_upper = set()
    productive_exposed = []
    unacknowledged = 0
    for window in windows:
        sealed = membership[window["step"]]
        candidates = eligibility.get(window["step"], {}).get("candidates", [])
        selected = sealed.get("selected", {})
        activated = set(sealed.get("activated", {}))
        base_ids = set(selected) - activated
        statuses = sealed.get("transport", {}).get("statuses", [])
        acked = [
            req
            for slot, req in enumerate(candidates)
            if len(statuses) >= 8 * (slot + 2)
            and statuses[8 * (slot + 1) : 8 * (slot + 2)] == [1] * 8
        ]
        unacknowledged += len(candidates) - len(acked)
        window.update(
            base_request_count=len(base_ids),
            base_tokens=sum(selected[req] for req in base_ids),
            published_candidates=len(candidates),
            late_requests=len(activated),
        )
        entry = window.get("b_entry_ns")
        missed = [
            req
            for req in acked
            if req not in activated
            and entry
            and ready_times.get(req, entry + 1) <= entry
        ]
        missed_at_entry.update(missed)
        window["missed_ready_at_entry_with_seal_ack"] = missed
        for bound, destination in (
            ("lower", missed_before_a_end_lower),
            ("upper", missed_before_a_end_upper),
        ):
            end = window.get(f"a_end_{bound}_ns", 0)
            recoverable = [
                req
                for req in acked
                if req not in selected
                and req not in sealed.get("transport", {}).get("cancelled", [])
                and window["seal_ns"] < ready_times.get(req, end + 1) <= end
            ]
            destination.update(recoverable)
            window[f"missed_candidates_ready_before_a_end_{bound}"] = recoverable
        # This is a diagnostic threshold, not an assumption that four is optimal.
        needed = max(0, 4 - len(base_ids))
        ready = sorted(ready_times[req] for req in acked if req in ready_times)
        if not needed or len(ready) >= needed:
            threshold = window["base_ns"] if not needed else ready[needed - 1]
            if "exposed_lower_ms" in window:
                begin = max(
                    window["base_ns"], window.get("a_end_upper_ns", 0), threshold
                )
                value = max(0, window["seal_ns"] - begin) / 1e6
                productive_exposed.append(value)
                window["exposed_with_four_requests_ready_ms"] = value
    # Count all-rank transfer completions inside preceding compute intervals.
    opportunities = set()
    intervals = []
    for row in replay_compute:
        ranks = completed.get(row["step"], {}).get("execution_ranks", [])
        if len(ranks) == 8 and all(r and r[3] and r[8] for r in ranks):
            intervals.append((max(r[3] for r in ranks), max(r[8] for r in ranks)))
    for req, ready in ready_times.items():
        if ready and any(start <= ready <= end for start, end in intervals):
            opportunities.add(req)
    result.update(
        compute_calls=len(replay_compute),
        dispatch_calls=len(dispatch),
        load_only_dispatch_calls=sum(
            not row.get("selected") and row.get("load_state", [0])[0] > 0
            for row in dispatch.values()
        ),
        no_work_dispatch_calls=sum(
            not row.get("selected") and row.get("load_state", [0])[0] == 0
            for row in dispatch.values()
        ),
        late_admissions=len(activations),
        recovered_compute_opportunities=len(opportunities & activations),
        kv_ready_during_compute=len(opportunities),
        kv_completed_requests=len(ready_times),
        kv_ready_outside_compute=len(set(ready_times) - opportunities),
        late_admissions_outside_compute=len(activations - opportunities),
        published_candidate_requests=len(published),
        compute_opportunities_not_published=len(opportunities - published),
        missed_ready_at_worker_entry_with_seal_ack=len(missed_at_entry),
        missed_candidates_ready_before_a_end_lower=len(missed_before_a_end_lower),
        missed_candidates_ready_before_a_end_upper=len(missed_before_a_end_upper),
        unacknowledged_candidate_instances_at_seal=unacknowledged,
        exposed_with_four_requests_ready_ms=stats(productive_exposed),
        compute_gpu_ms=stats(durations),
        worker_entry_to_gpu_ms=stats(worker_to_gpu),
        between_compute_gap_lower_ms=stats(between_compute_lower),
        between_compute_gap_upper_ms=stats(between_compute_upper),
        seal_to_worker_entry_ms=stats(seal_to_entry),
        seal_to_gpu_start_ms=stats(seal_to_gpu),
        hold_ms=stats(holds),
        exposed_lower_ms=stats(exposure_lower),
        exposed_upper_ms=stats(exposure_upper),
        seal_reasons=dict(reasons),
        prediction_samples=predictions,
        polling_samples=polling,
        adaptive_sleeps=sum(r["adaptive_sleeps"] for r in polling),
        adaptive_poll_cpu_ms=sum(r["adaptive_poll_cpu_ns"] for r in polling) / 1e6,
        adaptive_sleep_ms=sum(r["adaptive_sleep_ns"] for r in polling) / 1e6,
        hybrid_coarse_sleeps=sum(r["hybrid_coarse_sleeps"] for r in polling),
        windows=windows,
        matched_exposure_windows=len(exposure_lower),
    )
    (directory / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    from validate import validate_root

    validate_root(args.root)
    results = []
    for path in sorted(args.root.glob("**/result.json")):
        try:
            results.append(analyze(path.parent))
        except Exception as error:
            print(path, type(error).__name__, str(error))
            raise
    (args.root / "analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    for row in results:
        print(
            Path(row["path"]).parent.name,
            Path(row["path"]).name,
            round(row["throughput"], 2),
            row["compute_calls"],
            row["late_admissions"],
            round(row["exposed_lower_ms"].get("total", 0), 1),
            row["accepted"],
        )


if __name__ == "__main__":
    main()
