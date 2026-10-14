# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Count native KV-readiness notifications during useful forward execution."""

from collections import defaultdict


def execution_bounds(compute, windows):
    intervals, widths = [], []
    for event in compute:
        ranks = event["execution_ranks"]
        assert len(ranks) == 8
        for r in ranks:
            assert r and len(r) == 9 and r[0] == event["step"] and r[5] == 1
            assert 0 < r[1] <= r[2] <= r[3] <= r[4] <= event["time_ns"]
            assert r[6] > 0 and r[2] <= r[7] <= r[8] <= r[4]
        widths.append(max(r[3] - r[2] for r in ranks))
        intervals.append(
            (
                max(r[2] for r in ranks),
                max(r[3] for r in ranks),
                max(r[2] + r[6] for r in ranks),
                max(r[3] + r[6] for r in ranks),
            )
        )
    intervals.sort(key=lambda interval: interval[1])
    lower = upper = 0
    for start, end in windows:
        measured = [i for i in intervals if start <= i[1] <= end]
        for a, b in zip(measured, measured[1:]):
            lower += max(0, b[0] - a[3])
            upper += max(0, b[1] - a[2])
    return dict(
        native_execution_snapshots=len(compute),
        start_bracket_above_predictor_2ms=sum(w > 2_000_000 for w in widths),
        max_start_bracket_ms=max(widths) / 1e6 if widths else None,
        inter_compute_gap_lower_ms=lower / 1e6 if intervals else None,
        inter_compute_gap_upper_ms=upper / 1e6 if intervals else None,
        gpu_envelope_lower_ms=sum(max(0, i[2] - i[1]) for i in intervals) / 1e6
        if intervals
        else None,
        gpu_envelope_upper_ms=sum(max(0, i[3] - i[0]) for i in intervals) / 1e6
        if intervals
        else None,
        scope="Retrospective bounds preserve wider native start brackets; "
        "runtime predictor still rejects brackets above its unchanged2ms limit.",
    )


def summarize(events, protocol, compute, windows):
    owners = defaultdict(set)
    notifications = {}
    rank_notifications = defaultdict(dict)
    for event in events:
        if event["event"] == "job_observed_done":
            owners[event["job_id"]].add(event["request"])
        elif event["event"] == "early_all_ranks_ready":
            assert event["ranks"] == 8
            notifications[event["job_id"]] = event["completion_ns"]
        elif event["event"] == "early_notification_received":
            assert event["status"] == 1
            rank_notifications[event["job_id"]][event["rank"]] = event["completion_ns"]
    for job, timestamps in rank_notifications.items():
        if len(timestamps) == 8:
            assert set(timestamps) == set(range(8))
            completion = max(timestamps.values())
            assert job not in notifications or notifications[job] == completion
            notifications[job] = completion
    if not notifications:
        return dict(
            available=False,
            observed_load_jobs=len(owners),
            native_ready_jobs=0,
            native_ready_coverage_complete=False,
            missing_native_jobs=sorted(owners),
            late_admission_ratio_percent=None,
            reason="No complete native all-rank readiness records"
            if rank_notifications
            else "Native early-readiness channel absent",
        )
    assert all(len(requests) == 1 for requests in owners.values())
    assert set(notifications) <= set(owners), "Readiness notification has no load owner"
    ready = defaultdict(list)
    for job, timestamp in notifications.items():
        if any(a <= timestamp <= b for a, b in windows):
            ready[next(iter(owners[job]))].append(timestamp)
    activated = {
        request
        for event in protocol
        if event["event"] == "membership_sealed"
        for request in event.get("activated", {})
    }
    compute_activated = {
        request
        for event in protocol
        if event["event"] == "membership_sealed"
        and event.get("transport", {}).get("has_compute") is True
        for request in event.get("activated", {})
    }
    classified = {
        request
        for event in protocol
        if event["event"] == "membership_sealed"
        and type(event.get("transport", {}).get("has_compute")) is bool
        for request in event.get("activated", {})
    }
    certain, possible = [], []
    for step in compute:
        ranks = step["execution_ranks"]
        assert len(ranks) == 8 and all(r for r in ranks)
        certain.append((max(r[3] for r in ranks), max(r[2] + r[6] for r in ranks)))
        possible.append((max(r[2] for r in ranks), max(r[3] + r[6] for r in ranks)))

    def during(intervals):
        return {
            request
            for request, timestamps in ready.items()
            if any(a <= t <= b for a, b in intervals for t in timestamps)
        }

    definite = during(certain)
    potential = during(possible)
    recovered = compute_activated & definite
    unknown = {
        next(iter(requests))
        for job, requests in owners.items()
        if job not in notifications
    } - definite
    uncertain = unknown | (potential - definite)
    possible_recovered = compute_activated & uncertain
    largest_denominator = len(definite) + len(uncertain)
    upper_denominator = len(definite) + len(possible_recovered)
    return dict(
        available=True,
        observed_load_jobs=len(owners),
        native_ready_jobs=len(notifications),
        native_ready_coverage_complete=set(notifications) == set(owners),
        unique_cpu_load_owners_in_roi=len(ready),
        ready_during_compute_definite=len(definite),
        ready_during_compute_possible=len(potential),
        timing_boundary_ambiguous=len(potential - definite),
        late_admitted_during_compute_definite=len(recovered),
        late_admission_ratio_percent=100 * len(recovered) / len(definite)
        if definite and set(notifications) == set(owners) and classified == activated
        else None,
        missing_native_jobs=sorted(set(owners) - set(notifications)),
        missing_native_unique_requests=len(unknown),
        admission_predecessor_classification_complete=classified == activated,
        admission_ratio_lower_bound_percent=100 * len(recovered) / largest_denominator
        if largest_denominator and classified == activated
        else None,
        admission_ratio_upper_bound_percent=100
        * (len(recovered) + len(possible_recovered))
        / upper_denominator
        if upper_denominator and classified == activated
        else None,
        total_late_admissions=len(activated),
        ready_during_compute_requests=sorted(definite),
        recovered_requests=sorted(recovered),
        scope="Unique CPU-load owners; native all-rank completion-notification "
        "timestamps, not delayed worker job-observation timestamps. "
        "Definite/possible refer to native forward-start boundary uncertainty, "
        "not an exact GPU-copy completion oracle. Load-only admissions are "
        "outside this denominator. Ordinary ready-base scheduling is not counted "
        "as late admission. Bounds allow missing notifications and start-boundary "
        "ambiguity; they do not impute timestamps. No baseline ratio without "
        "native timing/readiness.",
    )
