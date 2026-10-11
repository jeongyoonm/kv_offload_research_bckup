# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Count native KV-readiness notifications during useful forward execution."""

from collections import defaultdict


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
        admission_predecessor_classification_complete=classified == activated,
        total_late_admissions=len(activated),
        ready_during_compute_requests=sorted(definite),
        recovered_requests=sorted(recovered),
        scope="Unique CPU-load owners; native all-rank completion-notification "
        "timestamps, not delayed worker job-observation timestamps. "
        "Definite/possible refer to native forward-start boundary uncertainty, "
        "not an exact GPU-copy completion oracle. Load-only admissions are "
        "outside this denominator. Ordinary ready-base scheduling is not counted "
        "as late admission. No baseline ratio without native timing/readiness.",
    )
