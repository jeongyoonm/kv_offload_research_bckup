# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit wait strategy and distinguish sealing exposure from total GPU gaps."""

from collections import Counter

import numpy as np


def diagnostics(late, completed, policy):
    windows = [
        point["transport"]
        for point in late
        if point["event"] == "membership_sealed"
        and "adaptive_sleeps" in point["transport"]
    ]
    waits = [wait for window in windows for wait in window["adaptive_waits"]]
    for window in windows:
        assert window["adaptive_poll_us"] == 2000
        assert window["full_prediction_wait"] == (policy != "poll")
        assert window["coalesced_notifications"] == (policy == "full")
        assert window["adaptive_sleeps"] == len(window["adaptive_waits"])
        if policy == "poll":
            assert window["hybrid_coarse_sleeps"] == 0
            assert window["progress_bridge_scans"] == 0
            assert window["progress_bridge_notifications"] == 0
        if policy == "full":
            assert window["progress_bridge_scans"] == 0
    overshoots = [
        max(0, wait["end_ns"] - wait["begin_ns"] - wait["requested_ns"]) / 1e6
        for wait in waits
    ]
    exposed_wait_ns = exposed_hold_ns = 0
    late_seals = []
    deadline_overshoot_ms = []
    for window in windows:
        prediction = window.get("prediction")
        if window.get("compute_seal_deadline_ns") is not None:
            deadline_overshoot_ms.append(
                max(0, window["seal_ns"] - window["compute_seal_deadline_ns"]) / 1e6
            )
        if not prediction:
            continue
        predecessor = completed[prediction["predecessor_step"]]
        end = max(row[3] + row[6] for row in predecessor["execution_ranks"])
        exposure_start = max(end, window["base_ready_ns"])
        exposure = max(0, window["seal_ns"] - exposure_start)
        exposed_hold_ns += exposure
        if exposure:
            late_seals.append(exposure / 1e6)
        for wait in window["adaptive_waits"]:
            exposed_wait_ns += max(
                0,
                min(wait["end_ns"], window["seal_ns"])
                - max(wait["begin_ns"], exposure_start),
            )
    return dict(
        windows=len(windows),
        progress_polls=sum(w["progress_polls"] for w in windows),
        scheduler_wait_returns=len(waits),
        coarse_wait_returns=sum(w["phase"] == "coarse" for w in waits),
        fine_wait_returns=sum(w["phase"] == "fine" for w in waits),
        notification_wait_returns=sum(w["reason"] == "progress" for w in waits),
        progress_bridge_scans=sum(w["progress_bridge_scans"] for w in windows),
        progress_bridge_notifications=sum(
            w["progress_bridge_notifications"] for w in windows
        ),
        poll_cpu_ms=sum(w["adaptive_poll_cpu_ns"] for w in windows) / 1e6,
        bridge_cpu_ms=sum(w["progress_bridge_cpu_ns"] for w in windows) / 1e6,
        total_wait_cpu_ms=sum(
            w["adaptive_poll_cpu_ns"] + w["progress_bridge_cpu_ns"] for w in windows
        )
        / 1e6,
        total_wait_wall_ms=sum(w["end_ns"] - w["begin_ns"] for w in waits) / 1e6,
        oversleep_total_ms=sum(overshoots),
        oversleep_p95_ms=float(np.quantile(overshoots, 0.95)) if overshoots else 0,
        oversleep_max_ms=max(overshoots, default=0),
        compute_exposed_hold_ms=exposed_hold_ns / 1e6,
        compute_exposed_wait_ms=exposed_wait_ns / 1e6,
        late_compute_seals=len(late_seals),
        late_compute_seal_max_ms=max(late_seals, default=0),
        deadline_overshoot_max_ms=max(deadline_overshoot_ms, default=0),
        seal_reasons=dict(Counter(w["seal_reason"] for w in windows)),
    )
