# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for late admission publication, membership, and retirement."""

import pickle
import queue
import struct
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext, suppress
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.v1.kv_connector.unit.offloading_connector.conftest import (
    request_runner as offloading_request_runner,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_offload.late_admission import AdmissionWindow, Candidate
from vllm.v1.kv_offload.late_admission_runtime import (
    CandidateNotice,
    PreparationLane,
    signal_prepared,
    stage_rpc,
)
from vllm.v1.request import RequestStatus

request_runner = offloading_request_runner


class ControlledExecutor:
    execute_model: Any

    def __init__(self, ranks=8, candidate_before_base_last=False):
        self.ranks = ranks
        self.candidate_before_base_last = candidate_before_base_last
        self.published = threading.Event()
        self.release_base = [threading.Event() for _ in range(ranks)]
        self.release_candidate = [threading.Event() for _ in range(ranks)]
        self.prepared_base = [threading.Event() for _ in range(ranks)]
        self.prepared_candidate = [threading.Event() for _ in range(ranks)]
        self.pool = ThreadPoolExecutor(max_workers=ranks)
        self.workers = []
        self.path = None

    def publish_late_preparation(self, data_path):
        with open(data_path, "rb") as source:
            _, step, base, candidates, self.path = pickle.load(source)
        assert base and len(candidates) <= 16
        self.published.set()

        def prepare(rank):
            if rank == 7 and self.candidate_before_base_last and candidates:
                assert self.release_candidate[rank].wait(5)
                for slot in range(len(candidates)):
                    signal_prepared(self.path, step, rank, 1, slot)
                self.prepared_candidate[rank].set()
            assert self.release_base[rank].wait(5)
            with suppress(OSError):
                signal_prepared(self.path, step, rank, 1)
            self.prepared_base[rank].set()
            if (
                candidates
                and not (rank == 7 and self.candidate_before_base_last)
                and self.release_candidate[rank].wait(5)
            ):
                with suppress(OSError):
                    for slot in range(len(candidates)):
                        signal_prepared(self.path, step, rank, 1, slot)
                self.prepared_candidate[rank].set()

        self.workers = [self.pool.submit(prepare, rank) for rank in range(self.ranks)]

    def finish(self):
        for release in self.release_base + self.release_candidate:
            release.set()
        for worker in self.workers:
            worker.result(timeout=5)
        self.pool.shutdown(wait=True)


class HeldPredecessor:
    def __init__(self):
        self.release = threading.Event()

    def poll(self):
        return self.release.is_set()


def _core(executor, predecessor=None):
    predecessor = predecessor or HeldPredecessor()
    return SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(world_size=8)),
        model_executor=executor,
        scheduler=SimpleNamespace(
            _late_generation=1, finished_recving_kv_req_ids=set()
        ),
        batch_queue=deque(
            [
                (
                    predecessor,
                    SimpleNamespace(total_num_scheduled_tokens=0),
                    predecessor,
                )
            ]
        ),
        aborts_queue=queue.Queue(),
        _refresh_early_load_readiness=MagicMock(),
    )


def _request(req_id):
    return SimpleNamespace(req_id=req_id, prefill_token_ids=[1, 2, 3], prompt_len=3)


@pytest.fixture
def deadline_core(monkeypatch):
    """Drive the real seal loop with deterministic clocks and rank acknowledgments."""
    import numpy as np

    from vllm.v1.kv_offload import late_admission_runtime as late

    clock = SimpleNamespace(now=1_001_000_000, sleeps=[])

    def sleep(seconds):
        ns = round(seconds * 1e9)
        assert ns >= 0
        clock.sleeps.append(ns)
        clock.now += ns

    monkeypatch.setattr(
        late,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now / 1e9,
            monotonic_ns=lambda: clock.now,
            thread_time_ns=lambda: 0,
            sleep=sleep,
        ),
    )

    def make(predicted=120_000_000, lead_us=0, shadow=False):
        path = []

        def publish(data_path):
            with open(data_path, "rb") as stream:
                _, step, _, candidates, ack_path = pickle.load(stream)
            path.append(ack_path)
            for rank in range(8):
                if rank != 7 or clock.base_at is None:
                    signal_prepared(ack_path, step, rank, 1)
                for slot, candidate in enumerate(candidates):
                    if rank != 7 or candidate.req_id != "partial":
                        signal_prepared(ack_path, step, rank, 1, slot)

        core = _core(SimpleNamespace(publish_late_preparation=publish))
        scheduler = core.scheduler
        scheduler._late_seal_policy = "predictive"
        scheduler._late_seal_prediction_to_readiness = True
        scheduler._late_seal_compute_deadline = not shadow
        scheduler._late_seal_compute_deadline_shadow = shadow
        scheduler._late_seal_compute_deadline_lead_us = lead_us
        scheduler._late_seal_target_candidates = 16
        scheduler._late_seal_repeat_us = 25_000
        scheduler._late_seal_prediction_scale_percent = 100
        scheduler._late_seal_prediction_reserve_us = 10_000
        scheduler._late_seal_safety_us = 1_000_000
        scheduler._late_seal_adaptive_poll_us = 2000
        clock.ready_at = {}
        clock.complete_at = None
        clock.base_at = None
        scheduler.requests = {}
        output = core.batch_queue[-1][1]
        output.total_num_scheduled_tokens = 1024
        output.readiness_timeline_step = 8

        def records(step):
            end = clock.complete_at
            completed = end is not None and clock.now >= end
            return [
                [
                    step,
                    1_000_000_000,
                    1_000_000_000,
                    1_000_000_000,
                    clock.now,
                    1,
                    end - 1_000_000_000 if completed else 0,
                    end if completed else 0,
                    end if completed else 0,
                ]
            ] * 8

        core.deferral_timing = SimpleNamespace(
            steps={8: {"step": 8, "request_shapes": [[1024, 8192]]}},
            read_execution=records,
            seal_estimator=SimpleNamespace(
                predict=lambda data: predicted,
                features=lambda data: np.ones(4),
                samples=[0] * 8,
                preparation_reserve=lambda extra: extra,
                preparation_ns=[],
            ),
        )

        def refresh(**kwargs):
            scheduler.finished_recving_kv_req_ids.update(
                req for req, at in clock.ready_at.items() if at <= clock.now
            )
            if clock.base_at is not None and clock.now >= clock.base_at:
                signal_prepared(path[0], 9, 7, 1)

        core._refresh_early_load_readiness.side_effect = refresh
        return core, clock

    return make


@pytest.mark.parametrize(
    "supported,base_at,margin",
    [
        (True, None, False),
        (False, None, False),
        (True, 1_060_000_000, False),
        (True, None, True),
        (False, None, True),
    ],
)
def test_frozen_shape_correction_controls_deadline_and_preserves_snapshot(
    deadline_core, supported, base_at, margin
):
    """A loaded calibration advances the deadline without changing admission gates."""
    import numpy as np

    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    core, clock = deadline_core(lead_us=50_000)
    clock.base_at = base_at
    core.scheduler._late_seal_prediction_bias_us = 90_000
    x = np.zeros(28)
    context = 15552 if supported else 8192
    x[:9] = [832, 1, 0, context, 832, 832, context, context, 896]
    state = dict(
        schema=3,
        features=["intercept"],
        coefficients=[30],
        support=dict(requests=[1, 32], context=[15552, 58368], suffix=[832, 3072]),
        geometry_support=dict(suffix_per_context=[0.01, 0.25]),
        bias_bounds_ms=[-500, 500],
        provenance={},
    )
    if margin:
        state.update(
            schema=4,
            deadline_margin=dict(
                finish_allowance_ms=dict(small_short=20),
                setup_allowance_ms=dict(small_short=10),
                finish_fallback_ms=40,
                setup_fallback_ms=35,
                extra_ms=2,
                bounds_ms=[10, 100],
            ),
        )
        core.scheduler._late_base_request_shapes = [[832, 15552]]
        core.scheduler.requests = {
            rid: SimpleNamespace(num_tokens=16384, num_computed_tokens=15552)
            for rid in ("early", "late")
        }
    calibration = BatchShapeCalibration(state)
    core.deferral_timing.seal_estimator.latency_surface = SimpleNamespace(
        requires_forecast=False,
        predict_with_features=lambda data: (120_000_000, x),
        duration_calibration=calibration,
        target_schema="cuda_span_max.v1",
    )
    lead = 32_000_000 if supported and margin else 50_000_000
    expected = max(1_090_000_000 - lead if supported else clock.now, base_at or 0)
    clock.ready_at = {"early": expected - 1, "late": expected + 1}
    acknowledged = stage_rpc(
        core, 9, [_request("base")], [_request("early"), _request("late")]
    )
    transport = core.scheduler._late_transport
    assert transport["seal_ns"] == expected
    assert transport["seal_reason"] == "compute_seal_deadline"
    assert transport["prediction"]["applied_prediction_bias_ns"] == (
        30_000_000 if supported else 90_000_000
    )
    assert transport["prediction"]["shape_calibration_fallback"] is not supported
    assert transport["compute_seal_deadline_lead_us"] == lead // 1000
    if margin:
        assert (
            transport["prediction"]["shape_deadline_margin_fallback"] is not supported
        )
    assert set(acknowledged) & core.scheduler._late_kv_snapshot == {"early"}
    assert transport["statuses"][:8] == [1] * 8
    clock.base_at = None  # The preparation acknowledgment file is retired at seal.
    clock.now += 1
    core._refresh_early_load_readiness()
    assert "late" in core.scheduler.finished_recving_kv_req_ids
    assert "late" not in core.scheduler._late_kv_snapshot
    assert not core.batch_queue[-1][0].poll()


@pytest.mark.parametrize("lead_us", [0, 5000, 10_000, 20_000])
def test_compute_deadline_seals_while_a_is_held_and_clips_sleep(deadline_core, lead_us):
    """A deadline is an enforced seal, rather than a transition to more polling."""
    core, clock = deadline_core(lead_us=lead_us)
    stage_rpc(core, 9, [_request("base")], [_request("a")])
    transport = core.scheduler._late_transport
    expected = 1_120_000_000 - lead_us * 1000
    assert transport["seal_reason"] == "compute_seal_deadline"
    assert transport["seal_ns"] == expected
    assert transport["compute_seal_deadline_ns"] == expected
    assert transport["compute_seal_deadline_overshoot_ns"] == 0
    assert max(clock.sleeps) <= 2_000_000
    assert not core.batch_queue[-1][0].poll()


@pytest.mark.parametrize("setup_ms", [0, 20, 80, None])
def test_learned_setup_deadline_uses_causal_proposal_and_falls_back(
    deadline_core, setup_ms
):
    """Learned B readiness replaces the lead, preserving held A and safety gates."""
    import numpy as np

    core, clock = deadline_core(lead_us=5000)
    core.scheduler._late_seal_condition_preparation = True
    core.scheduler._late_record_enabled = True
    core.scheduler._late_base_request_shapes = [[832, 15552]]
    estimator = core.deferral_timing.seal_estimator
    estimator.preparation_reserve = lambda extra, data=None: extra
    features = np.zeros(28)
    features[[0, 1, 5, 7]] = [1024, 1, 1024, 8192]
    seen = []

    def predict(x, native=True):
        seen.append(x.tolist())
        return setup_ms

    estimator.latency_surface = SimpleNamespace(
        requires_forecast=False,
        predict_with_features=lambda data: (120_000_000, features),
        target_schema="critical_end_after_all_rank_start.v1",
        setup_predictor=SimpleNamespace(model_id="test", predict_ms=predict),
    )
    deadline = 1_120_000_000 - round((setup_ms if setup_ms is not None else 5) * 1e6)
    clock.ready_at = {
        "ready": deadline - 1,
        "late": deadline + 1,
        "partial": deadline - 1,
        "cancelled": deadline - 1,
    }
    core.aborts_queue.put(["cancelled"])
    acked = stage_rpc(
        core, 9, [_request("base")], [_request(req) for req in clock.ready_at]
    )
    transport = core.scheduler._late_transport
    assert transport["seal_ns"] == deadline
    assert transport["compute_seal_deadline_ns"] == deadline
    assert transport["setup_prediction"]["fallback"] == (setup_ms is None)
    assert seen and seen[0][:5] == [1, 832, 15552, 832, 15552]
    assert seen[0][7] == 4
    assert seen[0][8:12] == [1, 1024, 1024, 8192]
    eligible = (
        set(acked) & core.scheduler._late_kv_snapshot
    ) - core.scheduler._late_cancelled
    assert eligible == {"ready"}
    clock.now += 1
    core._refresh_early_load_readiness()
    assert "late" not in core.scheduler._late_kv_snapshot
    assert not core.batch_queue[-1][0].poll()


@pytest.mark.parametrize(
    "bias_us,lead_us,expected",
    [
        (0, 5000, 1_115_000_000),
        (90_000, 5000, 1_025_000_000),
        (0, 100_000, 1_020_000_000),
        (90_000, 100_000, 1_001_000_000),
        (300_000, 5000, 1_001_000_000),
    ],
)
def test_prediction_bias_and_setup_lead_seal_without_waiting_for_a(
    deadline_core, bias_us, lead_us, expected
):
    """Correction and setup reserve advance sealing, retaining the base-ack gate."""
    core, _ = deadline_core(lead_us=lead_us)
    core.scheduler._late_seal_prediction_bias_us = bias_us
    stage_rpc(core, 9, [_request("base")], [_request("pending")])
    transport = core.scheduler._late_transport
    prediction = transport["prediction"]
    assert transport["seal_ns"] == expected
    assert transport["seal_reason"] == "compute_seal_deadline"
    assert prediction["raw_scaled_duration_ns"] == 120_000_000
    assert prediction["predicted_duration_ns"] == max(1, 120_000_000 - bias_us * 1000)
    assert prediction["prediction_bias_us"] == bias_us
    assert not core.batch_queue[-1][0].poll()


@pytest.mark.parametrize("duration", [90_000_000, 180_000_000, 2_000_000_000])
@pytest.mark.parametrize("bias_us", [0, 10_000])
def test_refinement_moves_seal_once_without_extending_safety_cap(
    deadline_core, duration, bias_us
):
    """Only observed all-rank CPU submission can revise an unexpired deadline."""
    import numpy as np

    core, clock = deadline_core()
    core.scheduler._late_seal_prediction_bias_us = bias_us
    core.scheduler._late_record_enabled = True
    estimator = core.deferral_timing.seal_estimator
    estimator.refinement_surface = SimpleNamespace(
        predict=lambda data: duration,
        input_features=lambda data: np.ones(29),
    )
    calls = []

    def read(step):
        submitted = 1_050_000_000 if clock.now >= 1_050_000_000 else 0
        calls.append(clock.now)
        return (
            True,
            True,
            [[step, 1_000_000_000, 1_001_000_000, submitted, 0, 0, clock.now, 1, 0]]
            * 8,
        )

    core.deferral_timing.read = read
    core.deferral_timing.loads = SimpleNamespace(
        forecast_features=lambda data, now: [0] * 10
    )
    stage_rpc(core, 9, [_request("base")], [_request("pending")])
    transport = core.scheduler._late_transport
    expected = min(1_000_000_000 + duration - bias_us * 1000, 2_001_000_000)
    assert transport["seal_ns"] == expected
    assert transport["prediction"]["predicted_duration_ns"] == (
        120_000_000 - bias_us * 1000
    )
    assert transport["refinement"]["raw_scaled_duration_ns"] == duration
    assert transport["refinement"]["predicted_duration_ns"] == duration - bias_us * 1000
    records = core.scheduler._late_admission_records
    assert sum(r["event"] == "seal_prediction_refined" for r in records) == 1
    assert all(b - a >= 10_000_000 for a, b in zip(calls, calls[1:]))
    assert max(clock.sleeps) <= 2_000_000
    assert not core.batch_queue[-1][0].poll()


@pytest.mark.parametrize("stop", ["partial", "compute", "candidate", "expired"])
def test_refinement_preserves_prior_stop_conditions(deadline_core, stop):
    """Partial submission and earlier stops cannot be turned into longer waits."""
    import numpy as np

    core, clock = deadline_core()
    calls = []

    def predict(data):
        calls.append(clock.now)
        if stop == "expired":
            clock.now += 70_000_000
        return 180_000_000

    core.deferral_timing.seal_estimator.refinement_surface = SimpleNamespace(
        predict=predict, input_features=lambda data: np.ones(29)
    )

    def read(step):
        submitted = 1_050_000_000 if clock.now >= 1_050_000_000 else 0
        records = [
            [step, 1_000_000_000, 1_001_000_000, submitted, 0, 0, clock.now, 1, 0]
            for _ in range(8)
        ]
        if stop == "partial":
            records[-1][3] = 0
        return True, True, records

    core.deferral_timing.read = read
    core.deferral_timing.loads = SimpleNamespace(
        forecast_features=lambda data, now: [0] * 10
    )
    if stop == "compute":
        clock.complete_at = 1_040_000_000
    elif stop == "candidate":
        clock.ready_at = {"pending": 1_040_000_000}
    stage_rpc(core, 9, [_request("base")], [_request("pending")])
    transport = core.scheduler._late_transport
    assert transport["refinement"] is None
    assert transport["compute_seal_deadline_ns"] == 1_120_000_000
    if stop in {"compute", "candidate"}:
        assert transport["seal_ns"] < 1_050_000_000
        assert not calls
    elif stop == "partial":
        assert transport["seal_ns"] == 1_120_000_000
        assert not calls
    else:
        assert transport["seal_ns"] >= 1_120_000_000
        assert len(calls) == 1


def test_compute_deadline_freezes_kv_ack_and_cancellation_snapshot(deadline_core):
    """After-deadline readiness cannot expand the sealed membership snapshot."""
    core, clock = deadline_core()
    clock.ready_at = {
        "a": 1_119_999_999,
        "late": 1_120_000_001,
        "partial": 1_119_999_999,
        "cancelled": 1_119_999_999,
    }
    core.aborts_queue.put(["cancelled"])
    acknowledged = stage_rpc(
        core,
        9,
        [_request("base")],
        [_request(req) for req in clock.ready_at],
    )
    scheduler = core.scheduler
    eligible = (
        set(acknowledged) & scheduler._late_kv_snapshot
    ) - scheduler._late_cancelled
    assert eligible == {"a"}
    clock.now += 1
    core._refresh_early_load_readiness()
    assert "late" in scheduler.finished_recving_kv_req_ids
    assert "late" not in scheduler._late_kv_snapshot
    assert scheduler._late_transport["seal_reason"] == "compute_seal_deadline"


@pytest.mark.parametrize("predicted", [None, 2_000_000_000])
def test_compute_deadline_uses_maximum_wait_when_prediction_is_missing_or_late(
    deadline_core, predicted
):
    core, _ = deadline_core(predicted=predicted)
    stage_rpc(core, 9, [_request("base")], [_request("a")])
    transport = core.scheduler._late_transport
    assert transport["seal_reason"] == "adaptive_safety_deadline"
    assert transport["seal_ns"] == transport["base_ready_ns"] + 1_000_000_000
    assert transport["effective_seal_expiry_ns"] == transport["seal_ns"]


def test_compute_deadline_waits_for_all_base_acknowledgments(deadline_core):
    core, clock = deadline_core()
    clock.base_at = 1_170_000_000
    stage_rpc(core, 9, [_request("base")], [_request("a")])
    transport = core.scheduler._late_transport
    assert transport["seal_ns"] >= clock.base_at
    assert transport["seal_reason"] == "compute_seal_deadline"
    assert transport["statuses"][:8] == [1] * 8


@pytest.mark.parametrize("shadow", [False, True])
def test_compute_deadline_preserves_earlier_compute_completion(deadline_core, shadow):
    core, clock = deadline_core(shadow=shadow)
    clock.complete_at = 1_060_000_000
    stage_rpc(core, 9, [_request("base")], [_request("a")])
    transport = core.scheduler._late_transport
    assert transport["seal_reason"] == "predecessor_compute_finished"
    assert transport["seal_ns"] < transport["compute_seal_deadline_ns"]


def test_compute_deadline_shadow_does_not_enforce_expiry(deadline_core):
    core, clock = deadline_core(shadow=True)
    clock.ready_at = {"a": 1_150_000_000}
    stage_rpc(core, 9, [_request("base")], [_request("a")])
    transport = core.scheduler._late_transport
    assert transport["seal_reason"] == "candidate_target_ready"
    assert transport["seal_ns"] > transport["compute_seal_deadline_ns"]


def test_submission_diagnostics_capture_only_observed_changes_without_moving_seal(
    deadline_core, monkeypatch
):
    """Tracing sees CPU submission when published, with no extra scheduler waits."""
    monkeypatch.setenv("VLLM_DEFERRAL_PREDICTOR_PROGRESS", "1")
    core, clock = deadline_core()
    core.scheduler._late_record_enabled = True
    at = 1_030_000_000

    def primary(step):
        submitted = at if clock.now >= at else 0
        return (
            True,
            False,
            [[step, 1_000_000_000, 1_000_000_000, submitted, 0, 0, clock.now, 1, 0]]
            * 8,
        )

    core.deferral_timing.read = primary
    core.deferral_timing.loads = SimpleNamespace(
        forecast_features=lambda data, now: [0] * 10
    )
    stage_rpc(core, 9, [_request("base")], [_request("a")])
    events = [
        r
        for r in core.scheduler._late_admission_records
        if r["event"] == "predictor_submission_progress"
    ]
    assert len(events) == 2
    assert all(r[3] == 0 for r in events[0]["primary_snapshot"])
    assert events[1]["observed_ns"] >= at
    assert all(r[3] == at for r in events[1]["primary_snapshot"])
    assert core.scheduler._late_transport["seal_ns"] == 1_120_000_000
    assert max(clock.sleeps) <= 2_000_000


def test_compute_deadline_preserves_load_only_readiness_policy(deadline_core):
    core, clock = deadline_core()
    core.batch_queue[-1][1].total_num_scheduled_tokens = 0
    clock.ready_at = {"a": 1_150_000_000}
    stage_rpc(core, 9, [_request("base")], [_request("a")])
    assert core.scheduler._late_transport["seal_reason"] == "candidate_target_ready"
    assert core.scheduler._late_transport["compute_seal_deadline_ns"] is None


@pytest.mark.parametrize("coalesced", [False, True])
@pytest.mark.parametrize("guard_us", [0, 50_000])
@pytest.mark.parametrize("poll_us", [2000, 5000, 10000])
@pytest.mark.parametrize("event", ["compute", "kv", "cancel", "fine", "deadline"])
def test_full_prediction_wait_is_interrupted_before_its_timer(
    tmp_path, monkeypatch, guard_us, poll_us, event, coalesced, coalesced_channel
):
    """A long forecast cannot hide physical completion, KV readiness, or aborts."""
    import numpy as np

    from vllm.v1.kv_offload.seal_wakeup import TransportWakeup
    from vllm.v1.kv_offload.step_timing import RankFile

    executor = ControlledExecutor()
    core = _core(executor)
    scheduler = core.scheduler
    scheduler._late_seal_policy = "predictive"
    scheduler._late_seal_prediction_to_readiness = True
    scheduler._late_seal_compute_deadline = event == "deadline"
    scheduler._late_seal_compute_deadline_lead_us = 5000
    scheduler._late_seal_full_prediction_wait = True
    scheduler._late_seal_coalesced_notifications = coalesced
    scheduler._late_seal_wait_trace = True
    scheduler._late_seal_condition_preparation = True
    scheduler._late_seal_hybrid_guard_us = guard_us
    scheduler._late_seal_adaptive_poll_us = poll_us
    scheduler._late_seal_target_candidates = 16
    scheduler._late_seal_repeat_us = 25_000
    scheduler._late_seal_prediction_scale_percent = 100
    scheduler._late_seal_prediction_reserve_us = 10_000
    scheduler._late_seal_safety_us = 1_000_000
    scheduler._late_base_request_shapes = []
    scheduler.requests = {}
    core.batch_queue[-1][1].total_num_scheduled_tokens = 1024
    core.batch_queue[-1][1].readiness_timeline_step = 8
    started = time.monotonic_ns()
    predicted = 120_000_000 if event in {"fine", "deadline"} else 500_000_000
    ranks = [RankFile(str(tmp_path / f"rank-{i}"), create=True) for i in range(8)]
    paths = [str(tmp_path / f"ring-{i}") for i in range(8)]
    for path in paths:
        with open(path, "wb") as stream:
            stream.write(bytes(64))
    for rank in ranks:
        rank.write(
            [8, started, started, started + 100, time.monotonic_ns(), 1, 0, 0, 0]
        )
    channel, lib, handles = coalesced_channel
    core.early_load_readiness = SimpleNamespace(
        paths=paths,
        seal_notifications=channel if coalesced else None,
        _released_jobs=set(),
        _rank_reports={},
    )
    if coalesced:
        from vllm.v1.kv_offload.seal_notifications import NotificationQueue

        core.aborts_queue = NotificationQueue(channel)
        scheduler.connector = SimpleNamespace(
            connector_scheduler=SimpleNamespace(
                _req_status={"a": SimpleNamespace(transfer_jobs={101})}
            )
        )
        for index, rank in enumerate(ranks):
            channel.descriptor(str(tmp_path / f"rank-{index}") + ".execution", index)
            # Separate worker writers use the same completion-publication hook.
            from vllm.v1.kv_offload.seal_notifications import SealNotifications

            rank.seal = SealNotifications(channel.path)
            rank.seal_rank = index
    core.deferral_timing = SimpleNamespace(
        execution_ranks=ranks,
        steps={8: {"step": 8, "request_shapes": [[1024, 8192]]}},
        read_execution=lambda step: [rank.read(step) for rank in ranks],
        seal_estimator=SimpleNamespace(
            predict=lambda data: predicted,
            features=lambda data: np.ones(4),
            samples=[0] * 8,
            preparation_reserve=lambda extra, upcoming=None: extra,
            preparation_ns=[],
        ),
    )
    kv = threading.Event()
    core._refresh_early_load_readiness.side_effect = lambda **kw: (
        scheduler.finished_recving_kv_req_ids.add("a") if kv.is_set() else None
    )
    waiting = threading.Event()
    wait_class = type(channel) if coalesced else TransportWakeup
    original_wait = wait_class.wait

    def wait(self, ticket, seconds):
        waiting.set()
        return original_wait(self, ticket, seconds)

    monkeypatch.setattr(wait_class, "wait", wait)
    if event == "fine":
        from vllm.v1.kv_offload import late_admission_runtime as late

        def fine_sleep(seconds):
            if seconds == poll_us / 1e6:
                kv.set()
            time.sleep(seconds)

        monkeypatch.setattr(
            late,
            "time",
            SimpleNamespace(
                monotonic=time.monotonic,
                monotonic_ns=time.monotonic_ns,
                thread_time_ns=time.thread_time_ns,
                sleep=fine_sleep,
            ),
        )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(
                stage_rpc, core, 9, [_request("base")], [_request("a")]
            )
            assert executor.published.wait(5)
            for release in executor.release_base + executor.release_candidate:
                release.set()
            for prepared in executor.prepared_base + executor.prepared_candidate:
                assert prepared.wait(5)
            assert waiting.wait(5)
            observed = time.monotonic_ns()
            if event == "compute":
                for rank in ranks:
                    now = time.monotonic_ns()
                    rank.write(
                        [
                            8,
                            started,
                            started,
                            started + 100,
                            now,
                            1,
                            now - started,
                            now - 100,
                            now,
                        ]
                    )
            elif event == "kv":
                kv.set()
                if coalesced:
                    for handle in handles:
                        assert lib.seal_kv(handle, 101, 1)
                for path in paths:
                    with open(path, "r+b") as ring_stream:
                        ring_stream.seek(24)
                        ring_stream.write(struct.pack("Q", 1))
            elif event == "cancel":
                core.aborts_queue.put(["a"])
            result.result(timeout=5)
        transport = scheduler._late_transport
        assert any(
            w["phase"] == "coarse" and w["requested_ns"] > 50_000_000
            for w in transport["adaptive_waits"]
        )
        if event == "deadline":
            assert transport["seal_reason"] == "compute_seal_deadline"
            assert transport["seal_ns"] >= started + predicted - 5_000_000
            expiry = transport["effective_seal_expiry_ns"]
            assert all(
                w["requested_ns"] <= max(0, expiry - w["decision_ns"])
                for w in transport["adaptive_waits"]
            )
        elif event == "fine":
            assert transport["seal_ns"] >= transport["hybrid_transition_ns"]
            assert any(
                w["phase"] == "fine" and w["requested_ns"] == poll_us * 1000
                for w in transport["adaptive_waits"]
            )
        else:
            assert (
                transport["seal_ns"]
                < started + predicted - 10_000_000 - guard_us * 1000
            )
            assert transport["seal_ns"] - observed < 100_000_000
        if coalesced:
            assert transport["coalesced_notifications"]
            assert transport["progress_bridge_scans"] == 0
            assert transport["progress_bridge_cpu_ns"] == 0
        else:
            assert transport["progress_bridge_cpu_ns"] > 0
        assert not core.batch_queue[-1][0].poll()
        if event == "compute":
            assert transport["seal_reason"] == "predecessor_compute_finished"
        elif event in ("kv", "fine"):
            assert transport["seal_reason"] == "candidate_target_ready"
            assert scheduler._late_kv_snapshot == {"a"}
        elif event == "cancel":
            assert scheduler._late_cancelled == {"a"}
    finally:
        executor.finish()
        for rank in ranks:
            rank.close()


def test_progress_between_snapshot_and_wait_is_not_lost():
    from vllm.v1.kv_offload.seal_wakeup import SealWakeup

    value = [0]
    wakeup = SealWakeup(lambda: value[0])
    ticket = wakeup.ticket()
    value[0] = 1
    try:
        assert wakeup.wait(ticket, 1) == "progress"
        # A notification observed before entering wait also remains visible.
        assert wakeup.wait(ticket, 1) == "progress"
    finally:
        wakeup.close()
    assert not wakeup.thread.is_alive()


@pytest.mark.parametrize(
    "conditioned,recent", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("hybrid_us", [0, 20_000, 50_000])
def test_prediction_reserves_worker_preparation_while_a_is_held(
    conditioned, recent, hybrid_us
):
    import numpy as np

    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_seal_policy = "predictive"
    core.scheduler._late_seal_condition_preparation = conditioned
    core.scheduler._late_seal_recent_preparation = recent
    core.scheduler._late_seal_hybrid_wait_us = hybrid_us
    core.scheduler._late_seal_hybrid_guard_us = 5000
    core.scheduler._late_base_request_shapes = [[1, 8192]]
    core.scheduler.requests = {}
    queries = []
    core.scheduler._late_seal_prediction_scale_percent = 100
    core.scheduler._late_seal_prediction_reserve_us = 10_000
    core.scheduler._late_seal_safety_us = 1_000_000
    core.batch_queue[-1][1].total_num_scheduled_tokens = 1024
    core.batch_queue[-1][1].readiness_timeline_step = 8
    start = time.monotonic_ns()

    def reserve(extra, upcoming=None):
        queries.append(upcoming)
        return extra + (5_000_000 if upcoming is not None else 20_000_000)

    def recent_reserve(extra, upcoming):
        queries.append(upcoming)
        return extra + 5_000_000

    estimator = SimpleNamespace(
        predict=lambda data: 50_000_000,
        features=lambda data: np.ones(4),
        samples=[0] * 8,
        preparation_reserve=reserve,
        preparation_ns=[20_000_000] * 8,
        recent_preparation_reserve=recent_reserve,
    )
    core.deferral_timing = SimpleNamespace(
        steps={8: {"step": 8, "request_shapes": [[1024, 8192]]}},
        read_execution=lambda step: (
            [(step, start, start, start + 100, time.monotonic_ns(), 1, 0, 0, 0)] * 8
        ),
        seal_estimator=estimator,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 9, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        result.result(timeout=5)
    transport = core.scheduler._late_transport
    assert transport["seal_reason"] == "predicted_compute_deadline"
    assert transport["prediction"]["predicted_duration_ns"] == 50_000_000
    expected_reserve = 15_000_000 if conditioned else 30_000_000
    assert transport["prediction"]["preparation_reserve_ns"] == expected_reserve
    assert transport["seal_ns"] >= start + 50_000_000 - expected_reserve
    if hybrid_us:
        assert transport["hybrid_coarse_sleeps"] > 0
    if conditioned:
        assert queries[-1] == {"request_shapes": [[1, 8192]]}
        assert transport["prediction"]["preparation_conditioned"]
    assert not core.batch_queue[-1][0].poll()
    executor.finish()


def test_hybrid_rechecks_predecessor_completion_after_a_bounded_wait(monkeypatch):
    """Completion during the coarse phase must stop before the forecast deadline."""
    import numpy as np

    from vllm.v1.kv_offload import late_admission_runtime as late

    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_seal_policy = "predictive"
    core.scheduler._late_seal_prediction_scale_percent = 100
    core.scheduler._late_seal_prediction_reserve_us = 10_000
    core.scheduler._late_seal_safety_us = 1_000_000
    core.scheduler._late_seal_hybrid_wait_us = 20_000
    core.scheduler._late_seal_hybrid_guard_us = 50_000
    core.batch_queue[-1][1].total_num_scheduled_tokens = 1024
    core.batch_queue[-1][1].readiness_timeline_step = 8
    started = time.monotonic_ns()
    core.deferral_timing = SimpleNamespace(
        steps={8: {"step": 8, "request_shapes": [[1024, 8192]]}},
        read_execution=lambda step: (
            [(step, started, started, started + 100, time.monotonic_ns(), 1, 0, 0, 0)]
            * 8
        ),
        seal_estimator=SimpleNamespace(
            predict=lambda data: 500_000_000,
            features=lambda data: np.ones(4),
            samples=[0] * 8,
            preparation_reserve=lambda extra: extra,
            preparation_ns=[],
        ),
    )
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if seconds >= 0.01:
            core.batch_queue[-1][0].release.set()
        time.sleep(seconds)

    monkeypatch.setattr(
        late,
        "time",
        SimpleNamespace(
            monotonic=time.monotonic,
            monotonic_ns=time.monotonic_ns,
            thread_time_ns=time.thread_time_ns,
            sleep=sleep,
        ),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 9, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        result.result(timeout=5)
    transport = core.scheduler._late_transport
    assert transport["seal_reason"] == "predecessor_compute_finished"
    assert transport["hybrid_coarse_sleeps"] == 1
    assert max(sleeps) <= 0.02
    assert transport["seal_ns"] < transport["prediction"]["deadline_ns"]
    executor.finish()


@pytest.mark.parametrize("poll_us", [1000, 2000, 5000, 10000])
def test_event_seal_repeats_until_candidate_ready_with_a_held(poll_us):
    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_seal_policy = "event"
    core.scheduler._late_seal_adaptive_poll_us = poll_us
    core.scheduler._late_seal_target_candidates = 1
    core.scheduler._late_seal_repeat_us = 1000
    core.scheduler._late_seal_safety_us = 1_000_000
    core.scheduler._late_record_enabled = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 9, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        for prepared in executor.prepared_base + executor.prepared_candidate:
            assert prepared.wait(5)
        for _ in range(1000):
            if any(
                row["event"] == "seal_repeat"
                for row in getattr(core.scheduler, "_late_admission_records", [])
            ):
                break
            threading.Event().wait(0.001)
        else:
            pytest.fail("event hold did not repeat")
        assert not result.done()
        core.scheduler.finished_recving_kv_req_ids.add("a")
        assert result.result(timeout=5) == ["a"]
    assert core.scheduler._late_transport["seal_reason"] == "candidate_target_ready"
    assert core.scheduler._late_transport["adaptive_poll_us"] == poll_us
    assert not core.batch_queue[-1][0].poll()
    executor.finish()


@pytest.mark.parametrize("compute_tokens", [0, 3])
def test_load_aware_target_is_lower_only_without_preceding_compute(compute_tokens):
    """Four transfers can start B after a load-only A; live compute waits for all."""
    executor = ControlledExecutor()
    core = _core(executor)
    core.batch_queue[0][1].total_num_scheduled_tokens = compute_tokens
    core.scheduler._late_seal_policy = "event"
    core.scheduler._late_seal_target_candidates = 16
    core.scheduler._late_seal_load_only_target_candidates = 4
    core.scheduler._late_seal_repeat_us = 1000
    core.scheduler._late_seal_safety_us = 1_000_000
    core.scheduler._late_record_enabled = True
    candidates = [_request(f"candidate-{i}") for i in range(8)]
    first = {c.req_id for c in candidates[:4]}
    core.scheduler.finished_recving_kv_req_ids.update(first)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 9, [_request("base")], candidates)
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        for prepared in executor.prepared_base + executor.prepared_candidate:
            assert prepared.wait(5)
        if compute_tokens:
            for _ in range(1000):
                if any(
                    row["event"] == "seal_repeat"
                    and row["ready"] == 4
                    and row["target"] == 8
                    and row["has_compute"]
                    for row in getattr(core.scheduler, "_late_admission_records", [])
                ):
                    break
                threading.Event().wait(0.001)
            else:
                pytest.fail("compute path did not hold after observing four ready")
            assert not result.done()
            core.scheduler.finished_recving_kv_req_ids.update(
                c.req_id for c in candidates
            )
            expected = {c.req_id for c in candidates}
        else:
            expected = first
        assert expected <= set(result.result(timeout=5))
        assert core.scheduler._late_kv_snapshot == expected
    assert not core.batch_queue[-1][0].poll()
    executor.finish()


def test_eight_rank_base_seal_does_not_wait_for_optional_ack_or_a():
    executor = ControlledExecutor()
    core = _core(executor)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 9, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for rank in range(7):
            executor.release_base[rank].set()
            executor.release_candidate[rank].set()
            assert executor.prepared_base[rank].wait(5)
            assert executor.prepared_candidate[rank].wait(5)
        assert not result.done()
        executor.release_base[7].set()
        assert result.result(timeout=5) == []
        assert not core.batch_queue[-1][0].poll()
    assert core.scheduler._late_transport["statuses"][:8] == [1] * 8
    executor.finish()


def test_candidate_ack_before_base_seal_is_selected():
    executor = ControlledExecutor(candidate_before_base_last=True)
    core = _core(executor)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 10, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for rank in range(7):
            executor.release_base[rank].set()
            executor.release_candidate[rank].set()
            assert executor.prepared_candidate[rank].wait(5)
        executor.release_candidate[7].set()
        assert executor.prepared_candidate[7].wait(5)
        executor.release_base[7].set()
        assert result.result(timeout=5) == ["a"]
    executor.finish()


def test_fixed_hold_admits_ack_after_all_base_ready():
    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_seal_delay_us = 100_000
    core.scheduler._late_record_enabled = True
    core.scheduler.finished_recving_kv_req_ids.add("a")
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 11, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for release in executor.release_base:
            release.set()
        for prepared in executor.prepared_base:
            assert prepared.wait(5)
        for _ in range(1000):
            if any(
                row["event"] == "all_base_ready"
                for row in getattr(core.scheduler, "_late_admission_records", [])
            ):
                break
            threading.Event().wait(0.001)
        assert not result.done()
        for release in executor.release_candidate:
            release.set()
        for prepared in executor.prepared_candidate:
            assert prepared.wait(5)
        assert result.result(timeout=5) == ["a"]
    transport = core.scheduler._late_transport
    assert transport["seal_hold_actual_ns"] >= 100_000_000
    assert not core.batch_queue[-1][0].poll()
    executor.finish()


def test_all_candidate_ready_seals_before_deadline_after_kv_arrives():
    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_seal_delay_us = 200_000
    core.scheduler._late_seal_when_all_candidates_ready = True
    core.scheduler._late_record_enabled = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 12, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        for prepared in executor.prepared_base + executor.prepared_candidate:
            assert prepared.wait(5)
        assert not result.done()
        core.scheduler.finished_recving_kv_req_ids.add("a")
        assert result.result(timeout=5) == ["a"]
    transport = core.scheduler._late_transport
    assert transport["seal_reason"] == "all_candidates_ready"
    assert transport["seal_hold_actual_ns"] < 200_000_000
    assert not core.batch_queue[-1][0].poll()
    executor.finish()


def test_all_candidate_ready_waits_for_ack_and_keeps_deadline_fallback():
    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_seal_delay_us = 200_000
    core.scheduler._late_seal_when_all_candidates_ready = True
    core.scheduler.finished_recving_kv_req_ids.add("a")
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 13, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for release in executor.release_base:
            release.set()
        for prepared in executor.prepared_base:
            assert prepared.wait(5)
        assert not result.done()
        for release in executor.release_candidate:
            release.set()
        for prepared in executor.prepared_candidate:
            assert prepared.wait(5)
        assert result.result(timeout=5) == ["a"]
    assert core.scheduler._late_transport["seal_reason"] == "all_candidates_ready"
    executor.finish()

    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_seal_delay_us = 50_000
    core.scheduler._late_seal_when_all_candidates_ready = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 14, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        assert result.result(timeout=5) == ["a"]
    assert core.scheduler._late_transport["seal_reason"] == "deadline"
    assert core.scheduler._late_kv_snapshot == set()
    executor.finish()


@pytest.mark.parametrize(
    "exit_kind,expected_reason",
    [
        ("kv_ready", "all_candidates_ready"),
        ("a_finished", "predecessor_finished_during_extension"),
        ("cap", "extension_deadline"),
    ],
)
def test_productive_predecessor_bounded_extension_exits_on_event_or_cap(
    exit_kind, expected_reason
):
    executor = ControlledExecutor()
    core = _core(executor)
    core.batch_queue[-1][1].total_num_scheduled_tokens = 1
    core.scheduler._late_seal_delay_us = 30_000
    core.scheduler._late_seal_extension_us = 100_000
    core.scheduler._late_seal_when_all_candidates_ready = True
    core.scheduler._late_record_enabled = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 50, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        for prepared in executor.prepared_base + executor.prepared_candidate:
            assert prepared.wait(5)
        if exit_kind != "cap":
            for _ in range(1000):
                if any(
                    row["event"] == "seal_extension_started"
                    for row in getattr(core.scheduler, "_late_admission_records", [])
                ):
                    break
                threading.Event().wait(0.001)
            else:
                pytest.fail("extension did not start")
            assert not result.done()
            if exit_kind == "kv_ready":
                core.scheduler.finished_recving_kv_req_ids.add("a")
            else:
                core.batch_queue[-1][0].release.set()
        assert result.result(timeout=5) == ["a"]
    transport = core.scheduler._late_transport
    assert transport["seal_reason"] == expected_reason
    assert transport["seal_hold_actual_ns"] >= 30_000_000
    assert transport["seal_extension_started_ns"] is not None
    if exit_kind == "kv_ready":
        assert transport["kv_ready"] == ["a"]
    elif exit_kind == "a_finished":
        assert transport["predecessor_ready"]
    else:
        assert transport["seal_hold_actual_ns"] >= 130_000_000
    executor.finish()


def test_extension_skips_empty_predecessor_at_fixed_deadline():
    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_seal_delay_us = 30_000
    core.scheduler._late_seal_extension_us = 100_000
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 51, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        assert result.result(timeout=5) == ["a"]
    transport = core.scheduler._late_transport
    assert transport["seal_reason"] == "deadline"
    assert transport["seal_extension_started_ns"] is None
    executor.finish()


@pytest.mark.parametrize(
    "candidate_count,ready_count,productive_predecessor,expected_reason",
    [
        (8, 4, False, "partial_ready_no_compute"),
        (9, 5, False, "partial_ready_no_compute"),
        (7, 4, False, "deadline"),
        (8, 4, True, "deadline"),
    ],
)
def test_partial_ready_seal_requires_productive_cohort_and_idle_compute(
    candidate_count,
    ready_count,
    productive_predecessor,
    expected_reason,
):
    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_seal_delay_us = 200_000
    core.scheduler._late_partial_min_pending_candidates = 8
    core.scheduler._late_partial_ready_percent = 50
    core.scheduler._late_record_enabled = True
    core.batch_queue[-1][1].total_num_scheduled_tokens = int(productive_predecessor)
    candidates = [_request(f"c{i}") for i in range(candidate_count)]
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 15, [_request("base")], candidates)
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        for prepared in executor.prepared_base + executor.prepared_candidate:
            assert prepared.wait(5)
        core.scheduler.finished_recving_kv_req_ids.update(
            f"c{i}" for i in range(ready_count)
        )
        assert result.result(timeout=5) == [f"c{i}" for i in range(candidate_count)]
    transport = core.scheduler._late_transport
    assert transport["seal_reason"] == expected_reason
    assert transport["partial_ready_count"] == ready_count
    assert transport["partial_target"] == (candidate_count + 1) // 2
    assert core.scheduler._late_kv_snapshot == {f"c{i}" for i in range(ready_count)}
    if expected_reason == "partial_ready_no_compute":
        assert transport["seal_hold_actual_ns"] < 200_000_000
    assert not core.batch_queue[-1][0].poll()
    executor.finish()


def test_partial_ready_seal_does_not_count_cancelled_candidate():
    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_seal_delay_us = 300_000
    core.scheduler._late_partial_min_pending_candidates = 8
    core.scheduler._late_partial_ready_percent = 50
    core.scheduler.finished_recving_kv_req_ids.update(f"c{i}" for i in range(4))
    core.aborts_queue.put("c0")
    candidates = [_request(f"c{i}") for i in range(8)]
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 16, [_request("base")], candidates)
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        for prepared in executor.prepared_base + executor.prepared_candidate:
            assert prepared.wait(5)
        threading.Event().wait(0.02)
        assert not result.done()
        core.scheduler.finished_recving_kv_req_ids.add("c4")
        assert result.result(timeout=5) == [f"c{i}" for i in range(8)]
    transport = core.scheduler._late_transport
    assert transport["seal_reason"] == "partial_ready_no_compute"
    assert transport["partial_ready_count"] == 4
    assert transport["cancelled"] == ["c0"]
    executor.finish()


@pytest.mark.parametrize("deadline", [False, True])
def test_execute_b_is_enqueued_while_a_main_rpc_is_held(monkeypatch, deadline):
    import numpy as np

    from vllm.v1.kv_offload import queue_stages

    monkeypatch.setattr(queue_stages, "active", False)
    executor = ControlledExecutor()
    predecessor = HeldPredecessor()
    core = EngineCore.__new__(EngineCore)
    core.batch_queue_size = 2
    a_output = SchedulerOutput.make_empty()
    a_output.readiness_timeline_step = 8
    core.batch_queue = deque([(predecessor, a_output, predecessor)])
    core.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=8))
    core.model_executor = executor
    core.aborts_queue = queue.Queue()
    core.early_load_readiness = SimpleNamespace(
        experiment_options={"cooperative_response_progress": True}
    )
    core._refresh_early_load_readiness = MagicMock()
    core._process_aborts_queue = MagicMock()
    core._prepare_batch_queue_scheduling = MagicMock(return_value=None)
    core._defer_batch_queue_scheduling = MagicMock(return_value=False)
    core._zero_token_queue_credit_available = MagicMock(return_value=False)
    core._should_throttle_prefills = MagicMock(return_value=False)
    core._timing_enabled = MagicMock(return_value=False)
    core._record_deferral_dispatch = MagicMock()
    core.log_error_detail = lambda _: nullcontext()
    core.is_ec_consumer = False
    core.is_pooling_model = True
    core.scheduler = MagicMock()
    core.scheduler._late_generation = 1
    core.scheduler._late_partial_min_pending_candidates = 0
    core.scheduler._late_partial_ready_percent = 0
    core.scheduler.finished_recving_kv_req_ids = set()
    core.scheduler.has_requests.return_value = True
    if deadline:
        scheduler = core.scheduler
        scheduler._late_seal_policy = "predictive"
        scheduler._late_seal_prediction_to_readiness = True
        scheduler._late_seal_compute_deadline = True
        scheduler._late_seal_compute_deadline_lead_us = 5000
        scheduler._late_seal_target_candidates = 16
        scheduler._late_seal_prediction_scale_percent = 100
        scheduler._late_seal_prediction_reserve_us = 10_000
        scheduler._late_seal_safety_us = 1_000_000
        scheduler._late_seal_adaptive_poll_us = 2000
        scheduler._late_seal_hybrid_wait_us = 0
        scheduler._late_seal_full_prediction_wait = False
        scheduler._late_seal_coalesced_notifications = False
        scheduler._late_seal_condition_preparation = False
        scheduler._late_cached_requests = False
        scheduler._late_seal_when_all_candidates_ready = False
        a_output.total_num_scheduled_tokens = 1024
        started = time.monotonic_ns()
        core.deferral_timing = SimpleNamespace(
            steps={8: {"step": 8, "request_shapes": [[1024, 8192]]}},
            read_execution=lambda step: (
                [[step, started, started, started, time.monotonic_ns(), 1, 0, 0, 0]] * 8
            ),
            seal_estimator=SimpleNamespace(
                predict=lambda data: 50_000_000,
                features=lambda data: np.ones(4),
                samples=[0] * 8,
                preparation_reserve=lambda extra: extra,
                preparation_ns=[],
            ),
        )

    def schedule(*_):
        candidates = [_request("a")] if deadline else []
        assert stage_rpc(core, 9, [_request("base")], candidates) == []
        output = SchedulerOutput.make_empty()
        output.readiness_timeline_step = 9
        return output

    core.scheduler.schedule.side_effect = schedule
    executor.execute_model = MagicMock(return_value=Future())
    with ThreadPoolExecutor(max_workers=1) as pool:
        progress = pool.submit(core.step_with_batch_queue)
        assert executor.published.wait(5)
        for release in executor.release_base:
            release.set()
        assert progress.result(timeout=5) == (None, False)
    executor.execute_model.assert_called_once()
    if deadline:
        assert core.scheduler._late_transport["seal_reason"] == "compute_seal_deadline"
    assert not predecessor.release.is_set()
    executor.finish()


@pytest.mark.parametrize(
    "order,expected",
    [
        ("ack_kv_seal", True),
        ("kv_ack_seal", True),
        ("seal_ack_kv", False),
        ("seal_kv_ack", False),
        ("ack_seal_kv", False),
        ("kv_seal_ack", False),
    ],
)
def test_trace_order_membership_is_immutable_on_all_ranks(order, expected):
    """An all-rank ack and KV readiness must both precede the seal."""
    candidate = Candidate("late", 3, 16)
    window = AdmissionWindow(2, 9, (candidate,), 8, 16)
    window.staged(2, 9, 0)
    kv_ready = False
    applied = []
    for event in order.split("_"):
        if event == "ack":
            for rank in range(8):
                window.ready(2, 9, rank, 0, 3)
        elif event == "kv":
            kv_ready = True
        else:
            proposal = window.propose_seal(2, 9)
            # The engine's scheduler can admit only KV-ready requests.
            active = proposal.active if kv_ready else ()
            if not kv_ready:
                window.cancel(2, 9, 0)
                proposal = window.propose_seal(2, 9)
            window.commit(proposal, lambda selected: applied.extend(selected))
            assert proposal.active == active
    assert (candidate in applied) is expected
    for rank in range(8):
        window.staged(2, 9, rank)
        assert (candidate in window.consume(2, 9, rank).active) is expected
        window.retire(2, 9, rank)
    assert window.reclaimable
    with pytest.raises(RuntimeError, match="duplicate retirement"):
        window.retire(2, 9, 0)


def test_partial_stale_cancel_and_budget_never_gain_membership():
    candidates = tuple(Candidate(str(i), i, 8) for i in range(4))
    window = AdmissionWindow(4, 12, candidates, 8, 16)
    window.staged(4, 12, 0)
    for rank in range(8):
        window.ready(4, 12, rank, 0, 0)
        window.ready(4, 12, rank, 1, 1)
    for rank in range(7):
        window.ready(4, 12, rank, 2, 2)
    with pytest.raises(ValueError, match="stale"):
        window.ready(3, 12, 7, 2, 2)
    with pytest.raises(ValueError, match="stale candidate"):
        window.ready(4, 12, 7, 2, 1)
    assert window.cancel(4, 12, 1)
    window.commit(window.propose_seal(4, 12), lambda _: None)
    assert window.sealed.active == (candidates[0],)
    window.ready(4, 12, 7, 2, 2)
    assert window.sealed.active == (candidates[0],)
    assert not window.cancel(4, 12, 0)


def test_predecessor_result_first_uses_ordinary_path():
    executor = ControlledExecutor()
    predecessor = HeldPredecessor()
    core = _core(executor, predecessor)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 13, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        predecessor.release.set()
        assert result.result(timeout=5) == []
    assert core.scheduler._late_transport["predecessor_ready"]
    executor.finish()


def test_seal_captures_kv_and_cancellation_once():
    executor = ControlledExecutor()
    core = _core(executor)
    core.aborts_queue.put(["a"])
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 14, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        for prepared in executor.prepared_candidate:
            assert prepared.wait(5)
        assert set(result.result(timeout=5)) <= {"a"}
    assert core.scheduler._late_cancelled == {"a"}
    assert core.scheduler._late_kv_snapshot == set()
    core.scheduler.finished_recving_kv_req_ids.add("a")
    assert core.scheduler._late_kv_snapshot == set()
    assert core.aborts_queue.get_nowait() == ["a"]
    executor.finish()


def test_publisher_failure_unlinks_control_and_data(monkeypatch):
    from pathlib import Path

    from vllm.v1.kv_offload import late_admission_runtime as late

    paths = []
    original_mkstemp = late.tempfile.mkstemp

    def capture_path(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        paths.append(path)
        return fd, path

    monkeypatch.setattr(late.tempfile, "mkstemp", capture_path)
    executor = MagicMock()
    executor.publish_late_preparation.side_effect = RuntimeError("publisher failed")
    core = _core(executor)
    with pytest.raises(RuntimeError, match="publisher failed"):
        stage_rpc(core, 9, [_request("base")], [_request("a")])
    assert paths and all(not Path(path).exists() for path in paths)


def test_worker_private_preparation_is_consumed_once():
    import struct
    import tempfile

    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    lane = PreparationLane(0, 3)
    with (
        tempfile.NamedTemporaryFile() as control,
        tempfile.NamedTemporaryFile() as data,
    ):
        control.write(struct.pack("<QQ", 7, 1) + bytes(2))
        control.flush()
        pickle.dump(
            (3, 7, [_request("base")], [_request("candidate")], control.name), data
        )
        data.flush()
        publisher = SimpleNamespace(world_size=1, _late_preparation_paths=[lane.path])
        MultiprocExecutor.publish_late_preparation(publisher, data.name)
        deadline = threading.Event()
        for _ in range(1000):
            if control.readable() and lane.prepared.get(7, {}).get("candidate"):
                break
            deadline.wait(0.001)
        selected = lane.consume(7, {"candidate"})
        assert list(selected) == ["candidate"]
        assert selected["candidate"].token_ids.tolist() == [1, 2, 3]
        assert lane.consume(7, {"candidate"}) == {}
    lane.close()


def test_notice_lane_acknowledges_identity_without_copying_tokens():
    import os
    import struct
    import tempfile

    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    lane = PreparationLane(0, 3, notice_only=True)
    with (
        tempfile.NamedTemporaryFile() as control,
        tempfile.NamedTemporaryFile() as data,
    ):
        control.write(struct.pack("<QQ", 7, 1) + bytes(2))
        control.flush()
        pickle.dump((3, 7, [], [CandidateNotice("candidate")], control.name), data)
        data.flush()
        publisher = SimpleNamespace(world_size=1, _late_preparation_paths=[lane.path])
        MultiprocExecutor.publish_late_preparation(publisher, data.name)
        for _ in range(1000):
            if lane.available(7) and os.pread(control.fileno(), 2, 16) == b"\1\1":
                break
            threading.Event().wait(0.001)
        assert lane.available(7) == ["candidate"]
        assert os.pread(control.fileno(), 2, 16) == b"\1\1"
        selected = lane.consume(7, {"candidate"})
        assert selected["candidate"].token_ids.size == 0
        assert lane.consume(7, {"candidate"}) == {}
    lane.close()


def test_worker_lane_reads_two_coalesced_publications():
    import os
    import struct
    import tempfile

    lane = PreparationLane(0, 3)
    with tempfile.TemporaryDirectory() as root:
        paths = []
        for step in (7, 8):
            control = f"{root}/control-{step}"
            data = f"{root}/data-{step}"
            with open(control, "wb") as target:
                target.write(struct.pack("<QQ", step, 1) + bytes(1))
            with open(data, "wb") as target:
                pickle.dump((3, step, [_request(f"base-{step}")], [], control), target)
            paths.append(data)
        fd = os.open(lane.path, os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(fd, ("\n".join(paths) + "\n").encode())
        finally:
            os.close(fd)
        for _ in range(1000):
            if all(lane.available(step) for step in (7, 8)):
                break
            threading.Event().wait(0.001)
        assert lane.available(7) == ["base-7"]
        assert lane.available(8) == ["base-8"]
        assert list(lane.consume(7, {"base-7"})) == ["base-7"]
        assert list(lane.consume(8, {"base-8"})) == ["base-8"]
    lane.close()


def test_worker_lane_finishes_slots_after_seal_unlinks_control(tmp_path, monkeypatch):
    import os
    import struct

    lane = PreparationLane(0, 3)
    control = tmp_path / "control"
    data = tmp_path / "data"
    control.write_bytes(struct.pack("<QQ", 7, 1) + bytes(3))
    with data.open("wb") as target:
        pickle.dump(
            (
                3,
                7,
                [_request("base")],
                [_request("candidate-1"), _request("candidate-2")],
                str(control),
            ),
            target,
        )
    candidate_started = threading.Event()
    release_candidate = threading.Event()
    original_prepare = lane._prepare_request

    def prepare(request):
        if request.req_id == "candidate-1":
            candidate_started.set()
            assert release_candidate.wait(5)
        return original_prepare(request)

    monkeypatch.setattr(lane, "_prepare_request", prepare)
    fd = os.open(lane.path, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(fd, (str(data) + "\n").encode())
        assert candidate_started.wait(5)
        assert control.read_bytes()[16] == 1
        control.unlink()
        release_candidate.set()
        for _ in range(1000):
            if "candidate-2" in lane.available(7):
                break
            threading.Event().wait(0.001)
        assert lane.available(7) == ["base", "candidate-1", "candidate-2"]
    finally:
        release_candidate.set()
        os.close(fd)
        lane.close()


def test_full_queue_polls_readiness_before_oldest_output(monkeypatch):
    """At capacity, cooperative progress polls without another schedule call."""
    from vllm.v1.kv_offload import queue_stages

    monkeypatch.setattr(queue_stages, "active", False)
    oldest: Any = Future()
    oldest.set_result("oldest-output")
    oldest.poll = MagicMock(return_value=False)
    newer: Any = Future()
    newer.set_result("newer-output")
    output = lambda step: SimpleNamespace(
        readiness_timeline_step=step,
        total_num_scheduled_tokens=1,
        pending_structured_output_tokens=False,
        kv_connector_metadata=None,
    )
    old_batch, new_batch = output(1), output(2)
    core = EngineCore.__new__(EngineCore)
    core.batch_queue = deque([(newer, new_batch, newer), (oldest, old_batch, oldest)])
    core.batch_queue_size = 2
    core.early_load_readiness = SimpleNamespace(
        experiment_options={"cooperative_response_progress": True}
    )
    core.scheduler = MagicMock()
    core.scheduler.has_requests.return_value = True
    core.scheduler.update_from_output.return_value = {}
    core._refresh_early_load_readiness = MagicMock(return_value=["late"])
    core._process_aborts_queue = MagicMock()
    core._attach_iteration_details = MagicMock()
    core.capture_iteration_details = core.log_error_detail = lambda _: nullcontext()
    core._timing_enabled = MagicMock(return_value=False)
    core._zero_token_queue_credit_available = MagicMock(return_value=False)
    core.is_ec_consumer = False
    core.check_for_draft_tokens = False
    core.step_with_batch_queue()
    core.scheduler.schedule.assert_not_called()
    assert len(core.batch_queue) == 2
    core._refresh_early_load_readiness.assert_called_once()
    core.scheduler.update_from_output.assert_not_called()
    oldest.poll.return_value = True
    core.step_with_batch_queue()
    core.scheduler.update_from_output.assert_called_once_with(
        old_batch, "oldest-output"
    )
    assert [entry[1].readiness_timeline_step for entry in core.batch_queue] == [2]


@pytest.mark.parametrize(
    "mode,ready,expected_tokens",
    [
        (1, True, 0),
        (2, False, 0),
        (2, True, 12),
        (3, False, 0),
        (3, True, 12),
        (4, False, 0),
        (4, True, 12),
    ],
)
def test_real_scheduler_seals_one_decision_with_exact_token_accounting(
    request_runner, mode, ready, expected_tokens
):
    runner = request_runner(
        block_size=4, num_gpu_blocks=100, async_scheduling=False, blocks_per_chunk=3
    )
    runner.new_request(token_ids=[1] * 24)
    scheduler = runner.scheduler
    assert scheduler.schedule().num_scheduled_tokens == {"0": 24}
    request = scheduler.requests["0"]
    original_blocks = scheduler.kv_cache_manager.get_blocks("0").get_block_ids()
    scheduler.running.remove(request)
    scheduler.waiting.prepend_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.num_computed_tokens = 12
    runner.connector_scheduler._current_batch_load_jobs = set()
    assert runner.connector_scheduler._req_status["0"].transfer_jobs
    scheduler._late_admission_mode = mode
    prepared = []

    def prepare(step, base, candidates):
        assert not base
        prepared.append((step, [candidate.req_id for candidate in candidates]))
        if ready:
            scheduler.finished_recving_kv_req_ids.add("0")
            scheduler._late_kv_snapshot = {"0"}
        return [candidate.req_id for candidate in candidates]

    scheduler._late_admission_prepare = prepare
    schedule = MagicMock(wraps=scheduler.schedule)
    result = schedule()
    schedule.assert_called_once_with()
    assert prepared == [(2, ["0"])]
    assert result.num_scheduled_tokens == (
        {"0": expected_tokens} if expected_tokens else {}
    )
    assert result.total_num_scheduled_tokens == expected_tokens
    assert request.num_computed_tokens == 12 + expected_tokens
    assert scheduler.requests["0"] is request
    assert scheduler.kv_cache_manager.get_blocks("0").get_block_ids() == original_blocks
    assert (request in scheduler.running) is bool(expected_tokens)


@pytest.mark.parametrize("mode,expected_base", [(2, ["1"]), (3, []), (4, [])])
def test_candidate_only_mode_uses_ordinary_base_request_path(
    request_runner, mode, expected_base
):
    runner = request_runner(
        block_size=4, num_gpu_blocks=100, async_scheduling=False, blocks_per_chunk=3
    )
    runner.new_request(token_ids=[1] * 24)
    scheduler = runner.scheduler
    assert scheduler.schedule().num_scheduled_tokens == {"0": 24}
    remote = scheduler.requests["0"]
    scheduler.running.remove(remote)
    scheduler.waiting.prepend_request(remote)
    remote.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    remote.num_computed_tokens = 12
    runner.connector_scheduler._current_batch_load_jobs = set()
    runner.new_request(token_ids=[2] * 24)
    scheduler._late_admission_mode = mode
    prepared_base: list[str] = []

    def prepare(step, base, candidates):
        prepared_base.extend(request.req_id for request in base)
        assert [candidate.req_id for candidate in candidates] == ["0"]
        return []

    scheduler._late_admission_prepare = prepare
    output = scheduler.schedule()
    assert prepared_base == expected_base
    assert output.num_scheduled_tokens == {"1": 24}


@pytest.mark.parametrize(
    "outcome", ["ready", "complete", "physical", "physical_cancel", "cancel"]
)
def test_prediction_to_readiness_keeps_waiting_after_forecast(outcome):
    """A forecast controls sleep size; it cannot finalize membership."""
    import numpy as np

    executor = ControlledExecutor()
    core = _core(executor)
    scheduler = core.scheduler
    scheduler._late_seal_policy = "predictive"
    scheduler._late_seal_prediction_to_readiness = True
    scheduler._late_seal_target_candidates = 16
    scheduler._late_seal_repeat_us = 1000
    scheduler._late_seal_adaptive_poll_us = 2000
    scheduler._late_seal_prediction_scale_percent = 100
    scheduler._late_seal_prediction_reserve_us = 10_000
    scheduler._late_seal_safety_us = 1_000_000
    scheduler._late_seal_hybrid_wait_us = 20_000
    scheduler._late_seal_hybrid_guard_us = 50_000
    scheduler._late_record_enabled = True
    core.batch_queue[-1][1].total_num_scheduled_tokens = 1024
    core.batch_queue[-1][1].readiness_timeline_step = 8
    started = time.monotonic_ns()
    physical_finished = threading.Event()
    finish_and_cancel = threading.Event()

    def refresh(poll_markers=False):
        if finish_and_cancel.is_set():
            scheduler.finished_recving_kv_req_ids.add("a")
            core.aborts_queue.put(["a"])
            physical_finished.set()

    core._refresh_early_load_readiness.side_effect = refresh

    def execution(step):
        now = time.monotonic_ns()
        duration = now - started if physical_finished.is_set() else 0
        ended = now if physical_finished.is_set() else 0
        return [
            (step, started, started, started + 100, now, 1, duration, ended, ended)
        ] * 8

    core.deferral_timing = SimpleNamespace(
        steps={8: {"step": 8, "request_shapes": [[1024, 8192]]}},
        read_execution=execution,
        seal_estimator=SimpleNamespace(
            predict=lambda data: 100_000_000,
            features=lambda data: np.ones(4),
            samples=[0] * 8,
            preparation_reserve=lambda extra: extra,
            preparation_ns=[],
        ),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 9, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        for prepared in executor.prepared_base + executor.prepared_candidate:
            assert prepared.wait(5)
        deadline = started + 120_000_000
        while time.monotonic_ns() < deadline:
            threading.Event().wait(0.002)
        assert not result.done()
        if outcome == "ready":
            scheduler.finished_recving_kv_req_ids.add("a")
        elif outcome == "complete":
            core.batch_queue[-1][0].release.set()
        elif outcome == "physical":
            physical_finished.set()
        elif outcome == "physical_cancel":
            finish_and_cancel.set()
        else:
            core.aborts_queue.put(["a"])
        # stage_rpc returns prepared slots; final admission uses the KV/cancel snapshot.
        assert result.result(timeout=5) == ["a"]
        assert scheduler._late_kv_snapshot == (
            {"a"} if outcome in {"ready", "physical_cancel"} else set()
        )
        assert scheduler._late_cancelled == (
            {"a"} if outcome in {"cancel", "physical_cancel"} else set()
        )
    transport = scheduler._late_transport
    assert transport["seal_reason"] == (
        "predecessor_compute_finished"
        if outcome in {"complete", "physical", "physical_cancel"}
        else "candidate_target_ready"
    )
    assert transport["seal_ns"] > transport["prediction"]["deadline_ns"]
    assert transport["hybrid_coarse_sleeps"] > 0
    assert transport["hybrid_polling_ns"] is not None
    if outcome in {"physical", "physical_cancel"}:
        assert not core.batch_queue[-1][0].poll()
    executor.finish()


def test_prediction_to_readiness_uses_partial_target_without_compute():
    executor = ControlledExecutor()
    core = _core(executor)
    scheduler = core.scheduler
    scheduler._late_seal_policy = "predictive"
    scheduler._late_seal_prediction_to_readiness = True
    scheduler._late_seal_target_candidates = 16
    scheduler._late_seal_load_only_target_candidates = 4
    scheduler._late_seal_repeat_us = 1000
    scheduler._late_seal_safety_us = 1_000_000
    candidates = [_request(f"candidate-{i}") for i in range(8)]
    first = {c.req_id for c in candidates[:4]}
    scheduler.finished_recving_kv_req_ids.update(first)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 9, [_request("base")], candidates)
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        assert first <= set(result.result(timeout=5))
        assert scheduler._late_kv_snapshot == first
    assert scheduler._late_transport["seal_reason"] == "candidate_target_ready"
    assert not core.batch_queue[-1][0].poll()
    executor.finish()


@pytest.mark.parametrize(
    "condition",
    ["idle", "base_ack", "partial_ack", "cancelled", "compute", "unsupported"],
)
def test_productive_load_only_exit_keeps_compute_ack_and_cancellation_gates(
    deadline_core, condition
):
    """Ready productive work can start early only after every required ACK."""
    import numpy as np

    from vllm.v1.kv_offload.productive_load_policy import DIRECT_FEATURES
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    core, clock = deadline_core()
    core.batch_queue[-1][1].total_num_scheduled_tokens = (
        1024 if condition == "compute" else 0
    )
    core.scheduler._late_seal_load_only_ready_percent = 50
    core.scheduler._late_base_request_shapes = []
    context = 8192 if condition == "unsupported" else 58368
    ids = [
        "partial" if i == 0 and condition == "partial_ack" else f"c{i}"
        for i in range(16)
    ]
    core.scheduler.requests = {
        rid: SimpleNamespace(num_tokens=context + 3072, num_computed_tokens=context)
        for rid in ids
    }
    clock.ready_at = {
        rid: 1_011_000_000 if i < 3 else 1_051_000_000 for i, rid in enumerate(ids)
    }
    if condition == "base_ack":
        clock.base_at = 1_031_000_000
    if condition == "cancelled":
        core.aborts_queue.put([ids[0]])
    calibration = BatchShapeCalibration(
        dict(
            schema=5,
            features=["intercept"],
            coefficients=[0],
            support=dict(requests=[1, 32], context=[15552, 58368], suffix=[832, 3072]),
            geometry_support=dict(suffix_per_context=[0.01, 0.25]),
            bias_bounds_ms=[-500, 500],
            provenance={},
            productive_load_policy=dict(
                schema=1,
                features=DIRECT_FEATURES,
                coefficients=[80, 0, 1000, 0, 0, 10],
                work_coefficients=[1000, 10],
                minimum_efficiency=0.7,
                minimum_ready_candidates=1,
                provenance={},
            ),
        )
    )
    x = np.zeros(28)
    x[:9] = [832, 1, 0, 15552, 832, 832, 15552, 15552, 896]
    core.deferral_timing.seal_estimator.latency_surface = SimpleNamespace(
        requires_forecast=False,
        predict_with_features=lambda data: (120_000_000, x),
        duration_calibration=calibration,
        target_schema="cuda_span_max.v1",
    )
    acknowledged = stage_rpc(
        core, 9, [_request("base")], [_request(rid) for rid in ids]
    )
    transport = core.scheduler._late_transport
    assert transport["statuses"][:8] == [1] * 8
    if condition in ("idle", "base_ack"):
        assert transport["seal_reason"] == "load_only_productive"
        assert transport["partial_ready_count"] == 3
        assert transport["seal_ns"] >= (clock.base_at or 1_011_000_000)
        assert transport["seal_ns"] < 1_051_000_000
        assert transport["load_only_productive_efficiency"] >= 0.7
        assert set(acknowledged) & core.scheduler._late_kv_snapshot == set(ids[:3])
    else:
        assert transport["seal_reason"] == "candidate_target_ready"
        assert transport["seal_ns"] >= 1_051_000_000
    assert not core.batch_queue[-1][0].poll()


@pytest.mark.parametrize("policy", ["event", "predictive"])
@pytest.mark.parametrize("count", [1, 2, 9, 14, 16])
def test_load_only_fraction_waits_for_half_the_published_cohort(
    deadline_core, policy, count
):
    """Ready loads do not shrink the denominator; odd counts round upward."""
    core, clock = deadline_core()
    core.batch_queue[-1][1].total_num_scheduled_tokens = 0
    scheduler = core.scheduler
    scheduler._late_seal_policy = policy
    scheduler._late_seal_target_candidates = 4
    scheduler._late_seal_load_only_target_candidates = 4
    scheduler._late_seal_load_only_ready_percent = 50
    target = (count + 1) // 2
    candidates = [_request(f"candidate-{i}") for i in range(count)]
    ready = {c.req_id for c in candidates[:target]}
    clock.ready_at = {c.req_id: clock.now for c in candidates[: target - 1]}
    clock.ready_at[candidates[target - 1].req_id] = 1_011_000_000
    stage_rpc(core, 9, [_request("base")], candidates)
    transport = scheduler._late_transport
    assert transport["seal_ns"] == 1_011_000_000
    assert transport["ready_target"] == target
    assert transport["load_only_pending_candidates"] == count
    assert transport["load_only_fraction_target"] == target
    assert transport["partial_ready_count"] == target
    assert transport["seal_reason"] == "candidate_target_ready"
    assert scheduler._late_kv_snapshot == ready
    assert not core.batch_queue[-1][0].poll()


def test_load_only_fraction_requires_eighth_rank_candidate_ack(deadline_core):
    """Eight ready loads are insufficient while the eighth slot lacks one ack."""
    core, clock = deadline_core()
    core.batch_queue[-1][1].total_num_scheduled_tokens = 0
    scheduler = core.scheduler
    scheduler._late_seal_load_only_ready_percent = 50
    candidates = [_request(f"candidate-{i}") for i in range(16)]
    candidates[7] = _request("partial")
    clock.ready_at = {c.req_id: clock.now for c in candidates[:8]}
    publish = core.model_executor.publish_late_preparation
    refresh = core._refresh_early_load_readiness.side_effect
    paths = []

    def capture(data_path):
        with open(data_path, "rb") as source:
            paths.append(pickle.load(source)[4])
        publish(data_path)

    def acknowledge(**kwargs):
        refresh(**kwargs)
        if clock.now >= 1_021_000_000:
            signal_prepared(paths[0], 9, 7, 1, 7)

    core.model_executor.publish_late_preparation = capture
    core._refresh_early_load_readiness.side_effect = acknowledge
    stage_rpc(core, 9, [_request("base")], candidates)
    assert scheduler._late_transport["seal_ns"] == 1_021_000_000
    assert scheduler._late_transport["partial_ready_count"] == 8


def test_load_only_fraction_requires_all_base_acks(deadline_core):
    """A ready half-cohort cannot bypass base preparation on the eighth rank."""
    core, clock = deadline_core()
    core.batch_queue[-1][1].total_num_scheduled_tokens = 0
    core.scheduler._late_seal_load_only_ready_percent = 50
    candidates = [_request(f"candidate-{i}") for i in range(16)]
    clock.ready_at = {c.req_id: clock.now for c in candidates[:8]}
    clock.base_at = 1_021_000_000
    stage_rpc(core, 9, [_request("base")], candidates)
    assert core.scheduler._late_transport["seal_ns"] == clock.base_at
    assert core.scheduler._late_transport["partial_ready_count"] == 8


def test_load_only_fraction_excludes_cancellation_without_shrinking_target(
    deadline_core,
):
    """Cancelling one of eight ready loads still requires an eighth live load."""
    core, clock = deadline_core()
    core.batch_queue[-1][1].total_num_scheduled_tokens = 0
    scheduler = core.scheduler
    scheduler._late_seal_load_only_ready_percent = 50
    candidates = [_request(f"candidate-{i}") for i in range(16)]
    clock.ready_at = {c.req_id: clock.now for c in candidates[:8]}
    clock.ready_at["candidate-8"] = 1_021_000_000
    core.aborts_queue.put(["candidate-7"])
    stage_rpc(core, 9, [_request("base")], candidates)
    transport = scheduler._late_transport
    assert transport["seal_ns"] == 1_021_000_000
    assert transport["ready_target"] == 8
    assert transport["partial_ready_count"] == 8
    assert scheduler._late_cancelled == {"candidate-7"}


def test_load_only_fraction_caps_target_at_remaining_live_candidates(deadline_core):
    """A frozen half-cohort target cannot strand fewer live loads after aborts."""
    core, clock = deadline_core()
    core.batch_queue[-1][1].total_num_scheduled_tokens = 0
    scheduler = core.scheduler
    scheduler._late_seal_load_only_ready_percent = 50
    candidates = [_request(f"candidate-{i}") for i in range(16)]
    clock.ready_at = {c.req_id: clock.now for c in candidates[:6]}
    core.aborts_queue.put([c.req_id for c in candidates[6:]])
    stage_rpc(core, 9, [_request("base")], candidates)
    transport = scheduler._late_transport
    assert transport["load_only_fraction_target"] == 8
    assert transport["ready_target"] == 6
    assert transport["partial_ready_count"] == 6
    assert transport["seal_reason"] == "candidate_target_ready"


def test_load_only_fraction_preserves_active_compute_target(deadline_core):
    """Half ready cannot seal B early while a productive A is still running."""
    core, clock = deadline_core()
    scheduler = core.scheduler
    scheduler._late_seal_load_only_ready_percent = 50
    candidates = [_request(f"candidate-{i}") for i in range(16)]
    clock.ready_at = {c.req_id: clock.now for c in candidates[:8]}
    clock.ready_at.update({c.req_id: 1_021_000_000 for c in candidates[8:]})
    stage_rpc(core, 9, [_request("base")], candidates)
    transport = scheduler._late_transport
    assert transport["seal_ns"] == 1_021_000_000
    assert transport["ready_target"] == 16
    assert transport["partial_ready_count"] == 16
    assert transport["prediction"] is not None
    assert not core.batch_queue[-1][0].poll()


def test_load_only_fraction_retains_safety_timeout(deadline_core):
    """Seven ready loads cannot wait indefinitely for the eighth."""
    core, clock = deadline_core()
    core.batch_queue[-1][1].total_num_scheduled_tokens = 0
    scheduler = core.scheduler
    scheduler._late_seal_load_only_ready_percent = 50
    candidates = [_request(f"candidate-{i}") for i in range(16)]
    clock.ready_at = {c.req_id: clock.now for c in candidates[:7]}
    stage_rpc(core, 9, [_request("base")], candidates)
    transport = scheduler._late_transport
    assert transport["seal_reason"] == "adaptive_safety_deadline"
    assert transport["ready_target"] == 8
    assert transport["partial_ready_count"] == 7
    assert transport["seal_ns"] == 2_001_000_000


def test_prediction_to_readiness_handles_physical_completion_before_base_ack():
    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_seal_policy = "predictive"
    core.scheduler._late_seal_prediction_to_readiness = True
    core.scheduler._late_seal_target_candidates = 16
    core.batch_queue[-1][1].total_num_scheduled_tokens = 1024
    core.batch_queue[-1][1].readiness_timeline_step = 8
    started = time.monotonic_ns()
    core.deferral_timing = SimpleNamespace(
        read_execution=lambda step: (
            [
                (
                    step,
                    started,
                    started,
                    started + 100,
                    time.monotonic_ns(),
                    1,
                    1000,
                    started + 1000,
                    started + 2000,
                )
            ]
            * 8
        ),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stage_rpc, core, 9, [_request("base")], [_request("a")])
        assert executor.published.wait(5)
        for release in executor.release_base + executor.release_candidate:
            release.set()
        result.result(timeout=5)
    assert (
        core.scheduler._late_transport["seal_reason"] == "predecessor_compute_finished"
    )
    assert core.scheduler._late_transport["prediction"] is None
    assert not core.batch_queue[-1][0].poll()
    executor.finish()


@pytest.fixture(scope="module")
def native_seal_library(tmp_path_factory):
    """Compile the same CUDA-independent publisher used by the H2D worker."""
    import ctypes
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    output = tmp_path_factory.mktemp("seal-native") / "publisher.so"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            "-O2",
            "-I",
            str(root / "csrc/libtorch_stable"),
            str(root / "experiments/coalesced_notify_20261004/native_harness.cpp"),
            "-o",
            str(output),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(output))
    lib.seal_open.argtypes = [ctypes.c_char_p]
    lib.seal_open.restype = ctypes.c_void_p
    lib.seal_close.argtypes = [ctypes.c_void_p]
    lib.seal_kv.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
    lib.seal_compute.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
    return lib


@pytest.fixture
def coalesced_channel(tmp_path, native_seal_library):
    from vllm.v1.kv_offload.seal_notifications import SealNotifications

    channel = SealNotifications(tmp_path / "seal", 123, 8)
    paths = [str(tmp_path / f"rank-{rank}") for rank in range(8)]
    handles = []
    for rank, path in enumerate(paths):
        channel.descriptor(path, rank)
        handle = native_seal_library.seal_open(path.encode())
        assert handle
        handles.append(handle)
    try:
        yield channel, native_seal_library, handles
    finally:
        for handle in handles:
            native_seal_library.seal_close(handle)
        channel.close()


@pytest.mark.parametrize("prep_first", [False, True])
def test_coalescing_requires_same_candidate_all_rank_kv_and_preparation(
    coalesced_channel, prep_first
):
    """Different per-rank job orders and partial ACKs produce no engine wake."""
    import select

    channel, lib, handles = coalesced_channel
    channel.arm(9, 1, [[101, 102], [201]], [8])
    ticket = channel.ticket()
    if prep_first:
        for slot in range(2):
            for rank in range(8):
                channel.prepared(9, rank, 1, slot)
    for rank, handle in enumerate(handles):
        assert lib.seal_kv(handle, 101 if rank % 2 else 201, 1)
    assert not select.select([channel.pipe], [], [], 0)[0]
    # Completing one of a candidate's multiple jobs still cannot notify.
    for handle in handles:
        assert lib.seal_kv(handle, 101, 1)
    assert channel.ticket() == ticket
    for handle in handles:
        assert lib.seal_kv(handle, 102, 1)
    if not prep_first:
        for rank in range(7):
            channel.prepared(9, rank, 1, 0)
        assert not select.select([channel.pipe], [], [], 0)[0]
        channel.prepared(9, 7, 1, 0)
    assert channel.wait(ticket, 1) == "progress"
    assert channel.ticket() == ticket + 1
    # Duplicate reports cannot produce a second notification for this candidate.
    for rank, handle in enumerate(handles):
        assert lib.seal_kv(handle, 102, 1)
        channel.prepared(9, rank, 1, 0)
    assert channel.ticket() == ticket + 1
    channel.disarm()


@pytest.mark.parametrize("event", ["kv", "ack", "compute", "cancel", "failure"])
def test_producer_notification_interrupts_one_blocking_wait(
    coalesced_channel, monkeypatch, event
):
    """No observer thread or periodic timeout is needed to interrupt a long wait."""
    from vllm.v1.kv_offload import seal_notifications as notices

    channel, lib, handles = coalesced_channel
    channel.arm(9, 1, [[101]], [8])
    for rank, handle in enumerate(handles):
        if rank != 7 or event == "ack":
            assert lib.seal_kv(handle, 101, 1)
        if event != "ack" or rank != 7:
            channel.prepared(9, rank, 1, 0)
        if rank != 7:
            assert lib.seal_compute(handle, 8, 1)
    ticket = channel.ticket()
    blocked = threading.Event()
    calls = []
    original = notices.select.select

    def select_once(read, write, error, timeout):
        calls.append(timeout)
        blocked.set()
        return original(read, write, error, timeout)

    monkeypatch.setattr(notices.select, "select", select_once)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(channel.wait, ticket, 2)
        assert blocked.wait(5)
        if event == "kv":
            assert lib.seal_kv(handles[7], 101, 1)
        elif event == "ack":
            channel.prepared(9, 7, 1, 0)
        elif event == "compute":
            assert lib.seal_compute(handles[7], 8, 1)
        elif event == "cancel":
            aborts = notices.NotificationQueue(channel)
            aborts.put_nowait(["candidate"])
            assert aborts.get_nowait() == ["candidate"]
        else:
            assert lib.seal_kv(handles[7], 101, 0)
        if event == "failure":
            with pytest.raises(RuntimeError, match="producer failed"):
                result.result(timeout=1)
        else:
            assert result.result(timeout=1) == "progress"
    assert len(calls) == 1
    assert calls[0] > 1.5
    assert channel.scans == 0
    assert not hasattr(channel, "thread")
    channel.disarm()


def test_notification_before_wait_and_generation_reset_are_not_lost(coalesced_channel):
    channel, lib, handles = coalesced_channel
    channel.arm(9, 1, [[101]], [8])
    ticket = channel.ticket()
    for rank, handle in enumerate(handles):
        channel.prepared(9, rank, 1, 0, generation=1)
        assert lib.seal_kv(handle, 101, 1)
    assert channel.wait(ticket, 1) == "progress"
    channel.disarm()
    channel.arm(9, 2, [[201]], [18])
    ticket = channel.ticket()
    for rank, handle in enumerate(handles):
        channel.prepared(9, rank, 1, 0, generation=1)
        assert lib.seal_kv(handle, 101, 1)
        assert lib.seal_compute(handle, 8, 1)
    assert channel.ticket() == ticket
    for rank, handle in enumerate(handles):
        channel.prepared(9, rank, 1, 0, generation=2)
        assert lib.seal_kv(handle, 201, 1)
    assert channel.wait(ticket, 1) == "progress"
    channel.disarm()


def test_arm_seeds_completions_that_preceded_candidate_publication(coalesced_channel):
    channel, _, _ = coalesced_channel
    channel.arm(9, 1, [[101, 102]], [8])
    ticket = channel.ticket()
    for rank in range(8):
        channel.prepared(9, rank, 1, 0)
    channel.seed(
        SimpleNamespace(_released_jobs={101}, _rank_reports={102: set(range(8))})
    )
    assert channel.wait(ticket, 1) == "progress"
    channel.disarm()


def test_full_fifo_retains_notification_sequence(coalesced_channel):
    channel, _, _ = coalesced_channel
    channel.arm(9, 1, [[101]], [])
    ticket = channel.ticket()
    import os

    with pytest.raises(BlockingIOError):
        while True:
            os.write(channel.pipe, bytes(4096))
    channel.notify()
    assert channel.wait(ticket, 1) == "progress"
    assert channel.ticket() == ticket + 1
    channel.disarm()


def _seal_process_publish(library, transport, channel_path, rank, barrier):
    import ctypes

    from vllm.v1.kv_offload.seal_notifications import SealNotifications

    lib = ctypes.CDLL(library)
    lib.seal_open.argtypes = [ctypes.c_char_p]
    lib.seal_open.restype = ctypes.c_void_p
    lib.seal_close.argtypes = [ctypes.c_void_p]
    lib.seal_kv.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
    handle = lib.seal_open(transport.encode())
    assert handle
    channel = SealNotifications(channel_path)
    try:
        barrier.wait(timeout=30)
        channel.prepared(9, rank, 1, 0)
        for job in [101, 102] if rank % 2 else [102, 101]:
            assert lib.seal_kv(handle, job, 1)
    finally:
        channel.close()
        lib.seal_close(handle)


def test_eight_processes_publish_without_losing_rank_updates(
    tmp_path, coalesced_channel
):
    """Separate worker processes share masks; concurrent updates notify once."""
    import multiprocessing

    channel, lib, _ = coalesced_channel
    channel.arm(9, 1, [[101, 102]], [])
    ticket = channel.ticket()
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(8)
    workers = []
    try:
        for rank in range(8):
            path = str(tmp_path / f"process-{rank}")
            channel.descriptor(path, rank)
            worker = context.Process(
                target=_seal_process_publish,
                args=(lib._name, path, channel.path, rank, barrier),
            )
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join(timeout=40)
            assert worker.exitcode == 0
        assert channel.wait(ticket, 1) == "progress"
        assert channel.ticket() == ticket + 1
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        channel.disarm()


def test_terminal_compute_stays_visible_after_a_busy_authoritative_read(
    coalesced_channel,
):
    """An advisory end must not be lost if the engine's timing read was busy."""
    channel, lib, handles = coalesced_channel
    channel.arm(9, 1, [[101]], [8])
    for handle in handles:
        assert lib.seal_compute(handle, 8, 1)
    ticket = channel.ticket()
    assert channel.wait(ticket, 1) == "progress"
    # Same ticket remains immediately actionable until the engine disarms.
    assert channel.wait(ticket, 1) == "progress"
    channel.disarm()


def test_notification_channel_cleanup_preserves_abort_queue(tmp_path):
    from pathlib import Path

    from vllm.v1.kv_offload.seal_notifications import (
        NotificationQueue,
        SealNotifications,
    )

    channel = SealNotifications(str(tmp_path / "seal"), 123, 8)
    descriptor = channel.descriptor(str(tmp_path / "worker"), 0)
    aborts = NotificationQueue(channel)
    channel.close()
    channel.close()
    assert not Path(descriptor).exists()
    assert not Path(channel.path).exists()
    assert not Path(channel.path + ".fifo").exists()
    aborts.put_nowait(["request"])
    assert aborts.get_nowait() == ["request"]


def test_disarm_wakes_waiter_and_next_window_has_no_old_membership(
    coalesced_channel, monkeypatch
):
    from vllm.v1.kv_offload import seal_notifications as notices

    channel, _, _ = coalesced_channel
    channel.arm(9, 1, [[101]], [8])
    ticket = channel.ticket()
    blocked = threading.Event()
    original = notices.select.select

    def wait(read, write, error, timeout):
        blocked.set()
        return original(read, write, error, timeout)

    monkeypatch.setattr(notices.select, "select", wait)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(channel.wait, ticket, 2)
        assert blocked.wait(5)
        channel.disarm()
        assert result.result(timeout=1) == "progress"
    channel.arm(19, 2, [[201]], [18])
    assert channel.wait(channel.ticket(), 0) == "timer"
    channel.disarm()


@pytest.mark.parametrize("status", [2, 3])
def test_coalesced_preparation_preserves_nonready_and_failure_status(
    coalesced_channel, status
):
    channel, _, _ = coalesced_channel
    channel.arm(9, 1, [[101]], [])
    ticket = channel.ticket()
    channel.prepared(9, 0, status, 0)
    if status == 3:
        with pytest.raises(RuntimeError, match="producer failed"):
            channel.wait(ticket, 0)
    else:
        assert channel.wait(ticket, 0) == "timer"
    channel.disarm()


def _cache_request(req_id, tokens=None):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    tokens = [17, 23, 41] if tokens is None else tokens
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=tokens,
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=2),
        pooling_params=None,
        block_ids=([1],),
        num_computed_tokens=1,
        lora_request=None,
        prefill_token_ids=list(tokens),
    )


@pytest.mark.parametrize("compact", [False, True])
def test_cached_reference_hydrates_tokens_and_preserves_final_allocation(
    tmp_path, compact
):
    """The sealed output remains authoritative for blocks and computed tokens."""
    from dataclasses import replace

    import numpy as np

    from vllm.v1.kv_offload.late_admission_runtime import (
        cache_references,
        compact_preparation_request,
        resolve_references,
    )

    lane = PreparationLane(0, 3, cached_requests=True)
    source = _cache_request("a", list(range(61440)))
    control, payload = tmp_path / "control", tmp_path / "payload"
    control.write_bytes(struct.pack("<QQ", 7, 1) + bytes(2))
    with payload.open("wb") as stream:
        wire = compact_preparation_request(source) if compact else source
        pickle.dump((3, 7, [], [wire], str(control)), stream)
    try:
        lane._prepare_path(str(payload))
        assert control.read_bytes()[16:] == b"\1\1"
        prepared = lane.consume(7, {"a"})
        final = replace(source, block_ids=([7, 8],), num_computed_tokens=61400)
        scheduler = SimpleNamespace(_late_reference_requests={"a": ((3, 7), source)})
        referenced = cache_references(scheduler, [final])
        assert referenced[0].prompt_token_ids is None
        assert referenced[0].prefill_token_ids is None
        assert len(pickle.dumps(referenced)) < len(pickle.dumps([final])) / 10
        output = SimpleNamespace(
            scheduled_new_reqs=referenced, readiness_timeline_step=7
        )
        resolve_references(output, prepared, 3)
        restored = output.scheduled_new_reqs[0]
        np.testing.assert_array_equal(
            restored.prefill_token_ids, source.prefill_token_ids
        )
        if compact:
            assert not restored.prefill_token_ids.flags.writeable
            assert restored.prefill_token_ids is prepared["a"].token_ids
            assert prepared["a"].source.prefill_token_ids is None
        assert restored.prompt_len == 61440
        assert restored.block_ids == ([7, 8],)
        assert restored.num_computed_tokens == 61400
        assert lane.consume(7, {"a"}) == {}
    finally:
        lane.close()


@pytest.mark.parametrize("failure", ["missing", "generation", "step", "revision"])
def test_cached_reference_rejects_stale_or_missing_preparation(failure):
    from dataclasses import replace

    from vllm.v1.kv_offload.late_admission_runtime import resolve_references

    lane = PreparationLane(0, 3, cached_requests=True)
    try:
        prepared = {"a": lane._prepare_data(_cache_request("a"), 7)}
        output = SimpleNamespace(
            scheduled_new_reqs=[
                replace(
                    _cache_request("a"),
                    prompt_token_ids=None,
                    prefill_token_ids=None,
                    late_cached_key=(3, 7),
                )
            ],
            readiness_timeline_step=7,
        )
        generation = 3
        if failure == "missing":
            prepared = {}
        if failure == "generation":
            generation = 4
        if failure == "step":
            output.readiness_timeline_step = 8
        if failure == "revision":
            prepared["a"] = replace(prepared["a"], cache_key=(3, 6))
        with pytest.raises(RuntimeError, match="missing or stale"):
            resolve_references(output, prepared, generation)
    finally:
        lane.close()


@pytest.mark.parametrize("compact", [False, True])
def test_cached_reference_falls_back_for_unacked_or_changed_prompts(compact):
    from dataclasses import replace

    from vllm.v1.kv_offload.late_admission_runtime import (
        cache_references,
        compact_preparation_request,
        resolve_references,
    )

    source = _cache_request("a")
    changed = replace(source, prefill_token_ids=[17, 23, 99])
    unacked = _cache_request("b")
    scheduler = SimpleNamespace(_late_reference_requests={"a": ((3, 7), source)})
    result = cache_references(scheduler, [changed, unacked])
    assert result[0] is changed and result[1] is unacked
    assert all(r.late_cached_key is None for r in result)
    lane = PreparationLane(0, 3, cached_requests=True)
    try:
        wire = compact_preparation_request(source) if compact else source
        prepared = {"a": lane._prepare_data(wire, 7)}
        output = SimpleNamespace(scheduled_new_reqs=result, readiness_timeline_step=7)
        resolve_references(output, prepared, 3)
        assert output.scheduled_new_reqs[0] is changed
        assert prepared == {}
    finally:
        lane.close()


@pytest.mark.parametrize("compact", [False, True])
def test_partial_rank_ack_keeps_candidate_payload_and_caches_base(compact):
    from vllm.v1.kv_offload.late_admission_runtime import cache_references

    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_cached_requests = True
    core.scheduler._late_compact_tokens = compact
    base, candidate = _cache_request("base"), _cache_request("a")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(stage_rpc, core, 9, [base], [candidate])
            assert executor.published.wait(5)
            for rank in range(7):
                executor.release_base[rank].set()
                executor.release_candidate[rank].set()
                assert executor.prepared_candidate[rank].wait(5)
            executor.release_base[7].set()
            assert result.result(timeout=5) == []
        refs = cache_references(core.scheduler, [base, candidate])
        assert refs[0].late_cached_key == (1, 9)
        assert refs[1] is candidate
    finally:
        executor.finish()


def test_compact_preparation_snapshots_tokens_without_worker_lists():
    """Later engine mutations cannot change the acknowledged worker snapshot."""
    import numpy as np

    from vllm.v1.kv_offload.late_admission_runtime import compact_preparation_request

    source = _cache_request("a")
    wire = compact_preparation_request(source)
    source.prompt_token_ids[0] = 99
    source.prefill_token_ids[1] = 100
    decoded = pickle.loads(pickle.dumps(wire, protocol=pickle.HIGHEST_PROTOCOL))
    assert decoded.source.prompt_token_ids is None
    assert decoded.source.prefill_token_ids is None
    lane = PreparationLane(0, 3, cached_requests=True)
    try:
        prepared = lane._prepare_data(decoded, 7)
        np.testing.assert_array_equal(prepared.token_ids, [17, 23, 41])
        assert not prepared.token_ids.flags.writeable
    finally:
        lane.close()


@pytest.mark.parametrize("failure", ["length", "metadata", "mutable_bytes"])
def test_compact_preparation_rejects_invalid_descriptor(failure):
    from dataclasses import replace

    from vllm.v1.kv_offload.late_admission_runtime import compact_preparation_request

    wire = compact_preparation_request(_cache_request("a"))
    if failure == "length":
        wire = replace(wire, token_bytes=wire.token_bytes[:-1])
    elif failure == "metadata":
        wire = replace(wire, source=_cache_request("a"))
    else:
        wire = replace(wire, token_bytes=bytearray(wire.token_bytes))
    lane = PreparationLane(0, 3, cached_requests=True)
    try:
        with pytest.raises(RuntimeError, match="invalid compact"):
            lane._prepare_data(wire, 7)
    finally:
        lane.close()


def test_compact_tokens_require_private_preparation():
    from vllm.v1.kv_offload.early_readiness import validate_experiment_options

    with pytest.raises(ValueError, match="compact tokens require"):
        validate_experiment_options(dict(late_compact_tokens=True))


@pytest.mark.parametrize(
    "options",
    [
        {"late_seal_compute_deadline": True},
        {"late_seal_compute_deadline_shadow": True},
        {"late_seal_compute_deadline_lead_us": 5000},
    ],
)
def test_compute_deadline_rejects_options_without_readiness_policy(options):
    from vllm.v1.kv_offload.early_readiness import validate_experiment_options

    with pytest.raises(ValueError, match="compute seal"):
        validate_experiment_options(options)


@pytest.mark.parametrize("cached,pinned", [(True, False), (False, True), (True, True)])
def test_private_preparation_options_require_mode_four(cached, pinned):
    from vllm.v1.kv_offload.early_readiness import validate_experiment_options

    with pytest.raises(ValueError, match="requires admission mode 4"):
        validate_experiment_options(
            dict(late_cached_requests=cached, late_pinned_packing=pinned)
        )


@pytest.mark.parametrize("length", [8192, 61440])
@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_pinned_preparation_adopts_selected_tokens_through_actual_runner(
    length, changed, compact
):
    """No live writes before execute; selected slots match ordinary runner state."""
    import os
    from dataclasses import replace

    if os.environ.get("CACHED_PINNED_GPU_TEST") != "1":
        pytest.skip("explicit idle-GPU mechanism test")
    import numpy as np
    import torch

    from vllm.v1.kv_offload.late_admission_runtime import (
        cache_references,
        compact_preparation_request,
        resolve_references,
    )
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.states import RequestState

    def runner():
        obj = object.__new__(GPUModelRunner)
        obj.req_states = RequestState(
            8, length + 8, 4096, 1, 1024, torch.device("cuda:0")
        )
        obj.req_states.all_token_ids.gpu.fill_(-1)
        obj._remove_request = obj.req_states.remove_request
        obj.adaptive_verification = obj.pooling_runner = obj.encoder_cache = None
        obj.model_state = MagicMock()
        obj.block_tables = MagicMock()
        obj.lora_state = MagicMock()
        obj.is_last_pp_rank = False
        obj.sampler = None
        return obj

    optimized, ordinary = runner(), runner()
    requests = [
        _cache_request(f"r{i}", (np.arange(length) + i).tolist()) for i in range(5)
    ]
    lane = PreparationLane(0, 3, cached_requests=True, pinned_packing=True)
    try:
        prepared = {
            r.req_id: lane._prepare_data(
                compact_preparation_request(r) if compact else r, 7
            )
            for r in requests
        }
        assert all(p.pinned_tokens.is_pinned() for p in prepared.values())
        # Preparation creates CPU tensors only, leaving the live UVA store unchanged.
        assert torch.all(optimized.req_states.all_token_ids.gpu == -1).item()
        selected = [requests[4], requests[1]]
        if changed:
            tokens = [99] * length
            selected[0] = replace(
                selected[0], prompt_token_ids=tokens, prefill_token_ids=tokens
            )
        scheduler = SimpleNamespace(
            _late_reference_requests={r.req_id: ((3, 7), r) for r in requests}
        )
        output = SimpleNamespace(
            scheduled_new_reqs=cache_references(scheduler, selected),
            readiness_timeline_step=7,
        )
        resolve_references(output, prepared, 3)
        optimized._late_prepared = {
            r.req_id: prepared[r.req_id] for r in selected if r.req_id in prepared
        }
        optimized.add_requests(output)
        ordinary.add_requests(SimpleNamespace(scheduled_new_reqs=selected))
        optimized.req_states.all_token_ids.drain_pinned_writes()
        torch.accelerator.synchronize()
        assert torch.equal(
            optimized.req_states.all_token_ids.gpu,
            ordinary.req_states.all_token_ids.gpu,
        )
        assert (
            optimized.req_states.req_id_to_index == ordinary.req_states.req_id_to_index
        )
        assert not optimized._late_prepared
    finally:
        lane.close()
