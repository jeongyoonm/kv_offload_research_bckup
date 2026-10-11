# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_offload import early_readiness as early
from vllm.v1.kv_offload.early_readiness import EarlyLoadReadiness
from vllm.v1.kv_offload.late_admission import AdmissionWindow, Candidate
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus


@pytest.mark.parametrize(
    "status,computed,ready,placeholders,expected_tokens,expected_unknown",
    [
        (RequestStatus.WAITING, 0, False, 0, 0, 1),
        (RequestStatus.PREEMPTED, 0, False, 0, 0, 1),
        (RequestStatus.WAITING, 29696, False, 0, 1024, 0),
        (RequestStatus.WAITING_FOR_REMOTE_KVS, 29696, False, 0, 0, 0),
        (RequestStatus.WAITING_FOR_REMOTE_KVS, 29696, True, 0, 1024, 0),
        (RequestStatus.WAITING_FOR_REMOTE_KVS, 30720, True, 0, 1, 0),
        (RequestStatus.RUNNING, 30720, False, 1, 1, 0),
        (RequestStatus.FINISHED_ABORTED, 29696, True, 0, 0, 0),
    ],
)
def test_deferral_capacity_uses_known_residual_work_without_discovery(
    status, computed, ready, placeholders, expected_tokens, expected_unknown
):
    scheduler = Scheduler.__new__(Scheduler)
    request = SimpleNamespace(
        request_id="req",
        status=status,
        num_computed_tokens=computed,
        num_tokens=30720,
        num_tokens_with_spec=30720,
        num_output_placeholders=placeholders,
        skip_reading_prefix_cache=False,
    )
    scheduler.requests = {"req": request}
    scheduler.finished_recving_kv_req_ids = {"req"} if ready else set()
    scheduler.connector = MagicMock()
    scheduler.kv_cache_manager = MagicMock(enable_caching=True)
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    scheduler.max_num_running_reqs = 32
    scheduler.max_num_scheduled_tokens = 61440
    before = vars(request).copy()
    for _ in range(2):
        snapshot = scheduler.deferral_capacity_snapshot()
        assert snapshot["known_tokens"] == expected_tokens
        assert snapshot["unexamined_requests"] == expected_unknown
        assert scheduler.deferral_has_capacity()
    assert vars(request) == before
    assert scheduler.finished_recving_kv_req_ids == ({"req"} if ready else set())
    assert not scheduler.connector.mock_calls
    assert not scheduler.kv_cache_manager.mock_calls


@pytest.mark.parametrize("limit", ["requests", "tokens", "uncached"])
def test_deferral_does_not_hold_when_known_work_saturates_budget(limit):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.requests = {
        str(i): SimpleNamespace(
            request_id=str(i),
            status=RequestStatus.WAITING,
            num_computed_tokens=0 if limit == "uncached" else 29696,
            num_tokens=30720,
            num_tokens_with_spec=30720,
            num_output_placeholders=0,
            skip_reading_prefix_cache=False,
        )
        for i in range(2)
    }
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.connector = None
    scheduler.kv_cache_manager = MagicMock(enable_caching=False)
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    scheduler.max_num_running_reqs = 2 if limit == "requests" else 32
    scheduler.max_num_scheduled_tokens = 2048 if limit == "tokens" else 61440
    assert not scheduler.deferral_has_capacity()


def _controller(monkeypatch, records, *, enabled=True, delay=0):
    monkeypatch.setattr(
        early.ops,
        "poll_async_batch_memcpy_completion_ring",
        lambda handle: (
            records[handle].popleft()
            if records[handle]
            else ([], [], [], [], False, False)
        ),
    )
    return EarlyLoadReadiness(7, list(range(8)), [], enabled, delay)


@pytest.mark.parametrize("policy", ["disabled", "timing"])
def test_all_ranks_are_required_and_completion_is_released_once(monkeypatch, policy):
    records = {
        rank: deque([([7], [11], [100 + rank], [1], False, False)]) for rank in range(8)
    }
    records[7].clear()
    readiness = _controller(monkeypatch, records)
    readiness.experiment_options["deferral_policy"] = policy
    assert readiness.drain() == []

    records[7].append(([7], [11], [107], [1], False, False))
    completed = readiness.drain()
    assert [(item.job_id, item.completion_ns) for item in completed] == [(11, 107)]
    assert completed[0].rank_completion_ns == (
        tuple(range(100, 108)) if policy == "timing" else ()
    )
    assert not readiness._rank_timestamps
    records[0].append(([7], [11], [108], [1], False, False))
    assert readiness.drain() == []


@pytest.mark.parametrize(
    "record, message",
    [
        (([], [], [], [], True, False), "overflowed"),
        (([7], [11], [100], [0], False, True), "failed"),
        (([8], [11], [100], [1], False, False), "stale"),
    ],
)
def test_invalid_native_notifications_fail_closed(monkeypatch, record, message):
    records: dict[int, deque] = {rank: deque() for rank in range(8)}
    records[0].append(record)
    readiness = _controller(monkeypatch, records)
    with pytest.raises(RuntimeError, match=message):
        readiness.drain()


def test_scheduler_early_readiness_is_stale_safe_and_deduplicated():
    scheduler = Scheduler.__new__(Scheduler)
    request = MagicMock()
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.requests = {"req": request}
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.connector = MagicMock()
    scheduler.connector.get_early_ready_request.return_value = "req"

    assert scheduler.apply_early_kv_load_completion(3) == "req"
    assert scheduler.apply_early_kv_load_completion(3) is None
    request.status = RequestStatus.FINISHED_ABORTED
    scheduler.finished_recving_kv_req_ids.clear()
    assert scheduler.apply_early_kv_load_completion(3) is None
    scheduler.connector.get_early_ready_request.return_value = None
    assert scheduler.apply_early_kv_load_completion(99) is None


def test_ordinary_readiness_after_early_promotion_is_deduplicated():
    scheduler = Scheduler.__new__(Scheduler)
    request = MagicMock()
    request.status = RequestStatus.RUNNING
    scheduler.requests = {"req": request}
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.connector = MagicMock()

    scheduler._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={"req"}))

    scheduler.connector.update_connector_output.assert_called_once()
    assert scheduler.finished_recving_kv_req_ids == set()


def test_finished_early_request_keeps_blocks_until_ordinary_ack():
    connector = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    request = MagicMock()
    request.request_id = "req"
    req_status = MagicMock()
    req_status.transfer_jobs = {4}
    job_status = MagicMock()
    job_status.is_store = False
    job_status.deferred_fence_block_ids = None
    connector._req_status = {"req": req_status}
    connector._jobs = {4: job_status}
    connector._block_id_to_pending_jobs = {}
    connector._maybe_observe_lookup_async_delay = MagicMock()

    delay_free, params = connector.request_finished(request)

    assert delay_free
    assert params is None


def test_finished_early_request_defers_store_until_load_ack():
    connector = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    request = MagicMock()
    request.request_id = "req"
    request.is_finished.return_value = True
    req_status = MagicMock()
    req_status.req = request
    req_status.transfer_jobs = {4}
    connector.config = MagicMock(blocks_per_chunk=1)
    connector._req_status = {"req": req_status}
    connector._jobs = {4: MagicMock(is_store=False)}
    connector._deferred_finished_req_ids = set()

    output = MagicMock()
    output.num_scheduled_tokens = {}
    output.finished_req_ids = {"req"}
    assert connector._build_store_jobs(output) == {}
    assert connector._deferred_finished_req_ids == {"req"}
    assert connector.has_pending_push_work()


@pytest.mark.parametrize(
    "delay_ms, expected_sleeps", [(0, []), (5, [0.005]), (120, [0.120])]
)
def test_batch_queue_delay_and_refresh(monkeypatch, delay_ms, expected_sleeps):
    core = EngineCore.__new__(EngineCore)
    core.early_load_readiness = MagicMock(schedule_delay_ms=delay_ms)
    core.scheduler = MagicMock()
    core.scheduler.has_waiting_kv_loads.return_value = True
    core._refresh_early_load_readiness = MagicMock()
    sleeps: list[float] = []
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", sleeps.append)

    in_flight: Future = Future()
    output = MagicMock(total_num_scheduled_tokens=1)
    core._prepare_batch_queue_scheduling(deque([(MagicMock(), output, in_flight)]))
    assert sleeps == expected_sleeps
    core._refresh_early_load_readiness.assert_called_once_with()


def test_idle_batch_queue_is_never_delayed(monkeypatch):
    core = EngineCore.__new__(EngineCore)
    core.early_load_readiness = MagicMock(schedule_delay_ms=120)
    core.scheduler = MagicMock()
    core.scheduler.has_waiting_kv_loads.return_value = True
    core._refresh_early_load_readiness = MagicMock()
    sleep = MagicMock()
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", sleep)
    core._prepare_batch_queue_scheduling(deque())
    sleep.assert_not_called()


@pytest.mark.parametrize("delay", [-1, 121, float("nan"), float("inf"), True, "20"])
def test_runtime_delay_rejects_invalid_values_without_changing_settings(
    monkeypatch, delay
):
    readiness = _controller(monkeypatch, {rank: deque() for rank in range(8)})
    with pytest.raises(ValueError, match="0 to 120"):
        readiness.configure(True, delay)
    assert readiness.schedule_delay_ms == 0


def test_runtime_delay_can_change_from_maximum_back_to_zero(monkeypatch):
    readiness = _controller(monkeypatch, {rank: deque() for rank in range(8)})
    readiness.configure(True, 120)
    assert readiness.schedule_delay_ms == 120
    readiness.configure(False, 0)
    assert readiness.schedule_delay_ms == 0
    assert not readiness.enabled


@pytest.mark.parametrize("delay, valid", [(0, True), (120, True), (120.01, False)])
def test_endpoint_delay_validation(delay, valid):
    from pydantic import ValidationError

    from vllm.entrypoints.serve.experimental.kv_load_scheduling import (
        KVLoadSchedulingRequest,
    )

    if valid:
        assert (
            KVLoadSchedulingRequest(schedule_delay_ms=delay).schedule_delay_ms == delay
        )
    else:
        with pytest.raises(ValidationError):
            KVLoadSchedulingRequest(schedule_delay_ms=delay)


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["127.0.0.1", "192.0.2.1"])
async def test_estimator_endpoint_preserves_loopback_guard_and_forwards_state(host):
    from unittest.mock import AsyncMock

    from fastapi import HTTPException
    from starlette.requests import Request

    from vllm.entrypoints.serve.experimental.kv_load_scheduling import (
        KVLoadSchedulingRequest,
        set_kv_load_scheduling,
    )

    rpc = AsyncMock(return_value={"seal_estimator": {"loaded": True}})
    app = SimpleNamespace(
        state=SimpleNamespace(
            engine_client=SimpleNamespace(set_kv_load_scheduling=rpc),
            vllm_config=SimpleNamespace(
                kv_transfer_config=SimpleNamespace(
                    kv_connector_extra_config={"early_load_completion": True}
                )
            ),
        )
    )
    raw = Request({"type": "http", "client": (host, 1234), "app": app})
    settings = KVLoadSchedulingRequest(seal_estimator={"action": "load", "state": {}})
    if host == "127.0.0.1":
        await set_kv_load_scheduling(settings, raw)
        rpc.assert_awaited_once_with(False, 0, None, settings.seal_estimator)
    else:
        with pytest.raises(HTTPException) as exc:
            await set_kv_load_scheduling(settings, raw)
        assert exc.value.status_code == 403
        rpc.assert_not_awaited()


@pytest.mark.parametrize("pending", [False, True])
def test_estimator_restore_rejection_does_not_replace_live_model(monkeypatch, pending):
    from vllm.v1.kv_offload.seal_predictor import ComputeEstimator

    core = EngineCore.__new__(EngineCore)
    original = ComputeEstimator()
    options = dict(early.EXPERIMENT_DEFAULTS)
    core.early_load_readiness = SimpleNamespace(
        experiment_options=options, configure=MagicMock()
    )
    core._refresh_early_load_readiness = MagicMock()
    core.scheduler = SimpleNamespace(has_requests=lambda: pending)
    core.batch_queue = deque()
    core.deferral_timing = SimpleNamespace(seal_estimator=original, collect=MagicMock())
    core.vllm_config = SimpleNamespace()
    monkeypatch.setattr(
        "vllm.v1.kv_offload.seal_predictor.estimator_context", lambda _: {}
    )
    with pytest.raises(RuntimeError if pending else ValueError):
        core.set_kv_load_scheduling(True, 0, {}, {"action": "load", "state": {}})
    assert core.deferral_timing.seal_estimator is original
    assert core.early_load_readiness.experiment_options == options
    core.early_load_readiness.configure.assert_not_called()


def test_completion_during_delay_is_ready_for_immediate_successor(monkeypatch):
    """A completed predecessor must not prevent the post-delay readiness drain."""
    records: dict[int, deque] = {rank: deque() for rank in range(8)}
    core = EngineCore.__new__(EngineCore)
    core.early_load_readiness = _controller(monkeypatch, records, delay=120)
    scheduler = Scheduler.__new__(Scheduler)
    request = MagicMock(status=RequestStatus.WAITING_FOR_REMOTE_KVS)
    scheduler.requests = {"req": request}
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.connector = MagicMock()
    scheduler.connector.get_early_ready_request.return_value = "req"
    core.scheduler = scheduler
    predecessor: Future = Future()
    sleeps: list[float] = []

    def complete(seconds):
        sleeps.append(seconds)
        predecessor.set_result(None)
        for rank in records:
            records[rank].append(([7], [11], [100 + rank], [1], False, False))

    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", complete)
    output = MagicMock(total_num_scheduled_tokens=1)
    queue: deque = deque([(MagicMock(), output, predecessor)])
    core._prepare_batch_queue_scheduling(queue)
    assert sleeps == [0.120]
    assert scheduler.readiness_snapshot() == {"req": True}
    core._prepare_batch_queue_scheduling(queue)
    assert sleeps == [0.120]


@pytest.mark.parametrize("delay", [0, 120, 121])
def test_configuration_delay_limit(monkeypatch, delay):
    from types import SimpleNamespace

    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "early_load_completion": True,
                "schedule_delay_ms": delay,
                "async_h2d_submission": True,
            }
        ),
        parallel_config=SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(
        early.ops, "create_async_batch_memcpy_completion_ring", lambda *a: 1
    )
    monkeypatch.setattr(
        early.ops, "destroy_async_batch_memcpy_completion_ring", lambda *a: None
    )
    if delay > 120:
        with pytest.raises(ValueError, match="0 to 120"):
            EarlyLoadReadiness.from_config(config)
    else:
        readiness = EarlyLoadReadiness.from_config(config)
        assert readiness.schedule_delay_ms == delay
        readiness.close()


def test_opportunity_tracks_predecessor_and_zero_delay_cutoff(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.kv_offload import timeline

    core = EngineCore.__new__(EngineCore)
    core.early_load_readiness = MagicMock(schedule_delay_ms=0)
    core.scheduler = MagicMock()
    core.scheduler.has_waiting_kv_loads.return_value = True
    core.scheduler.readiness_snapshot.return_value = {"req": False}
    events = []
    monkeypatch.setattr(timeline, "enabled", lambda: True)
    monkeypatch.setattr(
        timeline, "emit", lambda event, **kw: events.append((event, kw))
    )
    core._refresh_early_load_readiness = MagicMock()
    output = SimpleNamespace(total_num_scheduled_tokens=1, readiness_timeline_step=42)
    queue: deque = deque([(Future(), output, Future())])
    assert core._prepare_batch_queue_scheduling(queue) == 1
    assert core._prepare_batch_queue_scheduling(queue) == 2
    assert events[0] == (
        "opportunity_begin",
        {
            "opportunity_id": 1,
            "predecessor_steps": [42],
            "delay_ms": 0,
            "candidates": {"req": False},
        },
    )
    assert events[1] == ("readiness_cutoff", {"opportunity_id": 1})


def test_connector_only_work_does_not_delay(monkeypatch):
    core = EngineCore.__new__(EngineCore)
    core.early_load_readiness = MagicMock(schedule_delay_ms=120)
    core.scheduler = MagicMock()
    core.scheduler.has_waiting_kv_loads.return_value = True
    core._refresh_early_load_readiness = MagicMock()
    sleep = MagicMock()
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", sleep)
    output = MagicMock(total_num_scheduled_tokens=0)
    core._prepare_batch_queue_scheduling(deque([(Future(), output, Future())]))
    sleep.assert_not_called()
    core._refresh_early_load_readiness.assert_called_once()


# Deferral owns only the decision to call schedule. These CPU tests exercise
# its clock/snapshot contract and fail-open timing behavior without CUDA.
def _hold(**kw):
    from vllm.v1.kv_offload.scheduling_deferral import Hold

    args = dict(
        predecessor=7,
        now=100_000_000,
        dispatch=90_000_000,
        duration=200_000_000,
        reserve=20_000_000,
        snapshot={"a": False, "b": False},
        cap_ms=120,
        margin_ms=10,
        gain=2,
    )
    args.update(kw)
    return Hold.begin(**args)


def _poll(hold, snapshot=None, **kw):
    args = dict(
        now=110_000_000,
        snapshot={"a": False, "b": False} if snapshot is None else snapshot,
        policy="readiness",
        timing_valid=True,
        completed=False,
        capacity=True,
        outstanding=1,
    )
    args.update(kw)
    return hold.poll(**args)


def test_deferral_deadline_is_immutable_and_arrivals_do_not_count():
    hold = _hold()
    assert hold.deadline_ns == 220_000_000
    assert _poll(hold, {"a": True, "b": False, "arrival": True}) is None
    assert hold.deadline_ns == 220_000_000
    assert _poll(hold, {"a": True, "b": True}) == "readiness_target"


def test_deadline_policy_ignores_target_but_not_deadline():
    hold = _hold()
    assert _poll(hold, {"a": True, "b": True}, policy="deadline") is None
    assert _poll(hold, now=hold.deadline_ns, policy="deadline") == "deadline"


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"timing_valid": False}, "invalid_timing"),
        ({"completed": True}, "predecessor_completed"),
        ({"outstanding": 0}, "predecessor_completed"),
        ({"outstanding": 2}, "outstanding_compute_changed"),
        ({"capacity": False}, "capacity_saturated"),
        ({"snapshot": {}}, "candidates_cancelled"),
    ],
)
def test_deferral_safety_exits(options, reason):
    assert _poll(_hold(), **options) == reason


def test_cancelled_candidate_reduces_target_and_zero_budget_never_holds():
    assert _poll(_hold(), {"a": True}) == "readiness_target"
    assert _hold(cap_ms=0) is None
    assert _hold(duration=10_000_000) is None
    assert _hold(snapshot={"a": True}) is None


@pytest.mark.parametrize(
    "options",
    [
        {"deferral_policy": "bogus"},
        {"deferral_hold_ms": 121},
        {"deferral_hold_ms": -1},
        {"deferral_hold_ms": True},
        {"deferral_target_gain": 0},
        {"deferral_margin_ms": -1},
        {"deferral_start_wait_ms": 121},
        {"deferral_start_wait_ms": True},
        {"deferral_primary_retry_ms": -1},
        {"deferral_primary_retry_ms": 11},
        {"deferral_primary_retry_ms": True},
        {"deferral_policy": "readiness", "optimistic_cap": 1},
        {"deferral_policy": "deadline", "background_prepare": True},
    ],
)
def test_invalid_deferral_settings(options):
    with pytest.raises(ValueError):
        early.validate_experiment_options(options)


def test_predictor_requires_supported_observed_shapes_and_sample_count():
    from vllm.v1.kv_offload.scheduling_deferral import Predictor, bucket

    shape = ("prefill", 3, 3000, 10000)
    data: dict = dict(
        version=1,
        calibration_cells=4,
        reserve_ns=100,
        buckets={
            bucket(shape): dict(
                count=20,
                duration_ns=1000,
                minimum=[3, 3000, 10000],
                maximum=[4, 4000, 11000],
            )
        },
    )
    predictor = Predictor(data)
    assert predictor.estimate(shape) == 1000
    assert predictor.estimate(("prefill", 3, 2999, 10000)) is None
    assert predictor.estimate(("decode", 3, 3000, 10000)) is None
    data["buckets"][bucket(shape)]["count"] = 19
    with pytest.raises(ValueError):
        Predictor(data)


def test_rank_timing_missing_stale_overwritten_and_all_rank_completion(tmp_path):
    from vllm.v1.kv_offload.step_timing import CAPACITY, RankFile, TimingChannel

    timing = TimingChannel.__new__(TimingChannel)
    timing.ranks = [RankFile(str(tmp_path / str(i)), create=True) for i in range(8)]
    timing.steps = {7: dict(dispatch_ns=1)}
    timing.registrations = {7: (7, 1)}
    try:
        assert timing.read(7, 100)[0] is False
        for rank in timing.ranks:
            rank.write([7, 10, 20, 30, 40, 50, 50, 1, 0])
        assert timing.read(7, 100)[:2] == (True, True)
        timing.ranks[-1].write([7, 10, 20, 30, 40, 0, 50, 1, 0])
        assert timing.read(7, 100)[:2] == (True, False)
        assert timing.read(7, 20_000_000)[0] is False
        timing.ranks[-1].write([7 + CAPACITY, 10, 20, 30, 40, 50, 50, 1, 0])
        assert timing.read(7, 100)[0] is False
    finally:
        for rank in timing.ranks:
            rank.close()


def test_cooperative_deferral_does_not_call_scheduler_and_preserves_kv_failure(
    monkeypatch,
):
    from types import SimpleNamespace

    core = EngineCore.__new__(EngineCore)
    core.early_load_readiness = SimpleNamespace(
        experiment_options={**early.EXPERIMENT_DEFAULTS, "deferral_policy": "readiness"}
    )
    core.scheduler = MagicMock()
    core.scheduler.readiness_snapshot.return_value = {"a": False, "b": False}
    core.scheduler.deferral_has_capacity.return_value = True
    core._process_aborts_queue = MagicMock()
    core._refresh_early_load_readiness = MagicMock()
    core.deferral_timing = MagicMock()
    core.deferral_timing.hold = _hold()
    core.deferral_timing.read.return_value = True, False, []
    monkeypatch.setattr("vllm.v1.engine.core.time.monotonic_ns", lambda: 110_000_000)
    sleep = MagicMock()
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", sleep)
    queue: deque = deque(
        [(Future(), SimpleNamespace(total_num_scheduled_tokens=1), Future())]
    )
    assert core._defer_batch_queue_scheduling(queue)
    sleep.assert_called_once_with(0.001)
    core.scheduler.schedule.assert_not_called()
    core.scheduler.readiness_snapshot.return_value = {"a": True, "b": True}
    assert not core._defer_batch_queue_scheduling(queue)
    assert core.deferral_timing.hold is None
    core._refresh_early_load_readiness.side_effect = RuntimeError("KV failed")
    with pytest.raises(RuntimeError, match="KV failed"):
        core._defer_batch_queue_scheduling(queue)


def test_deferred_engine_schedules_exactly_once_when_hold_ends():
    from contextlib import nullcontext
    from types import SimpleNamespace

    core = EngineCore.__new__(EngineCore)
    core.batch_queue = deque([(Future(), MagicMock(), Future())])
    core.batch_queue_size = 3
    core.early_load_readiness = None
    core.scheduler = MagicMock()
    output = SimpleNamespace(
        total_num_scheduled_tokens=1, pending_structured_output_tokens=False
    )
    core.scheduler.schedule.return_value = output
    core._defer_batch_queue_scheduling = MagicMock(side_effect=[True, False])
    core._prepare_batch_queue_scheduling = MagicMock(return_value=None)
    core._timing_enabled = MagicMock(return_value=False)
    core._record_deferral_dispatch = MagicMock()
    core._should_throttle_prefills = MagicMock(return_value=False)
    core.log_error_detail = lambda _: nullcontext()
    core.model_executor = MagicMock()
    core.is_ec_consumer = True
    core.is_pooling_model = False
    assert core.step_with_batch_queue() == (None, False)
    core.scheduler.schedule.assert_not_called()
    assert core.step_with_batch_queue() == (None, True)
    core.scheduler.schedule.assert_called_once()
    core.model_executor.execute_model.assert_called_once()


def test_timing_writer_failure_is_not_a_kv_failure():
    from vllm.v1.kv_offload.step_timing import WorkerTiming

    timing = WorkerTiming.__new__(WorkerTiming)
    timing.rank = MagicMock()
    timing.rank.write.side_effect = OSError("timing transport unavailable")
    timing.begin(7)
    assert timing.failed
    assert timing.current is None
    timing.begin(8)
    assert timing.rank.write.call_args.args[0][0] == 8
    assert timing.rank.write.call_args.args[0][7] == 0


def _two_phase_core(monkeypatch):
    core = EngineCore.__new__(EngineCore)
    core.early_load_readiness = SimpleNamespace(
        experiment_options={
            **early.EXPERIMENT_DEFAULTS,
            "deferral_policy": "two_phase_pilot",
        }
    )
    core.scheduler = MagicMock()
    timing = core.deferral_timing = MagicMock()
    timing.hold = None
    timing.opportunity = 0
    timing.steps = {7: dict(step=7, execution_config="cfg")}
    timing.predictor.data = dict(version=4, ranks=8, execution_config="cfg")
    timing.predictor.reserve_ns = 77_578_048
    timing.predictor.estimate_completion.return_value = SimpleNamespace(
        completion_ns=500_000_000, reason="observed_all_rank_gpu_start"
    )
    timing.read.return_value = True, False, []
    timing.read_execution.return_value = [[7, 0, 0, 0, 1, 2, 0, 0, 0]] * 8
    timing.read_execution_with_status.side_effect = lambda step: (
        timing.read_execution.return_value,
        ["ok"] * 8,
    )
    clock = [100_000_000]
    monkeypatch.setattr("vllm.v1.engine.core.time.monotonic_ns", lambda: clock[0])
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", MagicMock())
    return core, timing, clock, [SimpleNamespace(readiness_timeline_step=7)]


def test_two_phase_start_transition_keeps_original_cap_and_candidate_snapshot(
    monkeypatch,
):
    core, timing, clock, outstanding = _two_phase_core(monkeypatch)
    snapshot = dict(a=False, b=False)
    assert core._defer_two_phase_scheduling(snapshot, outstanding, 1, True)
    assert timing.hold.deadline_ns == 130_000_000
    assert timing.hold.total_deadline_ns == 220_000_000
    clock[0] = 120_000_000
    timing.read_execution.return_value = [
        [7, 101_000_000, 110_000_000, 111_000_000, 111_000_000, 1, 0, 0, 0]
    ] * 8
    assert core._defer_two_phase_scheduling(
        dict(snapshot, arrival=True), outstanding, 1, True
    )
    assert timing.hold.phase == "execution"
    assert timing.hold.deadline_ns == 220_000_000
    assert timing.hold.candidates == frozenset(("a", "b"))
    clock[0] = 121_000_000
    assert core._defer_two_phase_scheduling(snapshot, outstanding, 1, True)
    timing.predictor.estimate_completion.assert_called_once()
    assert not core._defer_two_phase_scheduling(
        dict(a=True, b=True), outstanding, 1, True
    )
    assert timing.hold is None
    core.scheduler.schedule.assert_not_called()


@pytest.mark.parametrize(
    "exit",
    [
        "timeout",
        "completed",
        "invalid",
        "cancelled",
        "ready",
        "capacity",
        "successor",
        "unsupported",
    ],
)
def test_two_phase_wait_exits_without_scheduling_speculatively(monkeypatch, exit):
    core, timing, clock, outstanding = _two_phase_core(monkeypatch)
    snapshot = dict(a=False, b=False)
    assert core._defer_two_phase_scheduling(snapshot, outstanding, 1, True)
    expected = {
        "timeout": "start_timeout",
        "completed": "predecessor_completed",
        "invalid": "invalid_timing",
        "cancelled": "candidates_cancelled",
        "ready": "readiness_target",
        "capacity": "capacity_saturated",
        "successor": "outstanding_compute_changed",
        "unsupported": "unvalidated_execution_shape",
    }[exit]
    clock[0] += 31_000_000 if exit == "timeout" else 1_000_000
    if exit == "completed":
        timing.read.return_value = True, True, []
    if exit == "invalid":
        timing.read_execution.return_value = [None] * 8
    if exit == "cancelled":
        snapshot = {}
    if exit == "ready":
        snapshot = dict(a=True, b=True)
    if exit == "unsupported":
        timing.read_execution.return_value = [[7, 1, 1, 2, 2, 1, 0, 0, 0]] * 8
        timing.predictor.estimate_completion.return_value = SimpleNamespace(
            completion_ns=None, reason=expected
        )
    assert not core._defer_two_phase_scheduling(
        snapshot, outstanding, 2 if exit == "successor" else 1, exit != "capacity"
    )
    assert timing.emit.call_args.kwargs["reason"] == expected
    assert timing.hold is None
    core.scheduler.schedule.assert_not_called()


@pytest.mark.parametrize(
    "case", ["idle", "two_compute", "zero_cap", "zero_start", "no_candidates", "config"]
)
def test_two_phase_does_not_enter_without_eligible_work(monkeypatch, case):
    core, timing, clock, outstanding = _two_phase_core(monkeypatch)
    if case == "zero_cap":
        core.early_load_readiness.experiment_options["deferral_hold_ms"] = 0
    if case == "zero_start":
        core.early_load_readiness.experiment_options["deferral_start_wait_ms"] = 0
    if case == "config":
        timing.steps[7]["execution_config"] = "other"
    assert not core._defer_two_phase_scheduling(
        {} if case == "no_candidates" else dict(a=False),
        [] if case == "idle" else outstanding,
        2 if case == "two_compute" else 1,
        True,
    )
    assert timing.hold is None


def test_pending_start_state_does_not_mask_a_failed_rank():
    from vllm.v1.kv_offload.execution_timing import start_snapshot_state

    records = [[7, 0, 0, 0, 1, 2, 0, 0, 0] for _ in range(8)]
    assert start_snapshot_state(records, 7, 10, 8) == "pending"
    records[7][5] = 0
    assert start_snapshot_state(records, 7, 10, 8) == "invalid"
    assert start_snapshot_state(records[:7], 7, 10, 8) == "invalid"


def test_active_predictor_requires_held_out_coverage(tmp_path):
    import json

    from vllm.v1.kv_offload.scheduling_deferral import Predictor

    path = tmp_path / "predictor.json"
    data: dict = dict(
        version=1,
        calibration_cells=4,
        reserve_ns=100,
        buckets={},
        frozen=True,
        shadow_coverage=0.79,
    )
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="80%"):
        Predictor.load(str(path), active=True)
    data["shadow_coverage"] = 0.8
    path.write_text(json.dumps(data))
    Predictor.load(str(path), active=True)


@pytest.mark.parametrize("execution_enabled", [False, True])
def test_worker_marker_includes_v2_async_output_copy(tmp_path, execution_enabled):
    import time
    from types import SimpleNamespace

    import torch

    from vllm.v1.kv_offload.step_timing import RankFile, WorkerTiming

    if not torch.cuda.is_available():
        pytest.skip("CUDA marker test")
    path = str(tmp_path / "rank")
    owner = RankFile(path, create=True)
    execution_owner = (
        RankFile(path + ".execution", create=True) if execution_enabled else None
    )
    worker = WorkerTiming(path, torch.device("cuda:0"))
    worker.begin(7)
    worker.prepared()
    copy = torch.cuda.Stream()
    complete = torch.cuda.Event()
    with torch.cuda.stream(copy):
        torch.cuda._sleep(50_000_000)
        complete.record()
    worker.finish(SimpleNamespace(copy_event=complete))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = owner.read(7)
        if record is not None and record[5]:
            assert complete.query()
            assert 0 < record[1] <= record[2] <= record[3] <= record[4] <= record[5]
            break
        time.sleep(0.001)
    else:
        pytest.fail("completion marker was not published")
    if execution_owner is not None:
        while time.monotonic() < deadline:
            execution = execution_owner.read(7)
            if execution is not None and execution[8]:
                assert complete.query()
                assert execution[6] > 0
                break
            time.sleep(0.001)
        else:
            pytest.fail("execution duration omitted the output-copy join")
        worker.execution.close()
        execution_owner.close()
    worker.rank.close()
    owner.close()


@pytest.mark.parametrize(
    "policy,cap,count,valid,supported,capacity,expected",
    [
        ("disabled", 120, 1, True, True, True, False),
        ("readiness", 0, 1, True, True, True, False),
        ("readiness", 120, 0, True, True, True, False),
        ("readiness", 120, 2, True, True, True, False),
        ("readiness", 120, 1, False, True, True, False),
        ("readiness", 120, 1, True, False, True, False),
        ("readiness", 120, 1, True, True, False, False),
        ("shadow", 120, 1, True, True, True, False),
        ("timing", 120, 1, True, True, True, False),
        ("readiness", 120, 1, True, True, True, True),
    ],
)
def test_engine_deferral_eligibility(
    monkeypatch,
    policy,
    cap,
    count,
    valid,
    supported,
    capacity,
    expected,
):
    from types import SimpleNamespace

    core = EngineCore.__new__(EngineCore)
    core.early_load_readiness = SimpleNamespace(
        experiment_options={
            **early.EXPERIMENT_DEFAULTS,
            "deferral_policy": policy,
            "deferral_hold_ms": cap,
        }
    )
    core.scheduler = MagicMock()
    core.scheduler.readiness_snapshot.return_value = {"a": False, "b": False}
    core.scheduler.deferral_has_capacity.return_value = capacity
    core._refresh_early_load_readiness = MagicMock()
    core._process_aborts_queue = MagicMock()
    predictor = SimpleNamespace(
        data={},
        reserve_ns=20_000_000,
        estimate_completion=lambda *args: SimpleNamespace(
            completion_ns=290_000_000 if supported else None,
            model="test",
            reason="test",
        ),
    )
    core.deferral_timing = SimpleNamespace(
        hold=None,
        predictor=predictor,
        collect=lambda: None,
        observe=MagicMock(),
        watch=MagicMock(),
        read=lambda *args: (valid, False, []),
        read_execution=lambda *args: [],
        emit=MagicMock(),
        steps={7: dict(dispatch_ns=90_000_000, shape=("prefill", 4, 4096, 16000))},
    )
    monkeypatch.setattr("vllm.v1.engine.core.time.monotonic_ns", lambda: 100_000_000)
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", MagicMock())
    queue: deque = deque(
        (
            Future(),
            SimpleNamespace(total_num_scheduled_tokens=4096, readiness_timeline_step=7),
            Future(),
        )
        for _ in range(count)
    )
    assert core._defer_batch_queue_scheduling(queue) is expected
    core.scheduler.schedule.assert_not_called()
    if policy == "readiness" and cap == 0:
        fields = core.deferral_timing.emit.call_args.kwargs
        assert fields["predicted_ns"] == 290_000_000
        assert fields["deadline_ns"] == 100_000_000
        assert fields["reserve_ns"] == 20_000_000


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1, 0])
def test_corrupt_predictor_cannot_turn_invalid_duration_into_hold_cap(duration):
    from vllm.v1.kv_offload.scheduling_deferral import Predictor

    with pytest.raises(ValueError, match="bucket"):
        Predictor(
            dict(
                version=1,
                calibration_cells=4,
                reserve_ns=1,
                buckets={
                    "prefill:1:1024:32768": dict(
                        count=20,
                        duration_ns=duration,
                        minimum=[1, 1024, 30720],
                        maximum=[1, 1024, 30720],
                    )
                },
            )
        )


@pytest.mark.parametrize(
    "policy,delay,timing,pending,error",
    [
        ("timing", 5, True, False, ValueError),
        ("timing", 0, False, False, ValueError),
        ("deadline", 0, True, False, ValueError),
        ("timing", 0, True, True, RuntimeError),
    ],
)
def test_runtime_deferral_rejection_is_atomic(
    monkeypatch, policy, delay, timing, pending, error
):
    from types import SimpleNamespace

    monkeypatch.delenv("VLLM_DEFERRAL_PREDICTOR", raising=False)
    core = EngineCore.__new__(EngineCore)
    options = dict(early.EXPERIMENT_DEFAULTS)
    core.early_load_readiness = SimpleNamespace(
        experiment_options=options, configure=MagicMock()
    )
    core.deferral_timing = MagicMock() if timing else None
    core.scheduler = MagicMock()
    core.scheduler.has_requests.return_value = pending
    core.batch_queue = deque()
    core._refresh_early_load_readiness = MagicMock()
    with pytest.raises(error):
        core.set_kv_load_scheduling(True, delay, {"deferral_policy": policy})
    core.early_load_readiness.configure.assert_not_called()
    assert core.early_load_readiness.experiment_options == options


def test_two_phase_runtime_requires_explicit_pilot_opt_in(monkeypatch):
    from vllm.v1.kv_offload.scheduling_deferral import Predictor

    monkeypatch.delenv("VLLM_DEFERRAL_ALLOW_PILOT", raising=False)
    monkeypatch.setenv("VLLM_DEFERRAL_PREDICTOR", "unused")
    monkeypatch.setattr(
        Predictor,
        "load",
        lambda *a, **kw: SimpleNamespace(data=dict(version=4, development_only=True)),
    )
    core = EngineCore.__new__(EngineCore)
    core.early_load_readiness = SimpleNamespace(
        configure=MagicMock(), enabled=True, schedule_delay_ms=0
    )
    core.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={})
    )
    core.deferral_timing = MagicMock()
    core.scheduler = MagicMock()
    core.scheduler.has_requests.return_value = False
    core.batch_queue = deque()
    core._refresh_early_load_readiness = MagicMock()
    with pytest.raises(ValueError, match="development opt-in"):
        core.set_kv_load_scheduling(True, 0, {"deferral_policy": "two_phase_pilot"})
    core.early_load_readiness.configure.assert_not_called()
    monkeypatch.setenv("VLLM_DEFERRAL_ALLOW_PILOT", "1")
    assert (
        core.set_kv_load_scheduling(True, 0, {"deferral_policy": "two_phase_pilot"})[
            "deferral_start_wait_ms"
        ]
        == 30
    )


def test_registered_pending_timing_transitions_and_timeout(tmp_path):
    """Queued work is usable, but missing ranks and overdue preparation are not."""
    from vllm.v1.kv_offload.step_timing import PENDING_NS, RankFile, TimingChannel

    timing = TimingChannel.__new__(TimingChannel)
    timing.ranks = [RankFile(str(tmp_path / str(i)), create=True) for i in range(8)]
    timing.steps = {}
    timing.registrations = {}
    try:
        assert timing.read(7, 100)[:2] == (False, False)
        assert timing.register(dict(step=7, dispatch_ns=10))
        assert timing.read(7, 100)[:2] == (True, False)
        timing.ranks[0].write([7, 20, 0, 0, 0, 0, 20, 1, 0])
        timing.ranks[1].write([7, 20, 30, 0, 0, 0, 30, 1, 0])
        timing.ranks[2].write([7, 20, 30, 40, 50, 60, 60, 1, 0])
        assert timing.read(7, 100)[:2] == (True, False)
        assert timing.read(7, 11 + PENDING_NS)[:2] == (False, False)
        timing.ranks[-1].write([7, 20, 0, 0, 0, 0, 20, 0, 0])
        assert timing.read(7, 100)[:2] == (False, False)
    finally:
        for rank in timing.ranks:
            rank.close()


@pytest.mark.parametrize(
    "record",
    [
        [7, 20, 19, 0, 0, 0, 30, 1],
        [7, 20, 30, 0, 40, 0, 40, 1],
        [7, 0, 0, 0, 0, 0, 11, 2],
        [7, 9, 0, 0, 0, 0, 20, 1],
    ],
)
def test_pending_timing_rejects_inconsistent_records(tmp_path, record):
    from vllm.v1.kv_offload.step_timing import RankFile, TimingChannel

    timing = TimingChannel.__new__(TimingChannel)
    timing.ranks = [RankFile(str(tmp_path / "rank"), create=True)]
    timing.steps = {}
    timing.registrations = {}
    try:
        assert timing.register(dict(step=7, dispatch_ns=10))
        timing.ranks[0].write([*record, 0])
        assert timing.read(7, 100)[:2] == (False, False)
    finally:
        timing.ranks[0].close()


def test_completed_timing_survives_collection(tmp_path, monkeypatch):
    from vllm.v1.kv_offload.step_timing import RankFile, TimingChannel

    timing = TimingChannel.__new__(TimingChannel)
    timing.ranks = [RankFile(str(tmp_path / "rank"), create=True)]
    timing.steps = {}
    timing.registrations = {}
    timing.events = []
    timing.seal_estimator = MagicMock()
    monkeypatch.setattr("vllm.v1.kv_offload.step_timing.time.monotonic_ns", lambda: 100)
    try:
        assert timing.register(dict(step=7, dispatch_ns=10))
        timing.ranks[0].write([7, 20, 30, 40, 50, 60, 60, 1, 0])
        timing.collect()
        assert not timing.steps
        assert timing.read(7, 100)[:2] == (True, True)
        timing.collect()
        assert len(timing.events) == 1
        timing.audit = MagicMock()
        timing.watches = {}
        timing.watch(7, 1, 100, {"request": False})
        timing.audit.begin.assert_called_once_with(
            1,
            100,
            dict(step=7, dispatch_ns=10, completed_before_watch=True),
            {"request": False},
        )
        assert not timing.steps
    finally:
        timing.ranks[0].close()


def test_failed_registration_does_not_authorize_pending_hold(tmp_path):
    from vllm.v1.kv_offload.step_timing import RankFile, TimingChannel

    timing = TimingChannel.__new__(TimingChannel)
    timing.ranks = [RankFile(str(tmp_path / "rank"), create=True), MagicMock()]
    timing.ranks[1].write.side_effect = OSError("timing transport unavailable")
    timing.steps = {}
    timing.registrations = {}
    try:
        assert not timing.register(dict(step=7, dispatch_ns=10))
        assert timing.read(7, 100)[:2] == (False, False)
    finally:
        timing.ranks[0].close()


def _batch_artifact():
    def model(size):
        return dict(
            coefficients=[0.0] * size,
            mean=[0.0] * size,
            scale=[1.0] * size,
            intercept=200.0,
            correction_ms=-10.0,
        )

    return dict(
        version=3,
        feature_version=1,
        reserve_ns=77_578_048.34,
        selected="progress",
        domain=dict(
            phase="prefill", new_tokens=1024, context=29696, max_requests=32, ranks=8
        ),
        support={"2": dict(dispatch=True, prepared=True)},
        models=dict(shape=model(27), progress=model(34), prepared=model(27)),
    )


def test_batch_predictor_uses_latest_preparation_and_dispatch_fallback():
    from vllm.v1.kv_offload.scheduling_deferral import Predictor

    predictor = Predictor(_batch_artifact())
    data = dict(
        shape=["prefill", 2, 2048, 30720],
        request_shapes=[[1024, 29696]] * 2,
        dispatch_ns=1_000_000,
    )
    ranks = [
        [1, 2_000_000, 3_000_000 + i, 0, 0, 0, 4_000_000, 1, 1_000_000]
        for i in range(8)
    ]
    result = predictor.estimate_completion(data, 5_000_000, ranks)
    assert result.model == "prepared"
    assert result.completion_ns == 193_000_007
    ranks[7][2] = 0
    result = predictor.estimate_completion(data, 5_000_000, ranks)
    assert result.model == "progress"
    assert result.completion_ns == 191_000_000
    assert result.reason == "dispatch_fallback"


def test_batch_predictor_rejects_future_features_and_unsupported_shapes():
    from vllm.v1.kv_offload.scheduling_deferral import Predictor

    predictor = Predictor(_batch_artifact())
    data = dict(
        shape=["prefill", 2, 2048, 30720],
        request_shapes=[[1024, 29696]] * 2,
        dispatch_ns=1_000_000,
    )
    ranks = [[1, 2_000_000, 0, 0, 0, 0, 6_000_000, 1, 1_000_000]] * 8
    assert predictor.estimate_completion(data, 5_000_000, ranks).completion_ns is None
    data["request_shapes"] = [[1024, 30000], [1024, 29696]]
    assert predictor.estimate_completion(data, 5_000_000, []).reason == "outside_domain"


def test_batch_predictor_requires_preflight_before_active_loading(tmp_path):
    import json

    from vllm.v1.kv_offload.scheduling_deferral import Predictor

    artifact = _batch_artifact()
    artifact.update(
        frozen=True,
        shadow_cells=4,
        shadow_coverage=1.0,
        residual_cells=4,
        preflight_cells=4,
    )
    path = tmp_path / "predictor.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="preflight"):
        Predictor.load(str(path), active=True)
    artifact["preflight"] = dict(
        grid_coverage=1.0, coverage=0.95, collection_passed=True
    )
    path.write_text(json.dumps(artifact))
    Predictor.load(str(path), active=True)


def test_shadow_watch_distinguishes_readiness_from_selection():
    from vllm.v1.kv_offload.step_timing import TimingChannel

    timing = TimingChannel.__new__(TimingChannel)
    timing.watches = {}
    timing.events = []
    timing.watch(1, 10, 100, {"ready": False, "selected": False})
    timing.observe(200, {"ready": True, "selected": False}, ["ready", "selected"])
    timing.observe_selection({"selected": 1024}, 300, 2)
    assert [e["reason"] for e in timing.events] == [
        "all_rank_ready",
        "selected_before_observation",
    ]
    assert timing.events[0]["lower_ns"] == 100
    assert timing.events[0]["upper_ns"] == 200
    assert not timing.watches


def test_replay_unseen_region_uses_explicit_targeted_model():
    from vllm.v1.kv_offload.scheduling_deferral import Predictor

    artifact = _batch_artifact()
    artifact["support"]["2"]["dispatch"] = False
    artifact["models"]["target_2"] = dict(artifact["models"]["shape"])
    predictor = Predictor(artifact)
    data = dict(
        shape=["prefill", 2, 2048, 30720],
        request_shapes=[[1024, 29696]] * 2,
        dispatch_ns=1_000_000,
    )
    result = predictor.estimate_completion(data, 5_000_000, [])
    assert result.model == "target_2"
    assert result.reason == "targeted_shape_fallback"
    assert result.completion_ns == 191_000_000


def test_timing_snapshot_clock_is_read_after_rank_records(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.kv_offload.step_timing import TimingChannel

    clock = [100]

    def read_rank(step):
        clock[0] = 200
        return (step, 110, 120, 130, 140, 0, 150, 1, 90)

    timing = TimingChannel.__new__(TimingChannel)
    timing.registrations = {1: (1, 100)}
    timing.ranks = [SimpleNamespace(read=read_rank)] * 8
    monkeypatch.setattr(
        "vllm.v1.kv_offload.step_timing.time.monotonic_ns", lambda: clock[0]
    )
    assert timing.read(1)[0]
    assert not timing.read(1, 100)[0]


def _execution_artifact():
    from vllm.v1.kv_offload.execution_predictor import shape_key

    return dict(
        version=4,
        feature_version=1,
        reserve_ns=77_578_048,
        execution_config="test-config",
        ranks=8,
        models=[
            dict(
                coefficients=[0.0] * 12,
                mean=[0.0] * 12,
                scale=[1.0] * 12,
                intercept=200.0,
            )
            for _ in range(8)
        ],
        support={
            shape_key([[1024, 29696]] * 2): dict(
                batches=30, cells=3, correction_ms=-10.0
            )
        },
    )


def test_execution_prediction_uses_observed_rank_starts_not_dispatch_queue():
    from vllm.v1.kv_offload.scheduling_deferral import Predictor

    predictor = Predictor(_execution_artifact())
    start = 1_000_000_000
    ranks = [[7, start, start, start + 1, start + 2, 1, 0, 0, 0] for _ in range(8)]
    ranks[-1][1:5] = [start + 10_000_000] * 4
    data = dict(
        step=7,
        shape=["prefill", 2, 2048, 30720],
        request_shapes=[[1024, 29696]] * 2,
        dispatch_ns=1,
        execution_config="test-config",
        execution_snapshot=ranks,
    )
    now = start + 20_000_000
    first = predictor.estimate_completion(data, now, [])
    assert first.completion_ns == start + 200_000_000
    data["dispatch_ns"] = start - 1
    assert (
        predictor.estimate_completion(data, now + 1, []).completion_ns
        == first.completion_ns
    )
    ranks[-1][3] = 0
    assert (
        predictor.estimate_completion(data, now, []).reason == "gpu_start_unavailable"
    )


@pytest.mark.parametrize(
    "bad",
    ["missing_rank", "uncertain", "future", "overwritten", "configuration", "shape"],
)
def test_execution_prediction_rejects_unobserved_or_unvalidated_work(bad):
    from vllm.v1.kv_offload.scheduling_deferral import Predictor

    predictor = Predictor(_execution_artifact())
    ranks = [[7, 10, 10, 11, 12, 1, 0, 0, 0] for _ in range(8)]
    data = dict(
        step=7,
        shape=["prefill", 2, 2048, 30720],
        request_shapes=[[1024, 29696]] * 2,
        dispatch_ns=1,
        execution_config="test-config",
        execution_snapshot=ranks,
    )
    if bad == "missing_rank":
        ranks.pop()
    elif bad == "uncertain":
        ranks[0][3] = ranks[0][4] = 3_000_000
    elif bad == "future":
        ranks[0][4] = 20_000_000
    elif bad == "overwritten":
        ranks[0][0] += 1
    elif bad == "configuration":
        data["execution_config"] = "different-GPU-or-model"
    else:
        data["request_shapes"] = [[1024, 10]] * 2
    assert predictor.estimate_completion(data, 10_000_000, []).completion_ns is None


def test_execution_features_distinguish_cached_context_and_mixed_shapes():
    from vllm.v1.kv_offload.execution_predictor import execution_features, shape_key

    a, b, c = [[1, 100], [1, 200]], [[1, 1000], [1, 2000]], [[1, 100], [32, 200]]
    assert execution_features(a) != execution_features(b)
    assert execution_features(a) != execution_features(c)
    assert shape_key(a) == shape_key(list(reversed(a)))
    assert shape_key(a) != shape_key(b)


@pytest.mark.parametrize("device_index", range(8))
@pytest.mark.parametrize("backend", ["python", "native"])
def test_gpu_start_marker_excludes_queued_compute_on_each_rank(
    tmp_path, device_index, backend
):
    """CUDA elapsed labels exclude work queued ahead of the observed start."""
    import json
    import os
    import time

    import torch

    from vllm.v1.kv_offload.execution_timing import (
        ExecutionObserver,
        NativeExecutionObserver,
        valid_snapshot,
    )
    from vllm.v1.kv_offload.step_timing import RankFile

    if device_index >= torch.accelerator.device_count():
        pytest.skip("GPU unavailable")
    library = os.environ.get("VLLM_DEFERRAL_NATIVE_OBSERVER")
    if backend == "native" and not library:
        pytest.skip("native diagnostic library not configured")
    previous_device = torch.accelerator.current_device_index()
    torch.accelerator.set_device_index(device_index)
    path = str(tmp_path / "execution")
    owner = RankFile(path, create=True)
    device = torch.device(f"cuda:{device_index}")
    observer = (
        NativeExecutionObserver(path, device, library)
        if backend == "native"
        else ExecutionObserver(path, device)
    )
    try:
        stream = torch.cuda.Stream()
        queued_begin = torch.cuda.Event(enable_timing=True)
        queued_end = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            queued_begin.record()
            torch.cuda._sleep(150_000_000)
            queued_end.record()
            before = time.monotonic_ns()
            observer.start(7)
            returned = time.monotonic_ns()
            assert not queued_end.query(), (
                "start observation blocked compute submission"
            )
            torch.cuda._sleep(50_000_000)
            enqueued = time.monotonic_ns()
            end.record()
            observer.finish(7, end, enqueued, stream=stream)
        timeout = time.monotonic() + 5
        while time.monotonic() < timeout:
            r = owner.read(7)
            if r and r[8]:
                break
            time.sleep(0.001)
        else:
            pytest.fail("GPU start/completion was not published")
        now = time.monotonic_ns()
        result = dict(
            rank=device_index,
            record=r,
            start_call_ns=returned - before,
            queued_ms=queued_begin.elapsed_time(queued_end),
            execution_reference_ms=queued_end.elapsed_time(end),
        )
        (tmp_path / "result.json").write_text(json.dumps(result, indent=2))
        assert valid_snapshot([r], 7, now, completed=True)
        assert r[2] > returned
        assert abs(r[6] / 1e6 - result["execution_reference_ms"]) < 1
        assert result["queued_ms"] > 2 * r[6] / 1e6
    finally:
        observer.close()
        owner.close()
        torch.accelerator.set_device_index(previous_device)


def test_execution_channel_failure_leaves_completion_and_kv_checks_intact(tmp_path):
    from vllm.v1.kv_offload.step_timing import RankFile, TimingChannel

    timing = TimingChannel.__new__(TimingChannel)
    timing.ranks = [RankFile(str(tmp_path / "normal"), create=True)]
    timing.execution_ranks = [MagicMock()]
    timing.execution_ranks[0].read.side_effect = OSError("start channel failed")
    timing.registrations = {7: (7, 1)}
    try:
        timing.ranks[0].write([7, 10, 20, 30, 40, 50, 50, 1, 0])
        assert timing.read_execution(7) == []
        assert timing.read(7, 100)[:2] == (True, True)
    finally:
        timing.ranks[0].close()


def test_execution_active_mode_requires_independent_validation(tmp_path):
    import json

    from vllm.v1.kv_offload.scheduling_deferral import Predictor

    artifact = dict(
        _execution_artifact(), frozen=True, shadow_coverage=1.0, shadow_cells=4
    )
    path = tmp_path / "predictor.json"
    path.write_text(json.dumps(artifact))
    Predictor.load(str(path), active=False)
    with pytest.raises(ValueError, match="independent validation"):
        Predictor.load(str(path), active=True)


def test_timing_audit_distinguishes_contention_from_stale_and_failed(tmp_path):
    """Missing snapshots must not hide the specific cause or authorize waiting."""
    import fcntl
    import time

    from vllm.v1.kv_offload.shadow_timing_audit import sample
    from vllm.v1.kv_offload.step_timing import RankFile

    path = str(tmp_path / "timing")
    writer = RankFile(path, create=True)
    reader = RankFile(path)
    execution = RankFile(str(tmp_path / "execution"), create=True)
    try:
        now = time.monotonic_ns()
        writer.write([1, now, now, now, now, 0, now, 1, now])
        execution.write([1, now, now, now, now, 1, 0, 0, 0])
        fcntl.flock(writer.fd, fcntl.LOCK_EX)
        try:
            row = sample(1, now, [reader], [execution])
            assert row["primary_reasons"] == ["lock_busy"]
            assert not row["valid"]
            assert reader.read(1) is None
        finally:
            fcntl.flock(writer.fd, fcntl.LOCK_UN)
        past = now - 20_000_000
        writer.write([1, past, past, past, past, 0, past, 1, past])
        row = sample(1, past, [reader], [execution])
        assert row["primary_reasons"] == ["observer_stale"]
        assert not row["valid"]
        writer.write([1, now, 0, 0, 0, 0, now, 0, now])
        row = sample(1, now, [reader], [execution])
        assert row["primary_reasons"] == ["worker_failed"]
        assert not row["valid"]
    finally:
        writer.close()
        reader.close()
        execution.close()


def test_passive_shadow_audit_is_bounded_and_closes(tmp_path):
    import time

    from vllm.v1.kv_offload.shadow_timing_audit import ShadowTimingAudit
    from vllm.v1.kv_offload.step_timing import RankFile

    path = str(tmp_path / "timing")
    primary = RankFile(path, create=True)
    execution = RankFile(path + ".execution", create=True)
    now = time.monotonic_ns()
    primary.write([1, now, now, now, now, now, now, 1, now])
    execution.write([1, now, now, now, now, 1, 1, now, now])
    audit = ShadowTimingAudit([path])
    try:
        audit.begin(1, now, dict(step=1, dispatch_ns=now), {"a": False})
        deadline = time.monotonic() + 2
        rows = []
        while time.monotonic() < deadline:
            rows.extend(audit.drain())
            if any(r["event"] == "timing_audit_sample" for r in rows):
                break
            time.sleep(0.005)
        samples = [r for r in rows if r["event"] == "timing_audit_sample"]
        assert len(samples) == 1
        assert samples[0]["complete"]
    finally:
        audit.close()
        primary.close()
        execution.close()
    assert not audit.process.is_alive()


def test_shadow_audit_flush_preserves_samples_with_stalled_consumer(tmp_path):
    """A backlog larger than the old per-sample queue must drain without loss."""
    import time

    from vllm.v1.kv_offload.shadow_timing_audit import ShadowTimingAudit
    from vllm.v1.kv_offload.step_timing import RankFile

    path = str(tmp_path / "timing")
    primary = RankFile(path, create=True)
    execution = RankFile(path + ".execution", create=True)
    now = time.monotonic_ns()
    primary.write([1, now, now, now, now, now, now, 1, now])
    execution.write([1, now, now, now, now, 1, 1, now, now])
    audit = ShadowTimingAudit([path])
    try:
        # Bound command backpressure while deliberately not consuming results.
        for i in range(6000):
            audit.commands.put(
                (i, dict(step=1, dispatch=now, end=now, enqueue_ns=now)), timeout=2
            )
        rows = audit.flush()
        assert not any(r["event"] == "timing_audit_failure" for r in rows)
        samples = [r for r in rows if r["event"] == "timing_audit_sample"]
        assert len(samples) == 6000
        assert {r["opportunity_id"] for r in samples} == set(range(6000))
        assert all(r["complete"] for r in samples)
        assert any(r["event"] == "timing_audit_flushed" for r in rows)
    finally:
        audit.close()
        primary.close()
        execution.close()


def test_shadow_audit_flush_timeout_is_reported_as_failure(monkeypatch):
    import threading
    from types import SimpleNamespace

    from vllm.v1.kv_offload import shadow_timing_audit as module

    audit = module.ShadowTimingAudit.__new__(module.ShadowTimingAudit)
    audit.drain = MagicMock(return_value=[])
    audit.stopped = threading.Event()
    audit.flush_token = 0
    audit.commands = MagicMock()
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(monotonic=iter([0, 6]).__next__, monotonic_ns=lambda: 100),
    )
    (failure,) = audit.flush()
    assert failure["event"] == "timing_audit_failure"
    assert failure["error"] == "audit flush acknowledgement timeout"
    assert audit.stopped.is_set()


@pytest.mark.parametrize("state", ["missing", "stale", "pending", "invalid"])
def test_timing_audit_missing_collected_step_does_not_break_readiness_watch(
    state,
):
    from vllm.v1.kv_offload.step_timing import CAPACITY, TimingChannel

    timing = TimingChannel.__new__(TimingChannel)
    timing.steps = {}
    timing.registrations = (
        {} if state == "missing" else {1: (1 + CAPACITY if state == "stale" else 1, 10)}
    )
    timing.read = MagicMock(return_value=(state != "invalid", False, []))
    timing.events = []
    timing.watches = {}
    timing.audit = MagicMock()
    timing.watch(1, 2, 100, {"request": False})
    timing.audit.begin.assert_not_called()
    assert timing.events[0]["event"] == "timing_audit_unavailable"
    assert timing.watches[2]["pending"] == {"request": 100}


@pytest.mark.parametrize("failure", [False, True])
def test_two_phase_latches_starts_but_preserves_primary_health(monkeypatch, failure):
    core, timing, clock, outstanding = _two_phase_core(monkeypatch)
    timing.read_execution.return_value = [[7, 1, 1, 2, 2, 1, 0, 0, 0]] * 8
    snapshot = dict(a=False, b=False)
    assert core._defer_two_phase_scheduling(snapshot, outstanding, 1, True)
    deadline = timing.hold.deadline_ns
    timing.read_execution_with_status.side_effect = AssertionError("start reread")
    timing.read.return_value = not failure, False, []
    clock[0] += 1_000_000
    assert core._defer_two_phase_scheduling(snapshot, outstanding, 1, True) != failure
    if failure:
        assert timing.emit.call_args.kwargs["reason"] == "invalid_timing"
    else:
        assert timing.hold.deadline_ns == deadline
    core.scheduler.schedule.assert_not_called()


def test_two_phase_busy_start_retries_only_until_original_timeout(monkeypatch):
    core, timing, clock, outstanding = _two_phase_core(monkeypatch)
    timing.read_execution_with_status.side_effect = None
    timing.read_execution_with_status.return_value = [None] * 8, ["lock_busy"] * 8
    snapshot = dict(a=False, b=False)
    assert core._defer_two_phase_scheduling(snapshot, outstanding, 1, True)
    deadline = timing.hold.deadline_ns
    clock[0] += 1_000_000
    assert core._defer_two_phase_scheduling(snapshot, outstanding, 1, True)
    assert timing.hold.deadline_ns == deadline
    clock[0] = deadline
    assert not core._defer_two_phase_scheduling(snapshot, outstanding, 1, True)
    assert timing.emit.call_args.kwargs["reason"] == "start_timeout"


@pytest.mark.parametrize("status", ["step_mismatch", "read_error", "ok"])
def test_busy_start_does_not_mask_missing_or_failed_other_rank(status):
    from vllm.v1.kv_offload.execution_timing import start_read_state

    assert start_read_state([None, None], ["lock_busy", status], 7, 100, 2) == "invalid"


def test_physical_completion_is_logged_after_ordinary_admission():
    core = EngineCore.__new__(EngineCore)
    core.early_load_readiness = MagicMock()
    core.early_load_readiness.drain.return_value = [
        early.EarlyLoadCompletion(11, 107, tuple(range(100, 108)))
    ]
    core.scheduler = MagicMock()
    core.scheduler.apply_early_kv_load_completion.return_value = None
    core._timing_enabled = lambda: True
    core.deferral_timing = MagicMock()
    assert core._refresh_early_load_readiness() == []
    core.deferral_timing.emit.assert_called_once_with(
        "kv_all_rank_completion",
        job_id=11,
        completion_ns=107,
        rank_completion_ns=tuple(range(100, 108)),
        early_promoted_request=None,
    )


@pytest.mark.parametrize(
    "failure", ["duplicate", "slot_collision", "unknown_finish", "close_pending"]
)
def test_native_execution_failure_invalidates_pending_timing(tmp_path, failure):
    """Observer errors must invalidate evidence rather than leave a usable start."""
    import os

    import torch

    from vllm.v1.kv_offload.execution_timing import NativeExecutionObserver
    from vllm.v1.kv_offload.step_timing import RankFile

    library = os.environ.get("VLLM_DEFERRAL_NATIVE_OBSERVER")
    if not library or not torch.cuda.is_available():
        pytest.skip("native diagnostic library and GPU required")
    path = str(tmp_path / "execution")
    owner = RankFile(path, create=True)
    observer = NativeExecutionObserver(path, torch.device("cuda:0"), library)
    try:
        observer.start(1)
        if failure in ("duplicate", "slot_collision"):
            with pytest.raises(RuntimeError):
                observer.start(1 if failure == "duplicate" else 4097)
        elif failure == "unknown_finish":
            with pytest.raises(RuntimeError):
                observer.finish(2, None, 0, stream=torch.cuda.current_stream(0))
        else:
            observer.close()
            with pytest.raises(ValueError):
                observer.start(2)
        assert owner.read(1)[5] == 0
    finally:
        observer.close()
        owner.close()


def test_native_initialization_failure_preserves_primary_timing(tmp_path, monkeypatch):
    """A missing library is failed start evidence, not healthy queued work."""
    import torch

    from vllm.v1.kv_offload.step_timing import RankFile, WorkerTiming

    path = str(tmp_path / "timing")
    primary = RankFile(path, create=True)
    execution = RankFile(path + ".execution", create=True)
    monkeypatch.setenv("VLLM_DEFERRAL_NATIVE_OBSERVER", str(tmp_path / "missing.so"))
    monkeypatch.setattr("vllm.v1.kv_offload.step_timing.threading.Thread", MagicMock())
    timing = WorkerTiming(path, torch.device("cuda:0"))
    try:
        timing.begin(1)
        timing.prepared()
        assert primary.read(1)[7] == 1
        assert execution.read(1)[5] == 0
    finally:
        timing.execution.close()
        timing.rank.close()
        primary.close()
        execution.close()


def test_native_short_batch_publishes_consistent_completion_bounds(tmp_path):
    """Both events can complete between polls after a long queued predecessor."""
    import os
    import time

    import torch

    from vllm.v1.kv_offload.execution_timing import (
        NativeExecutionObserver,
        valid_snapshot,
    )
    from vllm.v1.kv_offload.step_timing import RankFile

    library = os.environ.get("VLLM_DEFERRAL_NATIVE_OBSERVER")
    if not library or not torch.cuda.is_available():
        pytest.skip("native diagnostic library and GPU required")
    path = str(tmp_path / "execution")
    owner = RankFile(path, create=True)
    observer = NativeExecutionObserver(path, torch.device("cuda:0"), library)
    try:
        stream = torch.cuda.Stream(device=0)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(30_000_000)
            observer.start(1)
            torch.cuda._sleep(100_000)
            observer.finish(1, None, 0, stream=stream)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            record = owner.read(1)
            if record and record[8]:
                break
            time.sleep(0.0005)
        else:
            pytest.fail("short batch completion was not published")
        assert valid_snapshot([record], 1, time.monotonic_ns(), completed=True)
    finally:
        observer.close()
        owner.close()


def test_native_reused_event_does_not_publish_previous_completion(tmp_path):
    """A recycled end event is not current completion until finish records it."""
    import os
    import time

    import torch

    from vllm.v1.kv_offload.execution_timing import NativeExecutionObserver
    from vllm.v1.kv_offload.step_timing import RankFile

    library = os.environ.get("VLLM_DEFERRAL_NATIVE_OBSERVER")
    if not library or not torch.cuda.is_available():
        pytest.skip("native diagnostic library and GPU required")
    owner = RankFile(str(tmp_path / "execution"), create=True)
    observer = NativeExecutionObserver(
        str(tmp_path / "execution"), torch.device("cuda:0"), library
    )
    try:
        for step in range(1, 5):
            observer.start(step)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                record = owner.read(step)
                if record and record[3]:
                    break
                time.sleep(0.0005)
            else:
                pytest.fail("start was not published")
            time.sleep(0.01)
            record = owner.read(step)
            assert record[5] == 1 and record[6] == record[8] == 0
            observer.finish(step, None, 0, stream=torch.cuda.current_stream(0))
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                record = owner.read(step)
                if record and record[8]:
                    break
                time.sleep(0.0005)
            else:
                pytest.fail("current completion was not published")
            assert record[5] == 1
    finally:
        observer.close()
        owner.close()


@pytest.mark.parametrize(
    "fault", ["none", "uncached", "generation", "stale", "failed", "mixed"]
)
def test_primary_busy_retry_requires_fresh_same_generation_evidence(fault):
    from vllm.v1.kv_offload.step_timing import primary_read_state

    record = [7, 101, 102, 103, 104, 0, 105, 1, 100]
    previous = [record.copy(), record.copy()]
    records, reasons = [None, record.copy()], ["lock_busy", "ok"]
    now = 106
    if fault == "uncached":
        previous = []
    elif fault == "generation":
        previous[0][0] = 8
    elif fault == "stale":
        now += 10_000_000
    elif fault == "failed":
        previous[0][7] = 0
    elif fault == "mixed":
        reasons[1] = "read_error"
    assert primary_read_state(records, reasons, previous, 7, 100, now) == (
        "retry" if fault == "none" else "invalid"
    )


@pytest.mark.parametrize("phase", ["await_start", "execution"])
def test_primary_retry_budget_is_cumulative_and_never_extends_deadline(
    monkeypatch, phase
):
    core, timing, clock, outstanding = _two_phase_core(monkeypatch)
    core.early_load_readiness.experiment_options["deferral_primary_retry_ms"] = 2
    timing.read_for_deferral.return_value = "ready", False, [[7]], ["ok"]
    if phase == "execution":
        timing.read_execution.return_value = [[7, 1, 1, 2, 2, 1, 0, 0, 0]] * 8
    snapshot = dict(a=False, b=False)
    assert core._defer_two_phase_scheduling(snapshot, outstanding, 1, True)
    deadline = timing.hold.deadline_ns
    for state, elapsed in [("retry", 1), ("ready", 1), ("retry", 1)]:
        clock[0] += elapsed * 1_000_000
        timing.read_for_deferral.return_value = state, False, [[7]], ["lock_busy"]
        assert core._defer_two_phase_scheduling(snapshot, outstanding, 1, True)
        assert timing.hold.deadline_ns == deadline
    clock[0] += 1_000_000
    assert not core._defer_two_phase_scheduling(snapshot, outstanding, 1, True)
    assert timing.emit.call_args.kwargs["reason"] == "invalid_timing"
    core.scheduler.schedule.assert_not_called()


@pytest.mark.parametrize(
    "exit", ["ready", "cancelled", "capacity", "deadline", "completed"]
)
def test_primary_retry_preserves_safety_exits(monkeypatch, exit):
    core, timing, clock, outstanding = _two_phase_core(monkeypatch)
    core.early_load_readiness.experiment_options["deferral_primary_retry_ms"] = 2
    timing.read_for_deferral.return_value = "ready", False, [[7]], ["ok"]
    snapshot = dict(a=False, b=False)
    assert core._defer_two_phase_scheduling(snapshot, outstanding, 1, True)
    timing.read_for_deferral.return_value = "retry", False, [None], ["lock_busy"]
    clock[0] += 1_000_000
    if exit == "ready":
        snapshot = dict(a=True, b=True)
    if exit == "cancelled":
        snapshot = {}
    if exit == "deadline":
        clock[0] = timing.hold.deadline_ns
    if exit == "completed":
        outstanding = []
    assert not core._defer_two_phase_scheduling(
        snapshot, outstanding, 1, exit != "capacity"
    )
    core.scheduler.schedule.assert_not_called()


def test_execution_fallback_keeps_independent_models_and_residual_corrections():
    from vllm.v1.kv_offload.execution_predictor import shape_key
    from vllm.v1.kv_offload.scheduling_deferral import Predictor

    artifact = _execution_artifact()
    artifact["fallback_models"] = [dict(m, intercept=400) for m in artifact["models"]]
    artifact["support"][shape_key([[1024, 29696]] * 17)] = dict(
        batches=30, cells=3, correction_ms=-20, model="fallback"
    )
    predictor = Predictor(artifact)
    for n, expected in [(2, 190_000_010), (17, 380_000_010)]:
        data = dict(
            step=7,
            execution_config="test-config",
            shape=["prefill", n, 1024 * n, 30720],
            request_shapes=[[1024, 29696]] * n,
            execution_snapshot=[[7, 10, 10, 11, 12, 1, 0, 0, 0]] * 8,
        )
        assert predictor.estimate_completion(data, 20, []).completion_ns == expected
    artifact.pop("fallback_models")
    with pytest.raises(ValueError, match="support model"):
        Predictor(artifact)


@pytest.mark.parametrize("enabled", [False, True])
def test_request_visibility_audit_is_opt_in_and_precedes_scheduler(
    monkeypatch, enabled
):
    from vllm.v1.kv_offload import timeline

    calls = []
    monkeypatch.setattr(timeline, "LOAD_AUDIT", enabled)
    monkeypatch.setattr(timeline, "enabled", lambda: True)
    monkeypatch.setattr(timeline, "emit", lambda event, **kw: calls.append(event))
    core = EngineCore.__new__(EngineCore)
    core.scheduler = SimpleNamespace(add_request=lambda req: calls.append("scheduler"))
    request = SimpleNamespace(
        request_id="req",
        pooling_params=None,
        kv_transfer_params=None,
        ec_transfer_params=None,
        abort_immediately=False,
    )
    core.add_request(request)
    assert calls == (
        ["engine_request_visible", "scheduler"] if enabled else ["scheduler"]
    )


@pytest.mark.parametrize(
    "horizon,limit",
    [("0", "100000"), ("1001", "100000"), ("241", "0"), ("241", "1000001")],
)
def test_invalid_shadow_audit_bounds_fail_before_process_spawn(
    monkeypatch, horizon, limit
):
    from vllm.v1.kv_offload.shadow_timing_audit import ShadowTimingAudit

    monkeypatch.setenv("VLLM_DEFERRAL_AUDIT_HORIZON_MS", horizon)
    monkeypatch.setenv("VLLM_DEFERRAL_AUDIT_SAMPLE_LIMIT", limit)
    with pytest.raises(ValueError, match="bounded"):
        ShadowTimingAudit([])


# The hold consumes readiness/timing observations and returns a defer decision.
# Unit tests protect bounded waiting without invoking scheduler allocation.
def _fill_core(monkeypatch, policy="fill_optional"):
    core, timing, clock, outstanding = _two_phase_core(monkeypatch)
    core.early_load_readiness.experiment_options.update(
        deferral_policy=policy, deferral_primary_retry_ms=2
    )
    timing.execution_ranks = [None] * 8
    timing.read_for_deferral.return_value = "ready", False, [], ["ok"] * 8
    timing.predictor.estimate_completion.return_value.model = "overlap"
    core.scheduler.deferral_capacity_snapshot.return_value = {"request_limit": 64}
    return core, timing, clock, outstanding


def test_fill_target_before_start_counts_initial_readiness_and_freezes_pool(
    monkeypatch,
):
    core, timing, clock, outstanding = _fill_core(monkeypatch)
    snapshot = {str(i): i < 5 for i in range(20)}
    assert core._defer_fill_scheduling(snapshot, outstanding, 1, True)
    hold = timing.hold
    assert hold.target == 16
    snapshot.update({f"arrival{i}": True for i in range(40)})
    clock[0] += 50_000_000
    assert core._defer_fill_scheduling(snapshot, outstanding, 1, True)
    assert hold.deadline_ns == hold.total_deadline_ns == 220_000_000
    snapshot.update({str(i): True for i in range(16)})
    assert not core._defer_fill_scheduling(snapshot, outstanding, 1, True)
    assert timing.emit.call_args.kwargs["reason"] == "readiness_target"
    assert len(timing.emit.call_args.kwargs["ready_gain"]) == 11
    timing.predictor.estimate_completion.assert_not_called()
    core.scheduler.schedule.assert_not_called()


@pytest.mark.parametrize("policy", ["fill", "fill_optional"])
def test_fill_pending_until_total_cap(monkeypatch, policy):
    core, timing, clock, outstanding = _fill_core(monkeypatch, policy)
    assert core._defer_fill_scheduling(dict(a=False), outstanding, 1, True)
    clock[0] += 119_000_000
    assert core._defer_fill_scheduling(dict(a=False), outstanding, 1, True)
    clock[0] += 1_000_000
    assert not core._defer_fill_scheduling(dict(a=False), outstanding, 1, True)
    assert timing.emit.call_args.kwargs["reason"] == "deadline"


@pytest.mark.parametrize("wait_ms", [10, 60])
@pytest.mark.parametrize("expired", [False, True])
def test_fill_prediction_tightens_once_even_after_legacy_start_bound(
    monkeypatch, wait_ms, expired
):
    core, timing, clock, outstanding = _fill_core(monkeypatch)
    snapshot = dict(a=False, b=False)
    assert core._defer_fill_scheduling(snapshot, outstanding, 1, True)
    clock[0] += wait_ms * 1_000_000
    timing.read_execution.return_value = [[7, 1, 1, 2, 2, 1, 0, 0, 0]] * 8
    timing.predictor.reserve_ns = 80_618_000
    timing.predictor.estimate_completion.return_value.completion_ns = (
        clock[0] if expired else 300_000_000
    )
    assert core._defer_fill_scheduling(snapshot, outstanding, 1, True) != expired
    if expired:
        assert timing.emit.call_args.kwargs["reason"] == "nonpositive_execution_budget"
    else:
        assert timing.hold.deadline_ns == 209_382_000
        timing.predictor.estimate_completion.return_value.completion_ns = 999_000_000
        assert core._defer_fill_scheduling(snapshot, outstanding, 1, True)
        assert timing.hold.deadline_ns == 209_382_000
    timing.predictor.estimate_completion.assert_called_once()


def test_fill_unsupported_shape_keeps_waiting(monkeypatch):
    core, timing, clock, outstanding = _fill_core(monkeypatch)
    timing.read_execution.return_value = [[7, 1, 1, 2, 2, 1, 0, 0, 0]] * 8
    estimate = timing.predictor.estimate_completion.return_value
    estimate.completion_ns = None
    estimate.reason = "unvalidated_execution_shape"
    assert core._defer_fill_scheduling(dict(a=False), outstanding, 1, True)
    clock[0] += 60_000_000
    assert core._defer_fill_scheduling(dict(a=False), outstanding, 1, True)
    assert timing.hold.deadline_ns == 220_000_000
    timing.predictor.estimate_completion.assert_called_once()


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "corrupt",
        "missing",
        "topology",
        "cancel",
        "capacity",
        "primary",
        "busy",
    ],
)
def test_fill_exits_safely(monkeypatch, case):
    core, timing, clock, outstanding = _fill_core(monkeypatch)
    snapshot = dict(a=False)
    assert core._defer_fill_scheduling(snapshot, outstanding, 1, True)
    expected = {
        "complete": "predecessor_completed",
        "corrupt": "invalid_timing",
        "missing": "missing_predecessor",
        "topology": "outstanding_compute_changed",
        "cancel": "candidates_cancelled",
        "capacity": "capacity_saturated",
        "primary": "invalid_timing",
        "busy": "invalid_timing",
    }[case]
    if case == "complete":
        timing.read_for_deferral.return_value = "ready", True, [], []
    if case == "corrupt":
        timing.read_execution.return_value = [None] * 8
    if case == "missing":
        timing.steps.clear()
    if case == "cancel":
        snapshot = {}
    if case == "primary":
        timing.read_for_deferral.return_value = "invalid", False, [], []
    if case == "busy":
        timing.read_execution_with_status.side_effect = None
        timing.read_execution_with_status.return_value = [None] * 8, ["lock_busy"] * 8
        assert core._defer_fill_scheduling(snapshot, outstanding, 1, True)
        clock[0] += 2_000_000
    assert not core._defer_fill_scheduling(
        snapshot, outstanding, 2 if case == "topology" else 1, case != "capacity"
    )
    assert timing.emit.call_args.kwargs["reason"] == expected


@pytest.mark.parametrize("case", ["idle", "two", "zero", "ready", "capacity"])
def test_fill_entry_requires_eligible_work(monkeypatch, case):
    core, timing, clock, outstanding = _fill_core(monkeypatch)
    if case == "zero":
        core.early_load_readiness.experiment_options["deferral_hold_ms"] = 0
    assert not core._defer_fill_scheduling(
        dict(a=case == "ready"),
        [] if case == "idle" else outstanding,
        2 if case == "two" else 1,
        case != "capacity",
    )
    assert timing.hold is None


def test_fill_cancellation_reduces_frozen_target(monkeypatch):
    core, timing, clock, outstanding = _fill_core(monkeypatch)
    assert core._defer_fill_scheduling(dict(a=False, b=False), outstanding, 1, True)
    assert not core._defer_fill_scheduling(dict(a=True), outstanding, 1, True)
    assert timing.emit.call_args.kwargs["reason"] == "readiness_target"


def test_dispatch_load_ledger_exact_tokens_and_reclamation():
    from vllm.v1.kv_offload.scheduling_deferral import OutstandingLoads

    ledger = OutstandingLoads()
    ledger.dispatch(
        {
            1: SimpleNamespace(external_tokens=123),
            2: SimpleNamespace(external_tokens=789),
        }
    )
    assert ledger.snapshot() == [2, 912]
    ledger.complete(1)
    ledger.complete(1)
    assert ledger.snapshot() == [1, 789]
    ledger.dispatch({2: SimpleNamespace(external_tokens=789)})
    assert ledger.snapshot() is None
    ledger.reset()
    assert ledger.snapshot() == [0, 0]
    ledger.dispatch({i: SimpleNamespace(external_tokens=1) for i in range(4097)})
    assert ledger.snapshot() is None
    assert len(ledger.jobs) == 4096
    ledger.reset()
    ledger.dispatch({1: SimpleNamespace(external_tokens=None)})
    assert ledger.snapshot() is None


def test_forecast_load_age_uses_received_completions_and_resets(monkeypatch):
    """Future receipts cannot change a forecast snapshot already returned."""
    from vllm.v1.kv_offload import scheduling_deferral as module

    now = [1_000_000]
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: now[0])
    ledger = module.OutstandingLoads()
    ledger.dispatch(
        {
            1: SimpleNamespace(external_tokens=100),
            2: SimpleNamespace(external_tokens=200),
        }
    )
    now[0] = 3_000_000
    ledger.dispatch({3: SimpleNamespace(external_tokens=300)})
    now[0] = 4_000_000
    ledger.complete(1)
    ledger.complete(1)
    data = {"dispatch_ns": 2_000_000}
    snapshot = ledger.forecast_features(data, 5_000_000)
    assert snapshot == [2, 500, 2.0, 4.0, 2.8, 1, 200, 1, 300, 100]
    now[0] = 6_000_000
    ledger.complete(2)
    assert snapshot == [2, 500, 2.0, 4.0, 2.8, 1, 200, 1, 300, 100]
    assert len(ledger.completed) == 2
    ledger.reset()
    assert ledger.forecast_features(data, 7_000_000) == [0] * 10


def test_fill_exits_on_execution_completion_before_primary_channel(monkeypatch):
    core, timing, clock, outstanding = _fill_core(monkeypatch)
    assert core._defer_fill_scheduling(dict(a=False), outstanding, 1, True)
    timing.read_execution.return_value = [[7, 1, 1, 2, 4, 1, 1, 3, 4]] * 8
    assert not core._defer_fill_scheduling(dict(a=False), outstanding, 1, True)
    assert timing.emit.call_args.kwargs["reason"] == "predecessor_completed"
    timing.predictor.estimate_completion.assert_not_called()


@pytest.mark.parametrize(
    "mode", ["baseline", "host", "record", "full", "full_retained"]
)
def test_timing_diagnostic_modes_require_opt_in_and_exclude_holds(monkeypatch, mode):
    """An ablation must never silently enable a batching policy."""
    options = dict(
        timing_diagnostic_mode=mode,
        deferral_policy="disabled" if mode == "baseline" else "timing",
    )
    monkeypatch.delenv("VLLM_TIMING_DIAGNOSIS", raising=False)
    with pytest.raises(ValueError, match="opt-in"):
        early.validate_experiment_options(options)
    monkeypatch.setenv("VLLM_TIMING_DIAGNOSIS", "1")
    assert early.validate_experiment_options(options)["timing_diagnostic_mode"] == mode
    for policy in ("fill", "fill_optional", "deadline", "two_phase_pilot"):
        with pytest.raises(ValueError, match="excludes holds"):
            early.validate_experiment_options(dict(options, deferral_policy=policy))


def test_timing_diagnostic_overflow_is_exported_and_cannot_be_hidden(
    monkeypatch, tmp_path
):
    """A truncated marker stream invalidates the cell instead of losing records."""
    import json

    from vllm.v1.kv_offload import timing_diagnostics as diagnostic

    monkeypatch.setenv("VLLM_DEFERRAL_TIMING_DIR", str(tmp_path))
    monkeypatch.setattr(diagnostic, "MARKER_CAPACITY", 2)
    monkeypatch.setattr(diagnostic, "_events", [])
    monkeypatch.setattr(diagnostic, "_overflow", False)
    monkeypatch.setattr(diagnostic, "_active", False)
    diagnostic.configure(True, 7)
    diagnostic.emit("one")
    diagnostic.emit("two")
    with pytest.raises(RuntimeError, match="exported"):
        diagnostic.configure(False, 8)
    with pytest.raises(RuntimeError, match="overflow"):
        diagnostic.export()
    saved = json.loads(next((tmp_path / "diagnostics").glob("*.json")).read_text())
    assert saved["overflow"] and saved["generation"] == 7
    assert [e["event"] for e in saved["events"]] == ["collector_activation", "one"]
    diagnostic.configure(False, 8)


def test_timing_diagnostic_host_path_never_touches_cuda(monkeypatch, tmp_path):
    """Host-only keeps registration/preparation bookkeeping without GPU work."""
    import torch

    from vllm.v1.kv_offload.step_timing import RankFile
    from vllm.v1.kv_offload.timing_diagnostics import DiagnosticWorkerTiming

    path = str(tmp_path / "rank")
    rank = RankFile(path, create=True)
    forbidden = MagicMock(side_effect=AssertionError("unexpected CUDA operation"))
    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    monkeypatch.setattr(torch.cuda, "Stream", forbidden)
    timing = DiagnosticWorkerTiming(path, SimpleNamespace(index=0), "host")
    timing.begin(1)
    timing.prepared()
    timing.finish(None)
    record = rank.read(1)
    assert record[1] <= record[2] <= record[3]
    assert record[5] == 0 and timing.query_count == 0
    assert timing.poller is None and timing.execution is None
    timing.rank.close()
    rank.close()


@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("failure", ["rank", "generation", "exception"])
def test_diagnostic_partial_configuration_fences_next_attempt(
    monkeypatch, failure, audit
):
    """A half-switched server cannot contribute a comparison cell."""
    monkeypatch.setenv("VLLM_TIMING_DIAGNOSIS", "1")
    monkeypatch.setenv("VLLM_LOAD_ONLY_WAIT_AUDIT", "1")
    options = (
        {"load_only_wait_audit": True}
        if audit
        else {"timing_diagnostic_mode": "baseline"}
    )
    mode = "legacy" if audit else "baseline"
    core = EngineCore.__new__(EngineCore)
    initial = dict(early.EXPERIMENT_DEFAULTS)
    core.early_load_readiness = SimpleNamespace(
        experiment_options=initial, configure=MagicMock()
    )
    core.deferral_timing = MagicMock()
    core.scheduler = MagicMock()
    core.scheduler.has_requests.return_value = False
    core.batch_queue = deque()
    core._refresh_early_load_readiness = MagicMock()
    core.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=8))
    core.model_executor = MagicMock()
    acks = [dict(rank=rank, mode=mode, generation=1) for rank in range(8)]
    if failure == "rank":
        acks[-1]["rank"] = 0
    elif failure == "generation":
        acks[-1]["generation"] = 0
    else:
        core.model_executor.collective_rpc.side_effect = RuntimeError("worker failed")
    core.model_executor.collective_rpc.return_value = acks
    with pytest.raises(RuntimeError):
        core.set_kv_load_scheduling(True, 0, options)
    core.early_load_readiness.configure.assert_not_called()
    assert core.early_load_readiness.experiment_options == initial
    core.model_executor.collective_rpc.reset_mock(side_effect=True)
    with pytest.raises(RuntimeError, match="restart"):
        core.set_kv_load_scheduling(True, 0, options)
    core.model_executor.collective_rpc.assert_not_called()


def test_controlled_replay_remaps_content_not_block_indices():
    """Repriming may assign different physical slots to identical content."""
    from vllm.v1.kv_offload.timing_replay import remap_cpu

    manager = SimpleNamespace(
        _policy=SimpleNamespace(
            blocks={
                b"a": SimpleNamespace(block_id=91, ref_cnt=0),
                b"b": SimpleNamespace(block_id=42, ref_cnt=0),
            }
        ),
        _get_num_free_blocks=lambda: 10,
        _allocate_blocks=lambda keys: [
            SimpleNamespace(block_id=100 + i) for i in range(len(keys))
        ],
    )
    core = SimpleNamespace(
        scheduler=SimpleNamespace(
            connector=SimpleNamespace(
                connector_scheduler=SimpleNamespace(manager=manager)
            )
        )
    )
    capture = dict(
        initial_cpu={0: b"a", 1: b"b"}, final_cpu={0: b"a", 1: b"b", 2: b"c"}
    )
    assert remap_cpu(capture, core) == {0: 91, 1: 42, 2: 100}
    del manager._policy.blocks[b"a"]
    with pytest.raises(ValueError, match="missing"):
        remap_cpu(capture, core)


def test_controlled_manifest_equality_ignores_aliases_but_preserves_content():
    """Remapping may alter pickle memoization, never tokens or transfer values."""
    import numpy as np

    from vllm.v1.kv_offload.timing_replay import content_digest

    shared = [1, 2]
    original = dict(tokens=[shared, shared], blocks=np.array([3, 4]))
    normalized = dict(tokens=[[1, 2], [1, 2]], blocks=np.array([3, 4]))
    assert original["tokens"][0] is original["tokens"][1]
    assert normalized["tokens"][0] is not normalized["tokens"][1]
    assert content_digest(original) == content_digest(normalized)
    normalized["tokens"][1][0] = 9
    assert content_digest(original) != content_digest(normalized)
    normalized["tokens"][1][0] = 1
    normalized["blocks"][0] = 9
    assert content_digest(original) != content_digest(normalized)


@pytest.mark.parametrize("policy", ["timing", "fill", "fill_optional"])
def test_load_only_audit_allows_holds_without_timing_ablation(monkeypatch, policy):
    options = dict(load_only_wait_audit=True, deferral_policy=policy)
    with pytest.raises(ValueError, match="development opt-in"):
        early.validate_experiment_options(options)
    monkeypatch.setenv("VLLM_LOAD_ONLY_WAIT_AUDIT", "1")
    assert early.validate_experiment_options(options)["load_only_wait_audit"]
    with pytest.raises(ValueError, match="excludes timing ablation"):
        early.validate_experiment_options(dict(options, timing_diagnostic_mode="full"))


@pytest.mark.parametrize(
    "tokens,loads,stores,kind",
    [
        (1, {}, {}, "compute"),
        (1, {1: None}, {}, "compute_transfers"),
        (0, {1: None}, {}, "load_only"),
        (0, {}, {2: None}, "store_only"),
        (0, {1: None}, {2: None}, "load_store"),
        (0, {}, {}, "empty"),
    ],
)
def test_load_only_audit_classifies_without_probing(tokens, loads, stores, kind):
    from vllm.v1.kv_offload.timing_diagnostics import batch_fields

    output = SimpleNamespace(
        total_num_scheduled_tokens=tokens,
        readiness_timeline_step=1,
        kv_connector_metadata=SimpleNamespace(load_jobs=loads, store_jobs=stores),
    )
    future = MagicMock(diagnostic_rpc=dict(rpc=1))
    assert batch_fields(output, future)["kind"] == kind
    assert not future.mock_calls


@pytest.mark.parametrize("markers", [False, True])
def test_future_audit_preserves_order_and_pending_publication(monkeypatch, markers):
    """A later result drains older responses even when KV has already published."""
    from vllm.v1.executor.multiproc_executor import FutureWrapper
    from vllm.v1.kv_offload import timing_diagnostics as diagnostic

    events, order = [], []
    monkeypatch.setattr(diagnostic, "_active", markers)
    monkeypatch.setattr(
        diagnostic, "emit", lambda event, **kw: events.append((event, kw))
    )
    queue: deque = deque()

    def response(label, value):
        order.append(label)
        return value

    older = FutureWrapper(queue, lambda: response("older_response", "compute"))
    older.diagnostic_rpc = dict(rpc=1, step=1)
    later = FutureWrapper(queue, lambda: response("load_response", "load"))
    later.diagnostic_rpc = dict(rpc=2, step=2)
    order.append("kv_publication")
    assert not older.done() and not later.done()
    assert later.result() == "load"
    assert older.result() == "compute"
    assert order == ["kv_publication", "older_response", "load_response"]
    assert not queue
    if markers:
        begins = [e for name, e in events if name == "executor_future_drain_begin"]
        assert [e["drained_rpc"]["rpc"] for e in begins] == [1, 2]
        assert all(e["requested_rpc"]["rpc"] == 2 for e in begins)
    else:
        assert not events


def test_future_audit_does_not_hide_worker_failure(monkeypatch):
    from vllm.v1.executor.multiproc_executor import FutureWrapper
    from vllm.v1.kv_offload import timing_diagnostics as diagnostic

    monkeypatch.setattr(diagnostic, "_active", False)

    def fail():
        raise RuntimeError("worker failed")

    future = FutureWrapper(deque(), fail)
    with pytest.raises(RuntimeError, match="worker failed"):
        future.result()


@pytest.mark.parametrize("markers", [False, True])
@pytest.mark.parametrize("cooperative", [False, True])
def test_load_only_full_queue_waits_on_older_compute_once(
    monkeypatch, markers, cooperative
):
    """Audit records the queue dependency without changing scheduling or aborts."""
    from contextlib import nullcontext

    from vllm.v1.executor.multiproc_executor import FutureWrapper
    from vllm.v1.kv_offload import timing_diagnostics as diagnostic

    events, order = [], []
    monkeypatch.setattr(diagnostic, "_active", markers)
    monkeypatch.setattr(
        diagnostic, "emit", lambda event, **kw: events.append((event, kw))
    )

    def output(step, tokens, jobs):
        return SimpleNamespace(
            readiness_timeline_step=step,
            total_num_scheduled_tokens=tokens,
            pending_structured_output_tokens=False,
            kv_connector_metadata=SimpleNamespace(load_jobs=jobs, store_jobs={}),
        )

    compute, load = output(1, 1, {}), output(2, 0, {3: None})
    futures: deque = deque()

    def response(label, value):
        order.append(label)
        return value

    prior = FutureWrapper(futures, lambda: response("compute_result", "result"))
    prior.diagnostic_rpc = dict(rpc=1, step=1)
    loading = FutureWrapper(futures, lambda: response("load_result", "load"))
    loading.diagnostic_rpc = dict(rpc=2, step=2)
    core = EngineCore.__new__(EngineCore)
    core.batch_queue, core.batch_queue_size = deque([(prior, compute, prior)]), 2
    core.scheduler = MagicMock()
    core.scheduler.schedule.return_value = load
    core._defer_batch_queue_scheduling = MagicMock(return_value=False)
    core._prepare_batch_queue_scheduling = MagicMock(return_value=None)
    core._timing_enabled = MagicMock(return_value=False)
    core._record_deferral_dispatch = MagicMock()
    core._should_throttle_prefills = MagicMock(return_value=False)
    core.log_error_detail = core.capture_iteration_details = lambda _: nullcontext()
    core.model_executor = MagicMock()
    core.model_executor.execute_model.return_value = loading
    core.is_ec_consumer, core.is_pooling_model = True, False
    core.early_load_readiness = None
    core._process_aborts_queue = lambda: order.append("aborts")
    core._attach_iteration_details = MagicMock()
    core.scheduler.update_from_output.side_effect = lambda *a: response("update", {})
    if cooperative:
        core.early_load_readiness = SimpleNamespace(
            experiment_options={"cooperative_response_progress": True}
        )
        core._refresh_early_load_readiness = MagicMock(return_value=["ready"])
        prior.poll_response = MagicMock(return_value=(False, None))
        core.step_with_batch_queue()
        assert len(core.batch_queue) == 2
        assert not prior.done()
        core.scheduler.update_from_output.assert_not_called()
        core._refresh_early_load_readiness.assert_called_once()
        core.step_with_batch_queue()
        core.scheduler.schedule.assert_called_once()
        assert order == ["aborts", "aborts"]
        prior.poll_response.return_value = (True, "result")
        order.clear()
    core.step_with_batch_queue()
    assert order == (
        ["aborts", "aborts", "update"]
        if cooperative
        else ["compute_result", "aborts", "update"]
    )
    assert len(core.batch_queue) == 1 and core.batch_queue[0][1] is load
    assert not loading.done()
    core.scheduler.schedule.assert_called_once()
    core.model_executor.execute_model.assert_called_once()
    if markers:
        q = next(e for name, e in events if name == "engine_queue_wait")
        assert q["reason"] == "queue_full"
        assert [e["kind"] for e in q["queue"]] == ["load_only", "compute"]


@pytest.mark.parametrize("busy", [False, True])
def test_cooperative_settings_require_opt_in_and_drain(monkeypatch, busy):
    core = EngineCore.__new__(EngineCore)
    core.early_load_readiness = MagicMock()
    core._refresh_early_load_readiness = MagicMock()
    core.scheduler = MagicMock()
    core.scheduler.has_requests.return_value = busy
    core.batch_queue = deque()
    core.model_executor = MagicMock()
    monkeypatch.delenv("VLLM_COOPERATIVE_RESPONSE_PROGRESS", raising=False)
    with pytest.raises(
        RuntimeError if busy else ValueError, match="drained" if busy else "opted-in"
    ):
        core.set_kv_load_scheduling(True, 0, {"cooperative_response_progress": True})
    core.early_load_readiness.configure.assert_not_called()
    core.model_executor.collective_rpc.assert_not_called()


@pytest.mark.parametrize("credit", [0, 1])
@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("result_tokens", [0, 1])
@pytest.mark.parametrize("zero_kind", ["empty", "load", "store", "load_store"])
@pytest.mark.parametrize("compute_stores", [False, True])
def test_zero_token_credit_retains_scheduler_decision(
    monkeypatch, credit, ready, result_tokens, zero_kind, compute_stores
):
    """Extra scheduling must keep even an empty decision and retire in order."""
    from contextlib import nullcontext

    from vllm.v1.executor.multiproc_executor import FutureWrapper
    from vllm.v1.kv_offload import queue_stages

    monkeypatch.setattr(queue_stages, "active", False)

    def output(step, tokens, loads=None, stores=None):
        return SimpleNamespace(
            readiness_timeline_step=step,
            total_num_scheduled_tokens=tokens,
            pending_structured_output_tokens=False,
            kv_connector_metadata=SimpleNamespace(
                load_jobs=loads or {}, store_jobs=stores or {}
            ),
        )

    compute = output(1, 1, stores={9: None} if compute_stores else {})
    zero = output(
        2,
        0,
        {1: None} if "load" in zero_kind else {},
        {2: None} if "store" in zero_kind else {},
    )
    extra = output(3, result_tokens)
    futures: deque = deque()
    first = FutureWrapper(futures, lambda: "first")
    first.poll_response = MagicMock(return_value=(False, None))
    second = FutureWrapper(futures, lambda: "second")
    third = FutureWrapper(futures, lambda: "third")
    core = EngineCore.__new__(EngineCore)
    core.batch_queue = deque(
        [(second, zero, second), (first, compute, first)], maxlen=3
    )
    core.batch_queue_size = 2
    core.early_load_readiness = SimpleNamespace(
        experiment_options={
            "cooperative_response_progress": True,
            "zero_token_queue_credit": credit,
        }
    )
    core._zero_token_credit_counts = dict(eligible=0, used=0, compute=0)
    core.scheduler = MagicMock()
    core.scheduler.zero_token_credit_has_ready_compute.return_value = ready
    core.scheduler.schedule.return_value = extra
    core._defer_batch_queue_scheduling = MagicMock(return_value=False)
    core._prepare_batch_queue_scheduling = MagicMock(return_value=None)
    core._timing_enabled = MagicMock(return_value=False)
    core._record_deferral_dispatch = MagicMock()
    core._should_throttle_prefills = MagicMock(return_value=False)
    core.log_error_detail = core.capture_iteration_details = lambda _: nullcontext()
    core.model_executor = MagicMock()
    core.model_executor.execute_model.return_value = third
    core.model_executor.sample_tokens.return_value = third
    core.is_ec_consumer, core.is_pooling_model = True, False
    core._process_aborts_queue = MagicMock()
    core._refresh_early_load_readiness = MagicMock(return_value=[])
    core._attach_iteration_details = MagicMock()
    expected = bool(credit and ready and zero_kind in ("empty", "load"))
    for _ in range(3):
        core.step_with_batch_queue()
    assert core.scheduler.schedule.call_count == int(expected)
    assert len(core.batch_queue) == 2 + expected
    assert core.batch_queue[-1][1] is compute
    assert core._zero_token_credit_counts["compute"] == int(expected and result_tokens)
    core.scheduler.update_from_output.assert_not_called()
    first.poll_response.return_value = (True, "first")
    core.step_with_batch_queue()
    core.scheduler.update_from_output.assert_called_once_with(compute, "first")
    assert core.batch_queue[-1][1] is zero


@pytest.mark.parametrize("tokens", [(1, 1), (0, 0, 0), (1,), ()])
def test_zero_token_credit_requires_ordinary_full_queue(tokens):
    from vllm.v1.kv_offload.queue_credit import qualifies

    queue = [
        (
            None,
            SimpleNamespace(
                total_num_scheduled_tokens=n,
                pending_structured_output_tokens=False,
                kv_connector_metadata=None,
            ),
            None,
        )
        for n in tokens
    ]
    assert not qualifies(queue, 2)


@pytest.mark.parametrize("value", [-1, 2, True, 1.0])
def test_zero_token_credit_rejects_unsupported_values(value):
    from vllm.v1.kv_offload.early_readiness import validate_experiment_options

    with pytest.raises(ValueError, match="invalid experimental option"):
        validate_experiment_options({"zero_token_queue_credit": value})


def test_zero_token_credit_requires_cooperative_no_hold():
    from vllm.v1.kv_offload.early_readiness import validate_experiment_options

    with pytest.raises(ValueError, match="without holds"):
        validate_experiment_options({"zero_token_queue_credit": 1})
    with pytest.raises(ValueError, match="without holds"):
        validate_experiment_options(
            {
                "zero_token_queue_credit": 1,
                "cooperative_response_progress": True,
                "deferral_policy": "fill",
            }
        )


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("unsupported", [False, True])
@pytest.mark.parametrize("free_blocks", [0, 1])
def test_zero_token_readiness_is_nonmutating(ready, unsupported, free_blocks):
    from vllm.v1.core.sched.interface import PauseState

    scheduler = Scheduler.__new__(Scheduler)
    request = SimpleNamespace(
        request_id="ready",
        status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        structured_output_request=object() if unsupported else None,
        num_tokens_with_spec=64,
        num_output_placeholders=0,
        mm_features=[],
        pooling_params=None,
    )
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.is_encoder_decoder = False
    scheduler.has_mamba_layers = False
    scheduler.max_model_len = 4096
    scheduler.num_lookahead_tokens = 0
    scheduler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=128))]
    )
    scheduler.lora_config = None
    scheduler.requests = {"ready": request}
    scheduler.running = []
    scheduler.finished_recving_kv_req_ids = {"ready"} if ready else set()
    scheduler.max_num_running_reqs = 32
    scheduler.max_num_scheduled_tokens = 1024
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.block_pool.get_num_free_blocks.return_value = free_blocks
    before = vars(request).copy()
    assert scheduler.zero_token_credit_has_ready_compute() == bool(
        ready and not unsupported and free_blocks
    )
    assert vars(request) == before
    scheduler.kv_cache_manager.allocate_slots.assert_not_called()


def test_sparse_stage_export_fences_generations_and_overflow(monkeypatch, tmp_path):
    import json

    from vllm.v1.kv_offload import queue_stages as stages

    monkeypatch.setenv("VLLM_DEFERRAL_TIMING_DIR", str(tmp_path))
    for name, value in [
        ("records", []),
        ("overflow", False),
        ("active", False),
        ("generation", 0),
        ("rank", -1),
        ("device_index", None),
        ("CAPACITY", 2),
    ]:
        monkeypatch.setattr(stages, name, value)
    stages.configure(True, 1, 3, worker_device=2)
    stages.emit("worker_entry", step=7)
    exported = stages.configure(False, 2, 3)
    with open(exported["export"]) as stream:
        data = json.load(stream)
    assert data["device_index"] == 2
    assert [(e["generation"], e["rank"], e["step"]) for e in data["events"]] == [
        (1, 3, 7)
    ]
    stages.emit("disabled")
    assert not stages.records
    stages.configure(True, 3, 3)
    for _ in range(3):
        stages.emit("overflow")
    assert len(stages.records) == 2
    with pytest.raises(RuntimeError, match="overflow"):
        stages.configure(False, 4, 3)
    assert stages.generation == 3


def test_zero_credit_uva_ownership_and_drained_rotation(monkeypatch):
    """Three ordered RPCs may stage metadata even with only two forwards."""
    from weakref import WeakSet

    import numpy as np
    import torch

    from vllm.v1.worker.gpu import buffer_utils as buffers

    class CpuBuffer:
        def __init__(self, size, dtype):
            self.cpu = torch.zeros(size, dtype=dtype)
            self.np = self.cpu.numpy()
            self.uva = self.cpu

    monkeypatch.setattr(buffers, "UvaBuffer", CpuBuffer)
    monkeypatch.setattr(buffers, "_UVA_POOLS", WeakSet())
    monkeypatch.setattr(buffers, "_QUEUE_CREDIT", 0)
    pool = buffers.UvaBufferPool(4, torch.int32, max_concurrency=2)
    baseline_slots = list(pool._uva_bufs)
    result = buffers.configure_queue_credit(1)
    assert result == dict(pools=1, retained_bytes=48)
    outputs = [pool.copy_to_uva(np.array([i])) for i in range(3)]
    assert [x.item() for x in outputs] == [0, 1, 2]
    assert len({x.data_ptr() for x in outputs}) == 3
    assert pool._uva_bufs[:2] == baseline_slots
    # Only after drain may the next RPC reuse the first slot.
    assert pool.copy_to_uva(np.array([3])).data_ptr() == outputs[0].data_ptr()
    late_pool = buffers.UvaBufferPool(4, torch.int32, max_concurrency=2)
    assert late_pool.max_concurrency == 3
    buffers.configure_queue_credit(0)
    assert pool.max_concurrency == late_pool.max_concurrency == 2
    assert len(pool._uva_bufs) == 3  # No allocator churn between comparison arms.
    first = pool.copy_to_uva(np.array([4]))
    second = pool.copy_to_uva(np.array([5]))
    assert first.data_ptr() != second.data_ptr()
    assert pool.copy_to_uva(np.array([6])).data_ptr() == first.data_ptr()
    for invalid in (-1, 2, True, 0.5):
        with pytest.raises(ValueError):
            buffers.configure_queue_credit(invalid)


def _admission_window():
    return AdmissionWindow(1, 9, (Candidate("a", 3, 10), Candidate("b", 4, 20)), 8, 20)


def _candidate_ready(w, slot, ranks=None):
    for rank in range(8) if ranks is None else ranks:
        w.ready(1, 9, rank, slot, w.candidates[slot].version)


def test_partial_rank_readiness_and_late_arrival_do_not_change_seal():
    w = _admission_window()
    _candidate_ready(w, 0)
    _candidate_ready(w, 1, range(7))
    w.staged(1, 9, 0)
    committed = []
    w.commit(w.propose_seal(1, 9), lambda active: committed.extend(active))
    _candidate_ready(w, 1)
    for rank in range(8):
        w.staged(1, 9, rank)
        assert w.consume(1, 9, rank).active == (w.candidates[0],)
        w.retire(1, 9, rank)
        assert w.reclaimable == (rank == 7)
    assert committed == [w.candidates[0]]
    with pytest.raises(RuntimeError, match="duplicate"):
        w.retire(1, 9, 7)


def test_cancellation_changes_proposal_but_never_rewrites_committed_membership():
    w = _admission_window()
    _candidate_ready(w, 0)
    _candidate_ready(w, 1)
    w.staged(1, 9, 0)
    old = w.propose_seal(1, 9)
    assert w.cancel(1, 9, 0)
    with pytest.raises(ValueError, match="proposal changed"):
        w.commit(old, lambda _: pytest.fail("must not mutate"))
    w.commit(w.propose_seal(1, 9), lambda _: None)
    assert w.sealed.active == (w.candidates[1],)
    assert not w.cancel(1, 9, 1)


def test_commitment_failure_poisoning_and_stale_generation():
    w = _admission_window()
    _candidate_ready(w, 0)
    w.staged(1, 9, 0)
    with pytest.raises(ValueError, match="stale"):
        w.consume(2, 9, 0)

    def fail(_):
        raise RuntimeError("allocation failure")

    with pytest.raises(RuntimeError, match="allocation failure"):
        w.commit(w.propose_seal(1, 9), fail)
    assert w.sealed is None and not w.reclaimable
    with pytest.raises(RuntimeError, match="commitment failed"):
        w.consume(1, 9, 0)


def test_empty_seal_and_token_budget_preserve_exact_membership():
    for budget in (0, 9, 10, 20, 30):
        w = AdmissionWindow(1, 9, _admission_window().candidates, 8, budget)
        _candidate_ready(w, 0)
        _candidate_ready(w, 1)
        w.staged(1, 9, 0)
        proposal = w.propose_seal(1, 9)
        assert sum(c.tokens for c in proposal.active) <= budget
        w.commit(proposal, lambda _: None)
        with pytest.raises(RuntimeError, match="already sealed"):
            w.commit(proposal, lambda _: pytest.fail("must commit once"))
        with pytest.raises(RuntimeError, match="not prepared"):
            w.consume(1, 9, 1)
        assert not w.reclaimable


def test_late_control_rejects_stale_identity_and_duplicate_publication(tmp_path):
    import struct

    from vllm.v1.kv_offload.late_admission_runtime import signal_prepared

    path = tmp_path / "control"
    path.write_bytes(struct.pack("<QQ", 9, 8) + bytes(8))
    # Worker RPC metadata carries the control path as a string.
    control = str(path)
    with pytest.raises(RuntimeError, match="identity"):
        signal_prepared(control, 8, 0, 1)
    signal_prepared(control, 9, 0, 1)
    with pytest.raises(RuntimeError, match="duplicate"):
        signal_prepared(control, 9, 0, 1)
    assert path.read_bytes()[16:] == bytes([1]) + bytes(7)


def test_late_uva_ownership_covers_two_prepare_execute_pairs(monkeypatch):
    from weakref import WeakSet

    import numpy as np
    import torch

    from vllm.v1.worker.gpu import buffer_utils as buffers

    class CpuBuffer:
        def __init__(self, size, dtype):
            self.cpu = torch.zeros(size, dtype=dtype)
            self.np = self.cpu.numpy()
            self.uva = self.cpu

    monkeypatch.setattr(buffers, "UvaBuffer", CpuBuffer)
    monkeypatch.setattr(buffers, "_UVA_POOLS", WeakSet())
    monkeypatch.setattr(buffers, "_QUEUE_CREDIT", 0)
    monkeypatch.setattr(buffers, "_LATE_ADMISSION", False)
    pool = buffers.UvaBufferPool(4, torch.int32, max_concurrency=2)
    buffers.configure_late_admission(True)
    outputs = [pool.copy_to_uva(np.array([i])) for i in range(4)]
    assert [x.item() for x in outputs] == [0, 1, 2, 3]
    assert len({x.data_ptr() for x in outputs}) == 4
    buffers.configure_late_admission(False)
    assert pool.max_concurrency == 2 and len(pool._uva_bufs) == 4


def test_late_preparation_retirement_waits_for_gpu_ownership_fence():
    from vllm.v1.kv_offload.late_admission_runtime import PreparationFence

    event = MagicMock()
    payload = {"rank": 2, "step": 9}
    output = PreparationFence(payload, event)
    event.synchronize.assert_not_called()
    assert output.get_output() is payload
    event.synchronize.assert_called_once()
    with pytest.raises(RuntimeError, match="consumed twice"):
        output.get_output()


@pytest.mark.parametrize(
    "overrides",
    [
        {"late_admission_mode": 5},
        {"late_admission_mode": True},
        {"late_admission_mode": 1},
        {"late_seal_delay_us": 500},
        {"late_seal_extension_us": 50_000},
        {"late_seal_extension_us": 100_001},
        {"late_seal_delay_us": -1},
        {"late_seal_delay_us": 300_001},
        {"late_seal_delay_us": 0.5},
        {"late_candidate_slots": 0},
        {"late_candidate_slots": 17},
        {"late_candidate_slots": True},
        {"late_candidate_slots": 8},
        {"late_seal_when_all_candidates_ready": True},
        {"late_partial_min_pending_candidates": 8},
        {"late_partial_ready_percent": 50},
        {"late_partial_ready_percent": 101},
        {
            "late_admission_mode": 4,
            "cooperative_response_progress": True,
            "late_seal_delay_us": 200_000,
            "late_partial_min_pending_candidates": 8,
            "late_partial_ready_percent": 50,
        },
        {
            "late_admission_mode": 2,
            "cooperative_response_progress": True,
            "late_seal_delay_us": 300_000,
            "late_seal_when_all_candidates_ready": True,
            "late_partial_min_pending_candidates": 8,
            "late_partial_ready_percent": 50,
        },
        {
            "late_admission_mode": 2,
            "cooperative_response_progress": True,
            "late_candidate_slots": 8,
        },
        {
            "late_admission_mode": 2,
            "cooperative_response_progress": True,
            "zero_token_queue_credit": 1,
        },
        {
            "late_admission_mode": 2,
            "cooperative_response_progress": True,
            "deferral_policy": "fill",
        },
    ],
)
def test_late_admission_rejects_unsupported_configuration(overrides):
    from vllm.v1.kv_offload.early_readiness import validate_experiment_options

    with pytest.raises(ValueError):
        validate_experiment_options(overrides)


def test_late_seal_delay_accepts_fixed_microseconds():
    from vllm.v1.kv_offload.early_readiness import validate_experiment_options

    options = validate_experiment_options(
        {
            "late_admission_mode": 2,
            "cooperative_response_progress": True,
            "late_seal_delay_us": 500,
        }
    )
    assert options["late_seal_delay_us"] == 500

    candidate_only = validate_experiment_options(
        {
            "late_admission_mode": 3,
            "cooperative_response_progress": True,
            "late_seal_delay_us": 250_000,
        }
    )
    assert candidate_only["late_seal_delay_us"] == 250_000

    notice_only = validate_experiment_options(
        {
            "late_admission_mode": 4,
            "cooperative_response_progress": True,
            "late_seal_delay_us": 250_000,
        }
    )
    assert notice_only["late_seal_delay_us"] == 250_000
    expanded_notice = validate_experiment_options(
        {
            "late_admission_mode": 4,
            "cooperative_response_progress": True,
            "late_candidate_slots": 10,
            "late_seal_delay_us": 300_000,
            "late_seal_when_all_candidates_ready": True,
            "late_partial_min_pending_candidates": 8,
            "late_partial_ready_percent": 50,
        }
    )
    assert expanded_notice["late_candidate_slots"] == 10
    assert expanded_notice["late_seal_delay_us"] == 300_000
    assert expanded_notice["late_seal_when_all_candidates_ready"]
    assert expanded_notice["late_partial_min_pending_candidates"] == 8
    assert expanded_notice["late_partial_ready_percent"] == 50

    extended = early.validate_experiment_options(
        {
            "cooperative_response_progress": True,
            "late_admission_mode": 4,
            "late_seal_delay_us": 200_000,
            "late_seal_extension_us": 50_000,
        }
    )
    assert extended["late_seal_extension_us"] == 50_000


def test_late_notice_candidate_slot_cap_still_respects_token_budget():
    from vllm.v1.kv_offload import late_admission_runtime as late

    requests = {
        str(i): SimpleNamespace(
            request_id=str(i),
            arrival_time=i,
            status=RequestStatus.WAITING_FOR_REMOTE_KVS,
            prompt_token_ids=[1],
            num_preemptions=0,
            num_output_tokens=0,
            num_computed_tokens=32,
            num_tokens=64,
            mm_features=[],
            structured_output_request=None,
            pooling_params=None,
        )
        for i in range(20)
    }
    connector = SimpleNamespace(
        _sliding_window_groups=[],
        _current_batch_load_jobs=set(),
        _req_status={
            rid: SimpleNamespace(transfer_jobs={i}) for i, rid in enumerate(requests)
        },
    )
    scheduler = SimpleNamespace(
        _late_admission_mode=4,
        _late_candidate_slots=16,
        has_mamba_layers=False,
        lora_config=None,
        is_encoder_decoder=False,
        num_lookahead_tokens=0,
        requests=requests,
        connector=SimpleNamespace(connector_scheduler=connector),
        max_num_running_reqs=64,
        running=[],
        finished_recving_kv_req_ids=set(),
        current_step=5,
        _late_cancelled=set(),
    )
    staged: list = []

    def prepare(step, base, candidates):
        staged.extend(c.req_id for c in candidates)
        scheduler._late_kv_snapshot = set(staged)
        return staged

    scheduler._late_admission_prepare = prepare
    assert late.prepare_candidates(scheduler, 512, []) == set(map(str, range(16)))
    assert staged == list(map(str, range(16)))
    staged.clear()
    assert late.prepare_candidates(scheduler, 256, []) == set(map(str, range(8)))
    assert staged == list(map(str, range(8)))


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_late_candidates_use_old_loads_spare_budget_and_authoritative_readiness(
    monkeypatch, mode
):
    from vllm.v1.kv_offload import late_admission_runtime as late

    requests = {}
    for i in range(8):
        rid = str(i)
        requests[rid] = SimpleNamespace(
            request_id=rid,
            arrival_time=i,
            status=RequestStatus.WAITING_FOR_REMOTE_KVS,
            prompt_token_ids=[1],
            num_preemptions=0,
            num_output_tokens=0,
            num_computed_tokens=32,
            num_tokens=64,
            _all_token_ids=[1],
            mm_features=[],
            structured_output_request=None,
            pooling_params=None,
        )
    connector = SimpleNamespace(
        _sliding_window_groups=[],
        _current_batch_load_jobs={0},
        _req_status={
            rid: SimpleNamespace(transfer_jobs={int(rid)}) for rid in requests
        },
    )
    scheduler = SimpleNamespace(
        _late_admission_mode=mode,
        has_mamba_layers=False,
        lora_config=None,
        is_encoder_decoder=False,
        num_lookahead_tokens=0,
        requests=requests,
        connector=SimpleNamespace(connector_scheduler=connector),
        max_num_running_reqs=32,
        running=[],
        finished_recving_kv_req_ids=set(),
        kv_cache_manager=MagicMock(),
        current_step=5,
        finished_req_ids=set(),
        reset_preempted_req_ids=set(),
        _late_cancelled={"2"},
    )
    monkeypatch.setattr(
        late.NewRequestData,
        "from_request",
        lambda r, *a: SimpleNamespace(req_id=r.request_id),
    )

    def prepare(step, base, candidates):
        assert base == []
        assert [c.req_id for c in candidates] == ["1", "2", "3"]
        scheduler.finished_recving_kv_req_ids.update({"1", "2", "4"})
        scheduler._late_kv_snapshot = {"1", "2"}
        return [c.req_id for c in candidates]

    scheduler._late_admission_prepare = prepare
    assert late.prepare_candidates(scheduler, 96, []) == (
        {"1"} if mode in (2, 3, 4) else set()
    )
    assert all(r.num_computed_tokens == 32 for r in requests.values())
    # Unsupported paths cannot cause speculative worker preparation.
    requests["0"].mm_features = [object()]
    scheduler._late_admission_prepare = MagicMock(side_effect=AssertionError)
    assert late.prepare_candidates(scheduler, 96, []) == set()


def test_late_opportunity_analysis_keeps_missing_ranks_and_avoids_future_selection():
    import importlib.util
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[3]
        / "experiments/late_admission_20260925/opportunities.py"
    )
    spec = importlib.util.spec_from_file_location("late_opportunities", source)
    assert spec is not None and spec.loader is not None
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    timing = [
        dict(event="kv_load_dispatched", request=rid, job_id=i, dispatch_ns=10)
        for i, rid in enumerate(["old", "first", "second"])
    ] + [
        dict(event="dispatch", step=0, dispatch_ns=20, time_ns=20, selected={"old": 1}),
        dict(event="dispatch", step=1, dispatch_ns=30, time_ns=30, selected={}),
        dict(
            event="dispatch",
            step=2,
            dispatch_ns=70,
            time_ns=70,
            selected={"old": 1, "second": 1},
        ),
        dict(event="kv_all_rank_completion", job_id=1, rank_completion_ns=[50] * 8),
        dict(event="kv_all_rank_completion", job_id=2, rank_completion_ns=[40] * 8),
    ]
    events = [dict(event="worker_entry", step=1, rank=r, time_ns=60) for r in range(7)]
    result = analysis.analyze(events, timing, candidate_cap=1)
    row = result["batches"][1]
    # Arrival/dispatch ordering selects first despite second becoming ready sooner;
    # old is excluded even though it also has a later decode dispatch.
    assert row["candidates"] == ["first"]
    assert row["boundaries"]["worker_entry"]["rank_coverage"] == 7
    assert result["summary"]["worker_entry"]["batches_observed"] == 0
    assert result["summary"]["worker_entry"]["batches_total"] == 3
    events.append(dict(event="worker_entry", step=1, rank=7, time_ns=60))
    result = analysis.analyze(events, timing, candidate_cap=1)
    assert result["batches"][1]["boundaries"]["worker_entry"]["became_ready"] == [
        "first"
    ]


@pytest.mark.parametrize("poll_us", [0, 5001, -1, True, 1.5])
def test_fixed_seal_poll_interval_rejects_busy_spin_and_invalid_values(poll_us):
    with pytest.raises(ValueError):
        early.validate_experiment_options({"late_seal_poll_us": poll_us})


def test_fixed_seal_poll_control_preserves_default():
    assert early.validate_experiment_options({})["late_seal_poll_us"] == 25
    assert (
        early.validate_experiment_options({"late_seal_poll_us": 1000})[
            "late_seal_poll_us"
        ]
        == 1000
    )


@pytest.mark.parametrize("poll_us", [1000, 2000, 5000, 10000])
def test_adaptive_poll_interval_preserves_pre_ack_interval(poll_us):
    options = early.validate_experiment_options({"late_seal_adaptive_poll_us": poll_us})
    assert options["late_seal_adaptive_poll_us"] == poll_us
    assert options["late_seal_poll_us"] == 25


@pytest.mark.parametrize("poll_us", [0, -1, 10001, True, 1.5])
def test_adaptive_poll_interval_is_positive_and_bounded(poll_us):
    with pytest.raises(ValueError):
        early.validate_experiment_options({"late_seal_adaptive_poll_us": poll_us})


@pytest.mark.parametrize(
    "invalid",
    [
        {"late_admission_mode": 0},
        {"late_seal_safety_us": 0},
        {"late_seal_target_candidates": 0},
        {"late_seal_repeat_us": 0},
    ],
)
def test_evaluation_event_options_require_mode_and_bounded_progress(invalid):
    options = dict(
        late_seal_policy="event",
        late_admission_mode=4,
        cooperative_response_progress=True,
        deferral_policy="timing",
    )
    assert early.validate_experiment_options(options)["late_seal_policy"] == "event"
    with pytest.raises(ValueError):
        early.validate_experiment_options(dict(options, **invalid))


@pytest.mark.parametrize("policy", ["fixed", "event"])
def test_prediction_to_readiness_requires_predictive_policy(policy):
    with pytest.raises(ValueError, match="prediction-to-readiness"):
        early.validate_experiment_options(
            {
                "late_admission_mode": 4,
                "cooperative_response_progress": True,
                "late_seal_policy": policy,
                "late_seal_prediction_to_readiness": True,
            }
        )


@pytest.mark.parametrize("bias", [-1, 300001, 1.0, True])
def test_prediction_bias_rejects_invalid_values(bias):
    with pytest.raises(ValueError, match="late_seal_prediction_bias_us"):
        early.validate_experiment_options({"late_seal_prediction_bias_us": bias})


def test_prediction_bias_requires_predictive_policy():
    with pytest.raises(ValueError, match="prediction bias requires predictive"):
        early.validate_experiment_options({"late_seal_prediction_bias_us": 90000})
    assert early.validate_experiment_options({})["late_seal_prediction_bias_us"] == 0


def test_prediction_to_readiness_is_opt_in():
    assert not early.validate_experiment_options({})[
        "late_seal_prediction_to_readiness"
    ]
    assert early.validate_experiment_options(
        {
            "late_admission_mode": 4,
            "cooperative_response_progress": True,
            "late_seal_policy": "predictive",
            "late_seal_prediction_to_readiness": True,
        }
    )["late_seal_prediction_to_readiness"]


@pytest.mark.parametrize("policy", ["event", "predictive"])
def test_load_only_percentage_requires_supported_readiness_path(policy):
    options = dict(
        late_admission_mode=4,
        cooperative_response_progress=True,
        late_seal_policy=policy,
        late_seal_prediction_to_readiness=policy == "predictive",
        late_seal_load_only_ready_percent=50,
    )
    assert (
        early.validate_experiment_options(options)["late_seal_load_only_ready_percent"]
        == 50
    )
    with pytest.raises(ValueError, match="load-only percentage"):
        early.validate_experiment_options(
            dict(
                options,
                late_seal_policy="fixed",
                late_seal_prediction_to_readiness=False,
            )
        )
    if policy == "predictive":
        with pytest.raises(ValueError, match="load-only percentage"):
            early.validate_experiment_options(
                dict(options, late_seal_prediction_to_readiness=False)
            )


@pytest.mark.parametrize("percent", [-1, 101, 50.0, True])
def test_load_only_percentage_rejects_invalid_bounds_or_types(percent):
    with pytest.raises(ValueError, match="late_seal_load_only_ready_percent"):
        early.validate_experiment_options(
            dict(
                late_admission_mode=4,
                cooperative_response_progress=True,
                late_seal_policy="event",
                late_seal_load_only_ready_percent=percent,
            )
        )


@pytest.mark.parametrize(
    "missing", ["late_seal_condition_preparation", "late_seal_prediction_to_readiness"]
)
def test_full_predicted_wait_requires_conditioned_readiness_policy(missing):
    options = dict(
        late_admission_mode=4,
        cooperative_response_progress=True,
        late_seal_policy="predictive",
        late_seal_condition_preparation=True,
        late_seal_prediction_to_readiness=True,
        late_seal_full_prediction_wait=True,
    )
    assert early.validate_experiment_options(options)["late_seal_full_prediction_wait"]
    options[missing] = False
    with pytest.raises(ValueError, match="full predicted wait"):
        early.validate_experiment_options(options)
    assert not early.validate_experiment_options({})["late_seal_full_prediction_wait"]


def test_coalesced_notifications_require_full_prediction_wait():
    assert not early.validate_experiment_options({})[
        "late_seal_coalesced_notifications"
    ]
    with pytest.raises(ValueError, match="require full predicted waits"):
        early.validate_experiment_options({"late_seal_coalesced_notifications": True})


def test_coalesced_startup_owns_rank_descriptors_and_closes_them():
    """Create real native CPU rings; allocating the channel requires no CUDA work."""
    from pathlib import Path
    from types import SimpleNamespace

    extra = {
        "early_load_completion": True,
        "async_h2d_submission": True,
        "coalesced_seal_notifications": True,
    }
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        parallel_config=SimpleNamespace(world_size=8),
    )
    readiness = EarlyLoadReadiness.from_config(config)
    assert readiness is not None
    try:
        channel = readiness.seal_notifications
        assert channel is not None
        descriptors = [Path(path + ".seal") for path in readiness.paths]
        assert all(path.exists() for path in descriptors)
        assert not hasattr(channel, "thread")
    finally:
        readiness.close()
    assert all(not path.exists() for path in descriptors)
    assert not Path(channel.path).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host,flag,status",
    [
        ("127.0.0.1", True, 200),
        ("127.0.0.1", False, 404),
        ("127.0.0.1", "true", 404),
        ("203.0.113.1", True, 403),
    ],
)
async def test_read_only_cohort_status_needs_explicit_opt_in_and_loopback(
    host, flag, status
):
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    from vllm.entrypoints.serve.experimental.kv_load_scheduling import (
        get_deferral_cohort,
    )

    rpc = AsyncMock(return_value={"requests": 0})
    raw = SimpleNamespace(
        client=SimpleNamespace(host=host),
        app=SimpleNamespace(
            state=SimpleNamespace(
                vllm_config=SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(
                        kv_connector_extra_config={"cohort_status_diagnostics": flag}
                    )
                ),
                engine_client=SimpleNamespace(get_deferral_cohort_status=rpc),
            )
        ),
    )
    if status == 200:
        assert await get_deferral_cohort(raw) == {"requests": 0}
        rpc.assert_awaited_once_with()
    else:
        with pytest.raises(HTTPException) as error:
            await get_deferral_cohort(raw)
        assert error.value.status_code == status
        rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_only_cohort_opt_in_does_not_enable_policy_mutation():
    from fastapi import HTTPException

    from vllm.entrypoints.serve.experimental.kv_load_scheduling import (
        KVLoadSchedulingRequest,
        set_kv_load_scheduling,
    )

    raw = SimpleNamespace(
        client=SimpleNamespace(host="127.0.0.1"),
        app=SimpleNamespace(
            state=SimpleNamespace(
                vllm_config=SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(
                        kv_connector_extra_config={"cohort_status_diagnostics": True}
                    )
                ),
            )
        ),
    )
    with pytest.raises(HTTPException) as error:
        await set_kv_load_scheduling(KVLoadSchedulingRequest(), raw)
    assert error.value.status_code == 404


def test_opportunity_ratio_uses_native_time_and_rejects_partial_rank_coverage():
    from experiments.load_only_overlap_20261011.opportunities import summarize

    notifications = [
        dict(
            event="early_notification_received",
            job_id=1,
            rank=rank,
            completion_ns=100,
            status=1,
        )
        for rank in range(8)
    ]
    owner = dict(event="job_observed_done", job_id=1, request="request", time_ns=500)
    transport = dict(has_compute=True)
    protocol = [
        dict(
            event="membership_sealed",
            activated={"request": 128},
            transport=transport,
        )
    ]
    compute = [dict(execution_ranks=[[1, 75, 78, 80, 250, 1, 120, 198, 200]] * 8)]
    windows = [(50, 600)]
    result = summarize([owner, *notifications], protocol, compute, windows)
    assert result["ready_during_compute_definite"] == 1
    assert result["late_admission_ratio_percent"] == 100
    result = summarize([owner, *notifications[:-1]], protocol, compute, windows)
    assert result["native_ready_coverage_complete"] is False
    assert result["late_admission_ratio_percent"] is None
    unknown_owner = dict(
        event="job_observed_done", job_id=2, request="other", time_ns=500
    )
    result = summarize(
        [owner, unknown_owner, *notifications], protocol, compute, windows
    )
    assert result["late_admission_ratio_percent"] is None
    assert result["admission_ratio_lower_bound_percent"] == 50
    assert result["admission_ratio_upper_bound_percent"] == 100
    transport["has_compute"] = False
    result = summarize([owner, *notifications], protocol, compute, windows)
    assert result["total_late_admissions"] == 1
    assert result["late_admitted_during_compute_definite"] == 0
