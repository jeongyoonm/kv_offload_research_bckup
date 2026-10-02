# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for late admission publication, membership, and retirement."""

import pickle
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext, suppress
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.v1.kv_connector.unit.offloading_connector.conftest import (
    request_runner as request_runner,
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


class ControlledExecutor:
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
    estimator = SimpleNamespace(
        predict=lambda data: 50_000_000,
        features=lambda data: np.ones(4),
        samples=[0] * 8,
        preparation_reserve=lambda extra, upcoming=None: (
            queries.append(upcoming)
            or extra + (5_000_000 if upcoming is not None else 20_000_000)
        ),
        preparation_ns=[20_000_000] * 8,
        recent_preparation_reserve=lambda extra, upcoming: (
            queries.append(upcoming) or extra + 5_000_000
        ),
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


def test_execute_b_is_enqueued_while_a_main_rpc_is_held(monkeypatch):
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

    def schedule(*_):
        assert stage_rpc(core, 9, [_request("base")], []) == []
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
    oldest = Future()
    oldest.set_result("oldest-output")
    oldest.poll = MagicMock(return_value=False)
    newer = Future()
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
    prepared_base = []

    def prepare(step, base, candidates):
        prepared_base.extend(request.req_id for request in base)
        assert [candidate.req_id for candidate in candidates] == ["0"]
        return []

    scheduler._late_admission_prepare = prepare
    output = scheduler.schedule()
    assert prepared_base == expected_base
    assert output.num_scheduled_tokens == {"1": 24}


@pytest.mark.parametrize("outcome", ["ready", "complete", "physical", "cancel"])
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
        else:
            core.aborts_queue.put(["a"])
        # stage_rpc returns prepared slots; final admission uses the KV/cancel snapshot.
        assert result.result(timeout=5) == ["a"]
        assert scheduler._late_kv_snapshot == ({"a"} if outcome == "ready" else set())
        assert scheduler._late_cancelled == ({"a"} if outcome == "cancel" else set())
    transport = scheduler._late_transport
    assert transport["seal_reason"] == (
        "predecessor_compute_finished"
        if outcome in {"complete", "physical"}
        else "candidate_target_ready"
    )
    assert transport["seal_ns"] > transport["prediction"]["deadline_ns"]
    assert transport["hybrid_coarse_sleeps"] > 0
    assert transport["hybrid_polling_ns"] is not None
    if outcome == "physical":
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


def test_prediction_to_readiness_handles_physical_completion_before_base_ack():
    executor = ControlledExecutor()
    core = _core(executor)
    core.scheduler._late_seal_policy = "predictive"
    core.scheduler._late_seal_prediction_to_readiness = True
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
