# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment-only independent completion transport for async CPU KV loads."""

import contextlib
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm import _custom_ops as ops
from vllm.v1.kv_offload import timeline

if TYPE_CHECKING:
    from vllm.config import VllmConfig

RING_CAPACITY = 4096
EARLY_COMPLETION_KEY = "early_load_completion"
SCHEDULE_DELAY_KEY = "schedule_delay_ms"
_RING_PATHS_KEY = "_early_load_completion_ring_paths"
_SESSION_KEY = "_early_load_completion_session"
EXPERIMENT_DEFAULTS: dict[str, Any] = {
    "cooperative_response_progress": False,
    "zero_token_queue_credit": 0,
    "late_admission_mode": 0,
    "late_admission_markers": False,
    "late_seal_delay_us": 0,
    "late_seal_extension_us": 0,
    "late_seal_policy": "fixed",
    "late_seal_target_candidates": 4,
    "late_seal_load_only_target_candidates": 0,
    "late_seal_repeat_us": 25_000,
    "late_seal_condition_preparation": False,
    "late_seal_poll_us": 25,
    "late_seal_prediction_reserve_us": 10_000,
    "late_seal_prediction_scale_percent": 100,
    "late_seal_safety_us": 2_000_000,
    "late_seal_when_all_candidates_ready": False,
    "late_candidate_slots": 4,
    "late_partial_min_pending_candidates": 0,
    "late_partial_ready_percent": 0,
    "zero_token_stage_markers": False,
    "load_only_wait_audit": False,
    "timing_diagnostic_mode": "legacy",
    "timing_replay_action": "none",
    "timing_diagnostic_markers": False,
    "optimistic_cap": 0,
    "optimistic_min_ranks": 7,
    "optimistic_window_ms": 20,
    "early_submission": False,
    "background_prepare": False,
    "copy_pipeline_depth": 1,
    "deferral_policy": "disabled",
    "deferral_hold_ms": 120,
    "deferral_target_gain": 2,
    "deferral_fill_fraction": 0.25,
    "deferral_margin_ms": 10,
    "deferral_start_wait_ms": 30,
    "deferral_primary_retry_ms": 0,
}


def validate_experiment_options(options: dict) -> dict[str, Any]:
    result = dict(EXPERIMENT_DEFAULTS)
    for key, value in options.items():
        if key not in result or type(value) is not type(result[key]):
            raise ValueError(f"invalid experimental option: {key}")
        result[key] = value
    for key, upper in (
        ("zero_token_queue_credit", 1),
        ("late_admission_mode", 4),
        ("late_seal_delay_us", 300_000),
        ("late_seal_extension_us", 100_000),
        ("late_seal_prediction_reserve_us", 100_000),
        ("late_seal_prediction_scale_percent", 200),
        ("late_seal_safety_us", 5_000_000),
        ("late_seal_target_candidates", 16),
        ("late_seal_load_only_target_candidates", 16),
        ("late_seal_repeat_us", 100_000),
        ("late_candidate_slots", 16),
        ("late_partial_min_pending_candidates", 16),
        ("late_partial_ready_percent", 100),
        ("optimistic_cap", 4),
        ("optimistic_min_ranks", 64),
        ("optimistic_window_ms", 120),
        ("copy_pipeline_depth", 8),
    ):
        if not 0 <= result[key] <= upper:
            raise ValueError(f"invalid experimental option: {key}")
    if not 1 <= result["late_seal_poll_us"] <= 5000:
        raise ValueError("invalid experimental option: late_seal_poll_us")
    if result["zero_token_queue_credit"] and (
        not result["cooperative_response_progress"]
        or result["deferral_policy"] not in ("disabled", "timing")
        or result["optimistic_cap"]
        or result["background_prepare"]
    ):
        raise ValueError(
            "zero-token credit requires cooperative progress without holds"
        )
    if result["late_seal_policy"] not in ("fixed", "predictive", "event") or (
        result["late_seal_policy"] == "predictive"
        and (
            result["late_admission_mode"] != 4
            or not result["late_seal_prediction_scale_percent"]
            or not result["late_seal_safety_us"]
        )
    ):
        raise ValueError("predictive seal requires mode 4 and positive timing")
    if result["late_seal_policy"] == "event" and (
        result["late_admission_mode"] != 4
        or not result["late_seal_safety_us"]
        or not result["late_seal_target_candidates"]
        or not result["late_seal_repeat_us"]
    ):
        raise ValueError("event seal requires mode 4 and positive target/timing")
    if result["late_admission_mode"] and (
        not result["cooperative_response_progress"]
        or result["zero_token_queue_credit"]
        or result["deferral_policy"] not in ("disabled", "timing")
        or result["optimistic_cap"]
        or result["background_prepare"]
    ):
        raise ValueError(
            "late admission requires cooperative no-hold, zero-credit mode"
        )
    if result["late_seal_delay_us"] and result["late_admission_mode"] not in (
        2,
        3,
        4,
    ):
        raise ValueError("late seal delay requires admission mode 2, 3, or 4")
    if result["late_seal_when_all_candidates_ready"] and (
        result["late_admission_mode"] != 4 or not result["late_seal_delay_us"]
    ):
        raise ValueError("all-candidate-ready seal requires held admission mode 4")
    if result["late_seal_extension_us"] and (
        result["late_admission_mode"] != 4 or not result["late_seal_delay_us"]
    ):
        raise ValueError("seal extension requires held admission mode 4")
    partial_min = result["late_partial_min_pending_candidates"]
    partial_percent = result["late_partial_ready_percent"]
    if bool(partial_min) != bool(partial_percent) or (
        partial_min
        and (
            result["late_admission_mode"] != 4
            or not result["late_seal_delay_us"]
            or partial_min > result["late_candidate_slots"]
        )
    ):
        raise ValueError("partial-ready seal requires held mode 4 and enough slots")
    if not result["late_candidate_slots"] or (
        result["late_candidate_slots"] != 4 and result["late_admission_mode"] != 4
    ):
        raise ValueError("expanded candidate slots require admission mode 4")
    from vllm.v1.kv_offload.scheduling_deferral import POLICIES

    if not 0 < result["deferral_fill_fraction"] <= 1:
        raise ValueError("invalid deferral fill fraction")
    if result["deferral_policy"] not in POLICIES:
        raise ValueError("invalid deferral policy")
    for key, lower, upper in (
        ("deferral_hold_ms", 0, 120),
        ("deferral_target_gain", 1, 64),
        ("deferral_margin_ms", 0, 120),
        ("deferral_start_wait_ms", 0, 120),
        ("deferral_primary_retry_ms", 0, 10),
    ):
        if not lower <= result[key] <= upper:
            raise ValueError(f"invalid experimental option: {key}")
    if result["deferral_policy"] != "disabled" and (
        result["optimistic_cap"] or result["background_prepare"]
    ):
        raise ValueError(
            "deferral excludes optimistic admission/background preparation"
        )
    from vllm.v1.kv_offload.timing_diagnostics import validate_mode

    validate_mode(result)
    if not result["copy_pipeline_depth"]:
        raise ValueError("copy pipeline depth must be positive")
    if result["optimistic_cap"] and not result["optimistic_min_ranks"]:
        raise ValueError("optimistic admission requires at least one completed rank")
    return result


@dataclass(frozen=True)
class EarlyLoadCompletion:
    job_id: int
    completion_ns: int
    rank_completion_ns: tuple[int, ...] = field(default=(), compare=False)


class EarlyLoadReadiness:
    """Own one native SPSC completion ring per model-worker rank."""

    def __init__(
        self,
        session: int,
        handles: list[int],
        paths: list[str],
        enabled: bool,
        schedule_delay_ms: float,
    ) -> None:
        self.session = session
        self.handles = handles
        self.paths = paths
        self.enabled = enabled
        self.schedule_delay_ms = schedule_delay_ms
        self._rank_reports: dict[int, set[int]] = {}
        self._rank_timestamps: dict[int, dict[int, int]] = {}
        self._completion_ns: dict[int, int] = {}
        self._released_jobs: set[int] = set()
        self._closed = False
        self.experiment_options = dict(EXPERIMENT_DEFAULTS)

    @classmethod
    def from_config(cls, vllm_config: "VllmConfig") -> "EarlyLoadReadiness | None":
        transfer_config = vllm_config.kv_transfer_config
        if transfer_config is None:
            return None
        extra = transfer_config.kv_connector_extra_config
        if not (EARLY_COMPLETION_KEY in extra or SCHEDULE_DELAY_KEY in extra):
            return None

        enabled = extra.get(EARLY_COMPLETION_KEY, False)
        if type(enabled) is not bool:
            raise ValueError("early_load_completion must be a boolean")
        delay = extra.get(SCHEDULE_DELAY_KEY, 0)
        if isinstance(delay, bool) or not isinstance(delay, (int, float)):
            raise ValueError("schedule_delay_ms must be a number from 0 to 120")
        delay = float(delay)
        if not 0 <= delay <= 120:
            raise ValueError("schedule_delay_ms must be a number from 0 to 120")
        if extra.get("async_h2d_submission") is not True:
            raise ValueError(
                "the early KV readiness experiment requires async_h2d_submission"
            )
        if extra.get("grouped_load_completion", False) is not False:
            raise ValueError(
                "grouped_load_completion must be disabled for this experiment"
            )

        world_size = vllm_config.parallel_config.world_size
        session = uuid.uuid4().int & ((1 << 63) - 1) or 1
        directory = Path("/dev/shm")
        if not directory.is_dir():
            raise RuntimeError("the early KV readiness experiment requires /dev/shm")
        prefix = f"vllm-kv-ready-{os.getpid()}-{uuid.uuid4().hex}"
        paths = [str(directory / f"{prefix}-r{rank}") for rank in range(world_size)]
        handles: list[int] = []
        try:
            for path in paths:
                handles.append(
                    ops.create_async_batch_memcpy_completion_ring(
                        path, session, RING_CAPACITY
                    )
                )
        except BaseException:
            for handle in handles:
                ops.destroy_async_batch_memcpy_completion_ring(handle)
            raise

        extra[_RING_PATHS_KEY] = paths
        extra[_SESSION_KEY] = session
        return cls(session, handles, paths, enabled, delay)

    @property
    def world_size(self) -> int:
        return len(self.handles)

    def configure(self, enabled: bool, schedule_delay_ms: float) -> None:
        if type(enabled) is not bool:
            raise ValueError("early_load_completion must be a boolean")
        if isinstance(schedule_delay_ms, bool) or not isinstance(
            schedule_delay_ms, (int, float)
        ):
            raise ValueError("schedule_delay_ms must be a number from 0 to 120")
        delay = float(schedule_delay_ms)
        if not 0 <= delay <= 120:
            raise ValueError("schedule_delay_ms must be a number from 0 to 120")
        self.drain()
        self._rank_reports.clear()
        self._rank_timestamps.clear()
        self._completion_ns.clear()
        self._released_jobs.clear()
        self.enabled = enabled
        self.schedule_delay_ms = delay

    def drain(self) -> list[EarlyLoadCompletion]:
        ready: list[EarlyLoadCompletion] = []
        capture_ranks = (
            self.experiment_options["deferral_policy"] != "disabled"
            or self.experiment_options.get("timing_diagnostic_markers", False)
            or self.experiment_options.get("load_only_wait_audit", False)
        )
        for rank, handle in enumerate(self.handles):
            sessions, job_ids, timestamps, statuses, overflow, failure = (
                ops.poll_async_batch_memcpy_completion_ring(handle)
            )
            if overflow:
                raise RuntimeError(
                    f"early KV completion ring overflowed on rank {rank}"
                )
            if not (len(sessions) == len(job_ids) == len(timestamps) == len(statuses)):
                raise RuntimeError("malformed early KV completion records")
            for session, job_id, timestamp, status in zip(
                sessions, job_ids, timestamps, statuses
            ):
                if session != self.session:
                    raise RuntimeError(
                        f"stale early KV completion session on rank {rank}"
                    )
                timeline.emit(
                    "early_notification_received",
                    rank=rank,
                    job_id=job_id,
                    completion_ns=timestamp,
                    status=status,
                )
                if status != 1:
                    raise RuntimeError(f"async KV load {job_id} failed on rank {rank}")
                if not self.enabled or job_id in self._released_jobs:
                    continue
                reports = self._rank_reports.setdefault(job_id, set())
                reports.add(rank)
                if capture_ranks:
                    rank_timestamps = self._rank_timestamps.setdefault(job_id, {})
                    rank_timestamps[rank] = max(
                        timestamp, rank_timestamps.get(rank, timestamp)
                    )
                self._completion_ns[job_id] = max(
                    timestamp, self._completion_ns.get(job_id, timestamp)
                )
                if len(reports) == self.world_size:
                    self._released_jobs.add(job_id)
                    self._rank_reports.pop(job_id, None)
                    completion_ns = self._completion_ns.pop(job_id)
                    timeline.emit(
                        "early_all_ranks_ready",
                        job_id=job_id,
                        completion_ns=completion_ns,
                        ranks=self.world_size,
                    )
                    ready.append(
                        EarlyLoadCompletion(
                            job_id,
                            completion_ns,
                            tuple(
                                self._rank_timestamps[job_id][r]
                                for r in range(self.world_size)
                            )
                            if capture_ranks
                            else (),
                        )
                    )
                    self._rank_timestamps.pop(job_id, None)
            if failure:
                raise RuntimeError(f"async KV transfer worker failed on rank {rank}")
        return ready

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for handle in self.handles:
            ops.destroy_async_batch_memcpy_completion_ring(handle)
        self.handles.clear()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
