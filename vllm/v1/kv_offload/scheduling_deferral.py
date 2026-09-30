# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment-only calibrated scheduling holds; no speculative scheduler calls."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

NS_MS = 1_000_000
POLICIES = {
    "disabled",
    "timing",
    "shadow",
    "deadline",
    "readiness",
    "two_phase_pilot",
    "fill",
    "fill_optional",
}


def bucket(shape: tuple[str, int, int, int]) -> str:
    phase, *sizes = shape
    if phase not in ("prefill", "decode") or any(n <= 0 for n in sizes):
        raise ValueError("invalid workload shape")
    return ":".join([phase, *(str(1 << (n - 1).bit_length()) for n in sizes)])


def quantile(values: list[int], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class Predictor:
    def __init__(self, data: dict):
        version = data.get("version")
        if version == 1:
            if data.get("calibration_cells") != 4:
                raise ValueError("four dedicated calibration cells are required")
        elif version == 2:
            if (
                data.get("calibration_cells") != 12
                or data.get("reserve_method") != "worker_available"
                or data.get("minimum_bucket_samples") != 20
                or data.get("reserve_samples", 0) < 20
            ):
                raise ValueError("invalid worker-available calibration protocol")
        elif version in (3, 4):
            from vllm.v1.kv_offload.batch_predictor import BatchPredictor
            from vllm.v1.kv_offload.execution_predictor import ExecutionPredictor

            self.batch = (
                ExecutionPredictor(data) if version == 4 else BatchPredictor(data)
            )
            self.data = data
            self.reserve_ns = self.batch.reserve_ns
            return
        else:
            raise ValueError("unsupported calibration version")
        self.data = data
        self.reserve_ns = data["reserve_ns"]
        if not math.isfinite(self.reserve_ns) or self.reserve_ns < 0:
            raise ValueError("invalid preparation reserve")
        for key, entry in data["buckets"].items():
            if (
                type(entry["count"]) is not int
                or entry["count"] < 20
                or not math.isfinite(entry["duration_ns"])
                or entry["duration_ns"] <= 0
            ):
                raise ValueError(f"invalid calibrated bucket: {key}")

    @classmethod
    def load(cls, path: str, *, active: bool = False) -> "Predictor":
        predictor = cls(json.loads(Path(path).read_text()))
        if active and (
            not predictor.data.get("frozen")
            or predictor.data.get("shadow_coverage", 0) < 0.8
            or (
                predictor.data["version"] in (2, 3)
                and predictor.data.get("shadow_cells") != 4
            )
        ):
            raise ValueError("active deferral requires frozen 80% shadow coverage")
        if active and predictor.data["version"] == 3:
            checks = predictor.data.get("preflight", {})
            if (
                checks.get("grid_coverage") != 1
                or checks.get("coverage", 0) < 0.95
                or not checks.get("collection_passed")
                or predictor.data.get("residual_cells") != 4
                or predictor.data.get("preflight_cells") != 4
                or (
                    any(k.startswith("target_") for k in predictor.data["models"])
                    and not checks.get("targeted_fallback_passed")
                )
            ):
                raise ValueError(
                    "batch predictor requires collection and preflight gates"
                )
        if active and predictor.data["version"] == 4:
            checks = predictor.data.get("validation", {})
            if not all(
                checks.get(k)
                for k in (
                    "all_rank_markers",
                    "targeted_fallback",
                    "held_out_execution",
                    "start_availability",
                    "configuration_verified",
                )
            ):
                raise ValueError("GPU-start predictor requires independent validation")
        return predictor

    def estimate_completion(self, data, now, ranks):
        from vllm.v1.kv_offload.batch_predictor import CompletionEstimate

        if self.data["version"] in (3, 4):
            return self.batch.estimate_completion(data, now, ranks)
        duration = self.estimate(data["shape"])
        return CompletionEstimate(
            data["dispatch_ns"] + duration if duration is not None else None,
            "bucket",
            "supported" if duration is not None else "unsupported_bucket",
        )

    def estimate(self, shape: tuple[str, int, int, int]) -> float | None:
        entry = self.data["buckets"].get(bucket(shape))
        if entry is None or any(
            n < lo or n > hi
            for n, lo, hi in zip(shape[1:], entry["minimum"], entry["maximum"])
        ):
            return None
        return entry["duration_ns"]


@dataclass
class Hold:
    predecessor: int
    started_ns: int
    deadline_ns: int
    candidates: frozenset[str]
    target: int
    predicted_ns: float
    reserve_ns: float
    polls: int = 0

    @classmethod
    def begin(
        cls,
        predecessor: int,
        now: int,
        dispatch: int,
        duration: float,
        reserve: float,
        snapshot: dict[str, bool],
        cap_ms: int,
        margin_ms: int,
        gain: int,
    ) -> "Hold | None":
        pending = frozenset(k for k, ready in snapshot.items() if not ready)
        deadline = int(
            min(now + cap_ms * NS_MS, dispatch + duration - reserve - margin_ms * NS_MS)
        )
        if not pending or deadline <= now:
            return None
        return cls(
            predecessor,
            now,
            deadline,
            pending,
            min(gain, len(pending)),
            dispatch + duration,
            reserve,
        )

    def poll(
        self,
        now: int,
        snapshot: dict[str, bool],
        *,
        policy: str,
        timing_valid: bool,
        completed: bool,
        capacity: bool,
        outstanding: int,
    ) -> str | None:
        self.polls += 1
        remaining = self.candidates.intersection(snapshot)
        if not timing_valid:
            return "invalid_timing"
        if completed or outstanding == 0:
            return "predecessor_completed"
        if outstanding != 1:
            return "outstanding_compute_changed"
        if not remaining:
            return "candidates_cancelled"
        if not capacity:
            return "capacity_saturated"
        if now >= self.deadline_ns:
            return "deadline"
        if policy == "readiness" and sum(snapshot[k] for k in remaining) >= min(
            self.target, len(remaining)
        ):
            return "readiness_target"
        return None


@dataclass
class TwoPhaseHold(Hold):
    total_deadline_ns: int = 0
    phase: str = "await_start"
    primary_retry_started_ns: int | None = None
    primary_retry_spent_ns: int = 0
    primary_records: tuple = ()

    def primary_retry(self, now, retryable, budget_ms):
        if self.primary_retry_started_ns is not None:
            self.primary_retry_spent_ns += now - self.primary_retry_started_ns
        self.primary_retry_started_ns = now if retryable else None
        return retryable and self.primary_retry_spent_ns < budget_ms * NS_MS

    @classmethod
    def awaiting(cls, predecessor, now, snapshot, cap_ms, start_ms, gain):
        pending = frozenset(k for k, ready in snapshot.items() if not ready)
        if not pending or min(cap_ms, start_ms) <= 0:
            return None
        return cls(
            predecessor=predecessor,
            started_ns=now,
            deadline_ns=now + min(cap_ms, start_ms) * NS_MS,
            candidates=pending,
            target=min(gain, len(pending)),
            predicted_ns=0,
            reserve_ns=0,
            total_deadline_ns=now + cap_ms * NS_MS,
        )

    def observe_start(self, predicted, reserve, margin_ms):
        if self.phase != "await_start":
            raise ValueError("execution deadline may only be established once")
        self.phase = "execution"
        self.predicted_ns = predicted
        self.reserve_ns = reserve
        self.deadline_ns = int(
            min(self.total_deadline_ns, predicted - reserve - margin_ms * NS_MS)
        )


@dataclass
class FillHold(TwoPhaseHold):
    """A frozen fill target whose total cap never depends on GPU start."""

    initially_ready: frozenset[str] = frozenset()
    prediction_reason: str = "pending"
    prediction_checked: bool = False
    last_poll_ns: int = 0

    @classmethod
    def filling(cls, predecessor, now, snapshot, cap_ms, fraction, capacity):
        if cap_ms <= 0 or not any(not ready for ready in snapshot.values()):
            return None
        target = min(len(snapshot), math.ceil(capacity * fraction))
        ready = frozenset(k for k, value in snapshot.items() if value)
        if len(ready) >= target:
            return None
        return cls(
            predecessor=predecessor,
            started_ns=now,
            deadline_ns=now + cap_ms * NS_MS,
            total_deadline_ns=now + cap_ms * NS_MS,
            candidates=frozenset(snapshot),
            target=target,
            predicted_ns=0,
            reserve_ns=0,
            initially_ready=ready,
            last_poll_ns=now,
        )


class OutstandingLoads:
    """Dispatch-visible loads; reclaim only at engine completion receipt."""

    def __init__(self):
        self.jobs = {}
        self.valid = True

    def dispatch(self, jobs):
        for job_id, job in jobs.items():
            tokens = job.external_tokens
            if (
                job_id in self.jobs
                or len(self.jobs) >= 4096
                or type(tokens) is not int
                or tokens <= 0
            ):
                self.valid = False
                continue
            self.jobs[job_id] = tokens

    def complete(self, job_id):
        self.jobs.pop(job_id, None)

    def snapshot(self):
        return [len(self.jobs), sum(self.jobs.values())] if self.valid else None

    def reset(self):
        self.jobs.clear()
        self.valid = True
