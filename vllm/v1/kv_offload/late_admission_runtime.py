# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment-only split preparation before the scheduler's final membership seal."""

import os
import pickle
import select
import struct
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.request import RequestStatus


class PreparationFence(AsyncModelRunnerOutput):
    """Retire staging only after its stream work, on the existing output thread."""

    def __init__(self, output, event):
        self.output, self.event = output, event
        self.consumed = False

    def get_output(self):
        if self.consumed:
            raise RuntimeError("preparation fence consumed twice")
        self.event.synchronize()
        self.consumed = True
        return self.output


@dataclass(frozen=True)
class CandidateNotice:
    req_id: str


def prepare_candidates(scheduler, token_budget, base_reqs):
    """Stage bounded old in-flight loads after ordinary ready work is selected."""
    mode = getattr(scheduler, "_late_admission_mode", 0)
    if not mode or scheduler.has_mamba_layers:
        return set()
    if (
        scheduler.lora_config
        or scheduler.is_encoder_decoder
        or scheduler.num_lookahead_tokens
    ):
        return set()
    if any(
        r.mm_features or r.structured_output_request or r.pooling_params
        for r in scheduler.requests.values()
    ):
        return set()
    connector = getattr(scheduler.connector, "connector_scheduler", None)
    if (
        connector is None
        or not hasattr(connector, "_req_status")
        or not hasattr(connector, "_current_batch_load_jobs")
        or getattr(connector, "_sliding_window_groups", True)
    ):
        return set()
    slots = getattr(scheduler, "_late_candidate_slots", 4)
    capacity = min(slots, scheduler.max_num_running_reqs - len(scheduler.running))
    capacity = max(capacity, 0)
    pending_now = set(connector._current_batch_load_jobs)
    candidates = []
    remaining = token_budget
    for request in sorted(
        scheduler.requests.values(), key=lambda r: (r.arrival_time, r.request_id)
    ):
        if (
            request.status != RequestStatus.WAITING_FOR_REMOTE_KVS
            or request.prompt_token_ids is None
            or getattr(request, "resumable", False)
            or request.num_preemptions
            or request.num_output_tokens
            or request.request_id in scheduler.finished_recving_kv_req_ids
            or not 0 < request.num_computed_tokens < request.num_tokens
        ):
            continue
        status = connector._req_status.get(request.request_id)
        if (
            status is None
            or not status.transfer_jobs
            or status.transfer_jobs & pending_now
        ):
            continue
        # No speculative suffix reservation: ordinary allocation remains at seal.
        # Whole residual prefills only in the first prototype, bounded by budget.
        demand = request.num_tokens - request.num_computed_tokens
        if demand > min(remaining, 4096):
            continue
        candidates.append(
            CandidateNotice(request.request_id)
            if mode == 4
            else NewRequestData.from_request(
                request,
                scheduler.kv_cache_manager.get_blocks(
                    request.request_id
                ).get_block_ids(),
                request._all_token_ids,
            )
        )
        remaining -= demand
        if len(candidates) == capacity:
            break
    staged = scheduler._late_admission_prepare(
        scheduler.current_step,
        base_reqs,
        candidates,
    )
    allowed = (
        set(staged) & getattr(scheduler, "_late_kv_snapshot", set())
        if mode in (2, 3, 4)
        else set()
    )
    allowed -= getattr(scheduler, "_late_cancelled", set())
    record(
        scheduler,
        "seal_eligibility",
        step=scheduler.current_step,
        candidates=[c.req_id for c in candidates],
        staged=list(staged),
        eligible=list(allowed),
        mode=mode,
        token_budget=token_budget,
        transport=getattr(scheduler, "_late_transport", {}),
    )
    return allowed


def signal_prepared(path, step, rank, status, slot=-1):
    """Publish preparation only; this does not consume or bypass an RPC response."""
    fd = os.open(path, os.O_RDWR)
    try:
        batch, world = struct.unpack("<QQ", os.pread(fd, 16, 0))
        if batch != step or not 0 <= rank < world or status not in (1, 2, 3):
            raise RuntimeError("late preparation control identity mismatch")
        offset = 16 + (slot + 1) * world + rank
        if os.pread(fd, 1, offset) != b"\0":
            raise RuntimeError("duplicate late preparation publication")
        os.pwrite(fd, bytes([status]), offset)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class PreparedRequest:
    req_id: str
    token_ids: np.ndarray
    prompt_len: int
    ready: threading.Event
    prepared_ns: int


class PreparationLane:
    """One private CPU preparation lane per worker, outside the RPC FIFO."""

    def __init__(
        self,
        rank: int,
        generation: int,
        notice_only: bool = False,
        candidate_slots: int = 4,
    ):
        self.rank = rank
        self.generation = generation
        self.notice_only = notice_only
        self.candidate_slots = candidate_slots
        self.directory = tempfile.TemporaryDirectory(prefix="vllm-late-worker-")
        self.path = str(Path(self.directory.name) / "prepare.fifo")
        os.mkfifo(self.path, 0o600)
        self.pipe = os.open(self.path, os.O_RDWR | os.O_NONBLOCK)
        self.lock = threading.Lock()
        self.prepared: dict[int, dict[str, PreparedRequest]] = {}
        self.discarded_through = -1
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        pending = bytearray()
        while not self.closed:
            ready, _, _ = select.select([self.pipe], [], [], 0.1)
            if not ready:
                continue
            pending.extend(os.read(self.pipe, 4096))
            while b"\n" in pending:
                raw_path, _, remainder = pending.partition(b"\n")
                pending = bytearray(remainder)
                if raw_path:
                    self._prepare_path(raw_path.decode())

    def _prepare_path(self, path):
        control = None
        step = None
        current_slot = -1
        try:
            with open(path, "rb") as source:
                generation, step, base, candidates, control = pickle.load(source)
            if generation != self.generation or len(candidates) > self.candidate_slots:
                raise RuntimeError("late preparation generation or capacity mismatch")
            staged = {}
            for request in base:
                staged[request.req_id] = self._prepare_request(request)
            with self.lock:
                if step > self.discarded_through:
                    if len(self.prepared) >= 2 or step in self.prepared:
                        raise RuntimeError("late preparation ownership limit exceeded")
                    self.prepared[step] = staged
            signal_prepared(control, step, self.rank, 1)
            for slot, request in enumerate(candidates):
                current_slot = slot
                prepared = (
                    self._prepare_notice(request)
                    if self.notice_only
                    else self._prepare_request(request)
                )
                with self.lock:
                    if step > self.discarded_through:
                        staged[request.req_id] = prepared
                # The engine removes the control file immediately after the
                # seal. Keep preparing later slots for entry-time diagnostics
                # even when their acknowledgments can no longer be published.
                with suppress(FileNotFoundError):
                    signal_prepared(control, step, self.rank, 1, slot)
        except Exception:
            if control is not None and step is not None:
                with suppress(OSError, RuntimeError):
                    signal_prepared(control, step, self.rank, 3, current_slot)

    @staticmethod
    def _prepare_request(request):
        tokens = request.prefill_token_ids
        if tokens is None:
            raise RuntimeError("late preparation requires text token ids")
        compact = np.asarray(tokens, dtype=np.int32).copy()
        compact.flags.writeable = False
        ready = threading.Event()
        prepared = PreparedRequest(
            request.req_id,
            compact,
            request.prompt_len,
            ready,
            time.monotonic_ns(),
        )
        ready.set()
        return prepared

    @staticmethod
    def _prepare_notice(request):
        if not isinstance(request, CandidateNotice):
            raise RuntimeError("late notice descriptor mismatch")
        ready = threading.Event()
        ready.set()
        return PreparedRequest(
            request.req_id,
            np.empty(0, dtype=np.int32),
            0,
            ready,
            time.monotonic_ns(),
        )

    def available(self, step):
        with self.lock:
            return sorted(self.prepared.get(step, {}))

    def snapshot(self, step):
        with self.lock:
            return {
                req_id: data.prepared_ns
                for req_id, data in self.prepared.get(step, {}).items()
            }

    def consume(self, step, selected):
        with self.lock:
            self.discarded_through = max(self.discarded_through, step)
            prepared = self.prepared.pop(step, {})
            for old in tuple(self.prepared):
                if old < step:
                    del self.prepared[old]
        adopted = {
            req_id: data for req_id, data in prepared.items() if req_id in selected
        }
        for data in adopted.values():
            data.ready.wait()
        return adopted

    def close(self):
        self.closed = True
        self.thread.join(timeout=1)
        os.close(self.pipe)
        with self.lock:
            self.prepared.clear()
        self.directory.cleanup()


def stage_rpc(core, step, base, candidates):
    """Publish B independently, then seal after the configured base-ready hold."""
    world = core.vllm_config.parallel_config.world_size
    queued = getattr(core, "batch_queue", None)
    core.scheduler._late_kv_snapshot = set()
    core.scheduler._late_cancelled = set()
    core.scheduler._late_transport = dict(path="ordinary", start_ns=time.monotonic_ns())
    if not queued:
        return []
    predecessor = queued[-1][0]
    if predecessor.poll():
        return []
    started = time.monotonic_ns()
    delay_us = getattr(core.scheduler, "_late_seal_delay_us", 0) if candidates else 0
    extension_us = getattr(core.scheduler, "_late_seal_extension_us", 0)
    seal_when_all_ready = getattr(
        core.scheduler, "_late_seal_when_all_candidates_ready", False
    )
    partial_min = getattr(core.scheduler, "_late_partial_min_pending_candidates", 0)
    partial_percent = getattr(core.scheduler, "_late_partial_ready_percent", 0)
    policy = getattr(core.scheduler, "_late_seal_policy", "fixed")
    prediction = None
    predicted_deadline_ns = None
    fd, path = tempfile.mkstemp(prefix="vllm-late-")
    data_fd, data_path = tempfile.mkstemp(prefix="vllm-late-data-")
    try:
        os.write(
            fd,
            struct.pack("<QQ", step, world) + bytes(world * (1 + len(candidates))),
        )
        with os.fdopen(data_fd, "wb") as destination:
            pickle.dump(
                (core.scheduler._late_generation, step, base, candidates, path),
                destination,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        core.model_executor.publish_late_preparation(data_path)
        deadline = time.monotonic() + 30
        previous = None
        previous_kv_ready = None
        base_ready_ns = None
        extension_started_ns = None
        seal_reason = None
        cancelled = None
        partial_ready_count = 0
        partial_target = (len(candidates) * partial_percent + 99) // 100
        while True:
            predecessor_ready = predecessor.poll()
            core._refresh_early_load_readiness(poll_markers=False)
            kv_ready = sorted(
                c.req_id
                for c in candidates
                if c.req_id in core.scheduler.finished_recving_kv_req_ids
            )
            if kv_ready != previous_kv_ready:
                record(
                    core.scheduler, "candidate_kv_transition", step=step, ready=kv_ready
                )
                previous_kv_ready = kv_ready
            statuses = os.pread(fd, world * (1 + len(candidates)), 16)
            if len(statuses) != world * (1 + len(candidates)) or any(
                s > 3 for s in statuses
            ):
                raise RuntimeError("invalid late preparation publication")
            if 3 in statuses:
                raise RuntimeError("worker preparation failed")
            if statuses != previous:
                record(
                    core.scheduler,
                    "preparation_ack_transition",
                    step=step,
                    base=list(statuses[:world]),
                    candidates=[
                        list(statuses[world * (slot + 1) : world * (slot + 2)])
                        for slot in range(len(candidates))
                    ],
                )
                previous = statuses
            if all(s == 1 for s in statuses[:world]):
                now_ns = time.monotonic_ns()
                if base_ready_ns is None:
                    base_ready_ns = now_ns
                    record(
                        core.scheduler, "all_base_ready", step=step, observed_ns=now_ns
                    )
                ready_acked = [
                    candidate.req_id
                    for slot, candidate in enumerate(candidates)
                    if candidate.req_id in kv_ready
                    and all(
                        status == 1
                        for status in statuses[world * (slot + 1) : world * (slot + 2)]
                    )
                ]
                if policy == "predictive" and candidates:
                    from vllm.v1.kv_offload.execution_timing import valid_snapshot
                    from vllm.v1.kv_offload.seal_progress import compute_progress

                    has_compute, compute_finished = compute_progress(core, queued)
                    if has_compute and compute_finished:
                        seal_reason = "predecessor_compute_finished"
                        break
                    timing = getattr(core, "deferral_timing", None)
                    a_output = next(
                        (
                            output
                            for _, output, _ in reversed(queued)
                            if getattr(output, "total_num_scheduled_tokens", 0) > 0
                        ),
                        None,
                    )
                    a_step = getattr(a_output, "readiness_timeline_step", None)
                    if timing and a_step and predicted_deadline_ns is None:
                        data = timing.steps.get(a_step)
                        records = timing.read_execution(a_step)
                        if (
                            data
                            and len(records) == world
                            and valid_snapshot(records, a_step, time.monotonic_ns())
                        ):
                            predicted = timing.seal_estimator.predict(data)
                            if predicted:
                                predicted = (
                                    predicted
                                    * core.scheduler._late_seal_prediction_scale_percent
                                    // 100
                                )
                                reserve_ns = timing.seal_estimator.preparation_reserve(
                                    core.scheduler._late_seal_prediction_reserve_us
                                    * 1000
                                )
                                predicted_deadline_ns = (
                                    max(r[3] for r in records) + predicted - reserve_ns
                                )
                                prediction = dict(
                                    predecessor_step=a_step,
                                    predicted_duration_ns=predicted,
                                    deadline_ns=predicted_deadline_ns,
                                    features=timing.seal_estimator.features(
                                        data
                                    ).tolist(),
                                    training_samples=len(timing.seal_estimator.samples),
                                    preparation_samples=len(
                                        timing.seal_estimator.preparation_ns
                                    ),
                                    preparation_reserve_ns=reserve_ns,
                                )
                                record(
                                    core.scheduler,
                                    "seal_prediction",
                                    step=step,
                                    **prediction,
                                )
                    if predicted_deadline_ns and now_ns >= predicted_deadline_ns:
                        seal_reason = "predicted_compute_deadline"
                        break
                    if has_compute:
                        if (
                            seal_when_all_ready
                            and candidates
                            and len(ready_acked) == len(candidates)
                        ):
                            seal_reason = "all_candidates_ready"
                            break
                        if (
                            now_ns
                            >= base_ready_ns
                            + core.scheduler._late_seal_safety_us * 1000
                        ):
                            seal_reason = "adaptive_safety_deadline"
                            break
                        time.sleep(0.001)
                        continue
                if (
                    seal_when_all_ready
                    and candidates
                    and len(ready_acked) == len(candidates)
                ):
                    seal_reason = "all_candidates_ready"
                    break
                if (
                    partial_min
                    and len(candidates) >= partial_min
                    and len(ready_acked) >= partial_target
                    and not any(
                        getattr(output, "total_num_scheduled_tokens", 0) > 0
                        and not future.poll()
                        for future, output, _ in queued
                    )
                ):
                    cancelled = cancelled_candidates(core, candidates)
                    partial_ready_count = len(set(ready_acked) - cancelled)
                    if partial_ready_count >= partial_target:
                        seal_reason = "partial_ready_no_compute"
                        break
                if now_ns >= base_ready_ns + delay_us * 1000:
                    if extension_started_ns is None:
                        cancelled_at_deadline = cancelled_candidates(core, candidates)
                        pending_kv = any(
                            c.req_id not in kv_ready
                            and c.req_id not in cancelled_at_deadline
                            for c in candidates
                        )
                        productive_a = (
                            getattr(queued[-1][1], "total_num_scheduled_tokens", 0) > 0
                        )
                        if not (
                            extension_us
                            and pending_kv
                            and productive_a
                            and not predecessor_ready
                        ):
                            seal_reason = "deadline"
                            break
                        extension_started_ns = now_ns
                        record(
                            core.scheduler,
                            "seal_extension_started",
                            step=step,
                            observed_ns=now_ns,
                        )
                    if predecessor_ready:
                        seal_reason = "predecessor_finished_during_extension"
                        break
                    if now_ns >= base_ready_ns + (delay_us + extension_us) * 1000:
                        seal_reason = "extension_deadline"
                        break
            elif predecessor_ready:
                record(core.scheduler, "predecessor_first_fallback", step=step)
                core.scheduler._late_transport = dict(
                    start_ns=started,
                    seal_ns=time.monotonic_ns(),
                    statuses=list(statuses),
                    predecessor_ready=True,
                )
                return []
            if time.monotonic() >= deadline:
                raise TimeoutError("late preparation timed out")
            time.sleep(0.000025)
        if cancelled is None:
            cancelled = cancelled_candidates(core, candidates)
        partial_ready_count = len(set(ready_acked) - cancelled)
        sealed_ns = time.monotonic_ns()
        core.scheduler._late_cancelled = cancelled
        core.scheduler._late_kv_snapshot = set(kv_ready)
        core.scheduler._late_transport = dict(
            start_ns=started,
            base_ready_ns=base_ready_ns,
            snapshot_ns=sealed_ns,
            seal_ns=sealed_ns,
            seal_delay_target_us=delay_us,
            seal_extension_target_us=extension_us,
            seal_extension_started_ns=extension_started_ns,
            seal_extension_actual_ns=(
                sealed_ns - extension_started_ns if extension_started_ns else 0
            ),
            seal_hold_actual_ns=sealed_ns - base_ready_ns,
            seal_reason=seal_reason,
            partial_ready_count=partial_ready_count,
            partial_target=partial_target,
            partial_pending_count=len(candidates),
            statuses=list(statuses),
            kv_ready=kv_ready,
            cancelled=sorted(cancelled),
            predecessor_ready=predecessor_ready,
            adaptive_policy=policy,
            prediction=prediction,
        )
        return [
            c.req_id
            for slot, c in enumerate(candidates)
            if all(s == 1 for s in statuses[world * (slot + 1) : world * (slot + 2)])
        ]
    finally:
        os.close(fd)
        os.unlink(path)
        os.unlink(data_path)


def cancelled_candidates(core, candidates):
    cancelled = set()
    aborts = getattr(core, "aborts_queue", None)
    if aborts is not None:
        ids = {c.req_id for c in candidates}
        with aborts.mutex:
            for pending in aborts.queue:
                cancelled.update(
                    ids.intersection(
                        (pending,) if isinstance(pending, str) else pending
                    )
                )
    return cancelled


def record(owner, event, **fields):
    counts = getattr(owner, "_late_counts", None)
    if counts is None:
        counts = owner._late_counts = {}
    counts[event] = counts.get(event, 0) + 1
    for key in ("candidates", "staged", "eligible", "activated"):
        if key in fields:
            name = event + "_" + key
            counts[name] = counts.get(name, 0) + len(fields[key])
    if not getattr(owner, "_late_record_enabled", False):
        return
    rows = getattr(owner, "_late_admission_records", None)
    if rows is None:
        rows = owner._late_admission_records = []
    if len(rows) >= 4096:
        raise RuntimeError("late admission record capacity exceeded")
    rows.append(dict(event=event, time_ns=time.monotonic_ns(), **fields))


def export_records(owner, role, rank=-1):
    import json
    from pathlib import Path

    root = Path(os.environ["VLLM_DEFERRAL_TIMING_DIR"]) / "late_admission"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{os.getpid()}-{role}-{time.monotonic_ns()}.json"
    rows = getattr(owner, "_late_admission_records", [])
    ended = time.monotonic_ns()
    target.write_text(
        json.dumps(
            dict(
                role=role,
                rank=rank,
                generation=getattr(owner, "_late_generation", 0),
                records=rows,
                capacity=4096,
                overflow=False,
                markers=getattr(owner, "_late_record_enabled", False),
                counts=getattr(owner, "_late_counts", {}),
                begin_ns=getattr(owner, "_late_period_begin_ns", 0),
                end_ns=ended,
            )
        )
    )
    owner._late_admission_records = []
    owner._late_counts = {}
    owner._late_period_begin_ns = ended
    return str(target)
