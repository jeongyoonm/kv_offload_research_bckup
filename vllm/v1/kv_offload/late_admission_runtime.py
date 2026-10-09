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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import numpy as np

from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_offload.dataset_shadow import collection_compute_held
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


@dataclass(frozen=True)
class CompactPreparationRequest:
    """Immutable token bytes and small metadata, encoded in the engine process."""

    source: NewRequestData
    token_bytes: bytes
    prompt_len: int

    @property
    def req_id(self):
        return self.source.req_id


def _immutable_text_prefill(request):
    return (
        request.prefill_token_ids is not None
        and request.prompt_token_ids is not None
        and request.prefill_token_ids == request.prompt_token_ids
        and not request.mm_features
        and request.prompt_embeds is None
        and request.lora_request is None
        and request.pooling_params is None
        and request.prompt_is_token_ids is None
    )


def compact_preparation_request(request):
    """Snapshot text tokens without retaining token lists in worker metadata."""
    if not _immutable_text_prefill(request):
        raise RuntimeError("compact preparation requires immutable text prefill")
    tokens = np.asarray(request.prefill_token_ids, dtype=np.int32)
    return CompactPreparationRequest(
        replace(request, prompt_token_ids=None, prefill_token_ids=None),
        tokens.tobytes(),
        request.prompt_len,
    )


def prepare_candidates(scheduler, token_budget, base_reqs):
    """Stage bounded old in-flight loads after ordinary ready work is selected."""
    mode = getattr(scheduler, "_late_admission_mode", 0)
    scheduler._late_reference_requests = {}
    materialize = getattr(scheduler, "_late_cached_requests", False) or getattr(
        scheduler, "_late_pinned_packing", False
    )
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
    if materialize:
        base_reqs = [
            r
            for r in base_reqs
            if r.prefill_token_ids == r.prompt_token_ids
            and r.prompt_token_ids is not None
        ]
    slots = getattr(scheduler, "_late_candidate_slots", 4)
    capacity = min(slots, scheduler.max_num_running_reqs - len(scheduler.running))
    capacity = max(capacity, 0)
    pending_now = set(connector._current_batch_load_jobs)
    candidates: list[CandidateNotice | NewRequestData] = []
    remaining = token_budget
    for request in sorted(
        scheduler.requests.values(), key=lambda r: (r.arrival_time, r.request_id)
    ):
        if len(candidates) >= capacity:
            break
        if collection_compute_held(request):
            continue
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
            if mode == 4 and not materialize
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
        from vllm.v1.kv_offload.seal_notifications import signal_preparation

        signal_preparation(path, step, rank, status, slot)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class PreparedRequest:
    req_id: str
    token_ids: np.ndarray
    prompt_len: int
    ready: threading.Event
    prepared_ns: int
    source: NewRequestData | None = None
    cache_key: tuple[int, int] | None = None
    pinned_tokens: "torch.Tensor | None" = None


class PreparationLane:
    """One private CPU preparation lane per worker, outside the RPC FIFO."""

    def __init__(
        self,
        rank: int,
        generation: int,
        notice_only: bool = False,
        candidate_slots: int = 4,
        cached_requests: bool = False,
        pinned_packing: bool = False,
        device: "torch.device | None" = None,
    ):
        self.rank = rank
        self.generation = generation
        self.notice_only = notice_only
        self.cached_requests = cached_requests
        self.pinned_packing = pinned_packing
        self.device = device
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
        if self.pinned_packing and self.device is not None:
            import torch

            torch.accelerator.set_device_index(self.device)
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
                staged[request.req_id] = self._prepare_data(request, step)
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
                    else self._prepare_data(request, step)
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

    def _prepare_data(self, request, step):
        if not (self.cached_requests or self.pinned_packing):
            return self._prepare_request(request)
        compact = isinstance(request, CompactPreparationRequest)
        if compact:
            source = request.source
            if (
                type(request.token_bytes) is not bytes
                or request.prompt_len < 0
                or len(request.token_bytes) != request.prompt_len * 4
                or source.prompt_token_ids is not None
                or source.prefill_token_ids is not None
                or source.mm_features
                or source.prompt_embeds is not None
                or source.lora_request is not None
                or source.pooling_params is not None
                or source.prompt_is_token_ids is not None
            ):
                raise RuntimeError("invalid compact preparation descriptor")
            tokens = np.frombuffer(request.token_bytes, dtype=np.int32)
            prompt_len = request.prompt_len
        else:
            if not _immutable_text_prefill(request):
                raise RuntimeError("cached preparation requires immutable text prefill")
            source = request
            tokens = np.asarray(request.prefill_token_ids, dtype=np.int32)
            prompt_len = request.prompt_len
        pinned = None
        if self.pinned_packing:
            import torch

            pinned = torch.empty(tokens.size, dtype=torch.int32, pin_memory=True)
            np.copyto(pinned.numpy(), tokens)
            tokens = pinned.numpy()
        elif not compact:
            tokens = tokens.copy()
        tokens.flags.writeable = False
        ready = threading.Event()
        ready.set()
        return PreparedRequest(
            request.req_id,
            tokens,
            prompt_len,
            ready,
            time.monotonic_ns(),
            source,
            (self.generation, step),
            pinned,
        )

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
        self.thread.join()
        os.close(self.pipe)
        with self.lock:
            self.prepared.clear()
        self.directory.cleanup()


def stage_rpc(core, step, base, candidates):
    """Publish B independently, then seal after the configured base-ready hold."""
    from vllm.v1.kv_offload.seal_progress import compute_progress

    world = core.vllm_config.parallel_config.world_size
    queued = getattr(core, "batch_queue", None)
    core.scheduler._late_reference_requests = {}
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
    load_only_ready_percent = getattr(
        core.scheduler, "_late_seal_load_only_ready_percent", 0
    )
    load_only_fraction_target = (
        (len(candidates) * load_only_ready_percent + 99) // 100
        if load_only_ready_percent
        else None
    )
    policy = getattr(core.scheduler, "_late_seal_policy", "fixed")
    repeated_intervals = 0
    prediction = None
    predicted_deadline_ns = None
    compute_seal_deadline_ns = None
    compute_seal_deadline_armed_ns = None
    submission_trace = os.environ.get("VLLM_DEFERRAL_PREDICTOR_PROGRESS") == "1"
    progress_read_ns = 0
    progress_signature = None
    progress_frame = None
    refinement = None
    effective_seal_expiry_ns = None
    adaptive_poll_us = getattr(core.scheduler, "_late_seal_adaptive_poll_us", 1000)
    progress_polls = 0
    poll_cpu_ns = 0
    sleep_ns = 0
    adaptive_sleeps = 0
    coarse_sleeps = 0
    requested_sleep_ns = 0
    hybrid_wait_us = getattr(core.scheduler, "_late_seal_hybrid_wait_us", 0)
    hybrid_guard_us = getattr(core.scheduler, "_late_seal_hybrid_guard_us", 50_000)
    prediction_to_readiness = getattr(
        core.scheduler, "_late_seal_prediction_to_readiness", False
    )
    compute_deadline = (
        getattr(core.scheduler, "_late_seal_compute_deadline", False) is True
    )
    compute_deadline_shadow = (
        getattr(core.scheduler, "_late_seal_compute_deadline_shadow", False) is True
    )
    compute_deadline_lead_us = getattr(
        core.scheduler, "_late_seal_compute_deadline_lead_us", 0
    )
    prediction_bias_us = getattr(core.scheduler, "_late_seal_prediction_bias_us", 0)
    if type(prediction_bias_us) is not int:
        prediction_bias_us = 0
    hybrid_transition_ns = None
    hybrid_polling_ns = None
    full_prediction_wait = getattr(
        core.scheduler, "_late_seal_full_prediction_wait", False
    )
    coalesced = (
        getattr(core.scheduler, "_late_seal_coalesced_notifications", False) is True
    )
    notifications = None
    notifications_armed = False
    if coalesced:
        notifications = getattr(core.early_load_readiness, "seal_notifications", None)
        if notifications is None or not full_prediction_wait:
            raise RuntimeError("coalesced seal notifications are unavailable")
    wait_trace = getattr(core.scheduler, "_late_seal_wait_trace", False)
    wakeup = None
    wake_ticket = None
    waits: list[dict[str, int | str]] = []

    def adaptive_sleep(predicted_deadline=None):
        nonlocal poll_cpu_ns, sleep_ns
        nonlocal adaptive_sleeps, coarse_sleeps, requested_sleep_ns
        nonlocal hybrid_transition_ns, hybrid_polling_ns
        nonlocal wakeup
        before = time.monotonic_ns()
        requested = adaptive_poll_us * 1000
        if full_prediction_wait and not has_compute:
            requested = 2_000_000
        coarse = False
        if (hybrid_wait_us or full_prediction_wait) and predicted_deadline:
            transition = predicted_deadline - hybrid_guard_us * 1000
            if prediction_to_readiness:
                if hybrid_transition_ns is None:
                    hybrid_transition_ns = transition
                transition = hybrid_transition_ns
                if before >= transition and hybrid_polling_ns is None:
                    hybrid_polling_ns = before
                    record(
                        core.scheduler,
                        "seal_prediction_polling",
                        step=step,
                        transition_ns=transition,
                        observed_ns=before,
                    )
            coarse_remaining = transition - before
            safety_remaining = (
                base_ready_ns + core.scheduler._late_seal_safety_us * 1000 - before
            )
            if coarse_remaining > requested and safety_remaining > requested:
                requested = min(
                    coarse_remaining if full_prediction_wait else hybrid_wait_us * 1000,
                    coarse_remaining,
                    safety_remaining,
                )
                coarse = True
        if compute_deadline and has_compute:
            expiry = effective_seal_expiry_ns or (
                base_ready_ns + core.scheduler._late_seal_safety_us * 1000
            )
            requested = min(requested, max(0, expiry - before))
        if full_prediction_wait and coarse and wakeup is None:
            from vllm.v1.kv_offload.seal_wakeup import TransportWakeup

            wakeup = (
                notifications
                if notifications is not None
                else TransportWakeup(core, fd, queued)
            )
            poll_cpu_ns += time.thread_time_ns() - poll_cpu_start
            # Recheck progress after installing the observer, before any wait.
            return
        if coarse:
            coarse_sleeps += 1
        adaptive_sleeps += 1
        requested_sleep_ns += requested
        poll_cpu_ns += time.thread_time_ns() - poll_cpu_start
        reason = "timer"
        wait_begin = time.monotonic_ns()
        if full_prediction_wait and coarse:
            assert wakeup is not None
            reason = wakeup.wait(wake_ticket, requested / 1e9)
        else:
            if wakeup is not None:
                wakeup.pause()
            time.sleep(requested / 1e9)
        ended = time.monotonic_ns()
        sleep_ns += ended - before
        if wait_trace:
            if len(waits) >= 4096:
                raise RuntimeError("seal wait trace capacity exceeded")
            waits.append(
                dict(
                    begin_ns=wait_begin,
                    end_ns=ended,
                    requested_ns=requested,
                    progress_begin_ns=progress_begin_ns,
                    decision_ns=before,
                    phase="coarse" if coarse else "fine",
                    reason=reason,
                )
            )

    fd, path = tempfile.mkstemp(prefix="vllm-late-")
    data_fd, data_path = tempfile.mkstemp(prefix="vllm-late-data-")
    try:
        os.write(
            fd,
            struct.pack("<QQ", step, world) + bytes(world * (1 + len(candidates))),
        )
        with os.fdopen(data_fd, "wb") as destination:
            if getattr(core.scheduler, "_late_compact_tokens", False) is True:
                wire_base = [compact_preparation_request(r) for r in base]
                wire_candidates = [compact_preparation_request(r) for r in candidates]
            else:
                wire_base, wire_candidates = base, candidates
            pickle.dump(
                (
                    core.scheduler._late_generation,
                    step,
                    wire_base,
                    wire_candidates,
                    path,
                ),
                destination,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        if notifications is not None:
            connector = core.scheduler.connector.connector_scheduler
            candidate_jobs = [
                sorted(connector._req_status[c.req_id].transfer_jobs)
                for c in candidates
            ]
            compute_steps = [
                output.readiness_timeline_step
                for _, output, _ in queued
                if output.total_num_scheduled_tokens > 0
            ]
            notifications.arm(
                step, core.scheduler._late_generation, candidate_jobs, compute_steps
            )
            notifications_armed = True
            notifications.descriptor(
                path, 0, core.scheduler._late_generation, track=False
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
        target = 0
        partial_target = (len(candidates) * partial_percent + 99) // 100
        while True:
            if wakeup is not None:
                wake_ticket = wakeup.ticket()
            progress_polls += 1
            progress_begin_ns = time.monotonic_ns()
            poll_cpu_start = time.thread_time_ns()
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
            if notifications is not None:
                notifications.seed(
                    core.early_load_readiness,
                    (slot for slot, c in enumerate(candidates) if c.req_id in kv_ready),
                )
                for compute_step in compute_steps:
                    notifications.seed_compute(
                        compute_step, core.deferral_timing.read_execution(compute_step)
                    )
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
                if candidates and (
                    policy == "event"
                    or (
                        prediction_to_readiness
                        and not compute_progress(core, queued)[0]
                    )
                ):
                    cancelled = cancelled_candidates(core, candidates)
                    active = {c.req_id for c in candidates} - cancelled
                    ready_count = len(set(ready_acked) & active)
                    target = min(
                        core.scheduler._late_seal_target_candidates, len(active)
                    )
                    has_compute, compute_finished = compute_progress(core, queued)
                    load_only_target = getattr(
                        core.scheduler, "_late_seal_load_only_target_candidates", 0
                    )
                    if not has_compute:
                        if load_only_fraction_target is not None:
                            target = min(load_only_fraction_target, len(active))
                        elif load_only_target:
                            target = min(target, load_only_target)
                    if ready_count >= target:
                        seal_reason = "candidate_target_ready"
                        break
                    if has_compute and compute_finished:
                        seal_reason = "predecessor_compute_finished"
                        break
                    if (
                        now_ns
                        >= base_ready_ns + core.scheduler._late_seal_safety_us * 1000
                    ):
                        seal_reason = "adaptive_safety_deadline"
                        break
                    interval = core.scheduler._late_seal_repeat_us * 1000
                    rounds = (now_ns - base_ready_ns) // interval
                    if rounds > repeated_intervals:
                        repeated_intervals = rounds
                        record(
                            core.scheduler,
                            "seal_repeat",
                            step=step,
                            intervals=rounds,
                            ready=ready_count,
                            target=target,
                            has_compute=has_compute,
                        )
                    adaptive_sleep()
                    continue
                if policy == "predictive" and candidates:
                    from vllm.v1.kv_offload.execution_timing import valid_snapshot

                    has_compute, compute_finished = compute_progress(core, queued)
                    if (
                        has_compute
                        and (compute_deadline or compute_deadline_shadow)
                        and effective_seal_expiry_ns is None
                    ):
                        effective_seal_expiry_ns = (
                            base_ready_ns + core.scheduler._late_seal_safety_us * 1000
                        )
                    if prediction_to_readiness:
                        cancelled = cancelled_candidates(core, candidates)
                        active = {c.req_id for c in candidates} - cancelled
                        target = min(
                            core.scheduler._late_seal_target_candidates, len(active)
                        )
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
                    refinement_model = (
                        getattr(timing.seal_estimator, "refinement_surface", None)
                        if timing
                        else None
                    )
                    if timing and a_step and (submission_trace or refinement_model):
                        progress_now = time.monotonic_ns()
                        if progress_now - progress_read_ns >= 10_000_000:
                            progress_read_ns = progress_now
                            data = timing.steps.get(a_step)
                            primary_valid, _, primary = timing.read(a_step)
                            signature = tuple(r[3] if r else 0 for r in primary)
                            if (
                                data
                                and primary_valid
                                and signature != progress_signature
                            ):
                                execution = timing.read_execution(a_step)
                                observed_ns = time.monotonic_ns()
                                if valid_snapshot(execution, a_step, observed_ns):
                                    progress_signature = signature
                                    progress_frame = (primary, execution)
                                    if submission_trace:
                                        record(
                                            core.scheduler,
                                            "predictor_submission_progress",
                                            step=step,
                                            predecessor_step=a_step,
                                            observed_ns=observed_ns,
                                            primary_snapshot=primary,
                                            execution_snapshot=execution,
                                            forecast_load_features=timing.loads.forecast_features(
                                                data, observed_ns
                                            ),
                                        )
                    if timing and a_step and predicted_deadline_ns is None:
                        data = timing.steps.get(a_step)
                        records = timing.read_execution(a_step)
                        if (
                            data
                            and len(records) == world
                            and valid_snapshot(records, a_step, time.monotonic_ns())
                        ):
                            model = getattr(
                                timing.seal_estimator, "latency_surface", None
                            )
                            prediction_data = dict(
                                data,
                                native_predictor=getattr(
                                    core.scheduler, "_late_seal_native_predictor", False
                                ),
                            )
                            forecast_ns = time.monotonic_ns()
                            if model is not None and model.requires_forecast:
                                prediction_data = dict(
                                    prediction_data,
                                    execution_snapshot=records,
                                    forecast_ns=forecast_ns,
                                    forecast_load_features=(
                                        timing.loads.forecast_features(
                                            data, forecast_ns
                                        )
                                        if model.requires_load_progress
                                        else None
                                    ),
                                )
                            prediction_begin_ns = time.monotonic_ns()
                            if model is not None:
                                predicted, point_features = model.predict_with_features(
                                    prediction_data
                                )
                            else:
                                predicted, point_features = None, None
                            if predicted is None:
                                predicted = timing.seal_estimator.predict(
                                    prediction_data
                                )
                            prediction_cost_ns = (
                                time.monotonic_ns() - prediction_begin_ns
                            )
                            if predicted:
                                predicted = (
                                    predicted
                                    * core.scheduler._late_seal_prediction_scale_percent
                                    // 100
                                )
                                raw_duration_ns = predicted
                                applied_bias_ns = prediction_bias_us * 1000
                                shape_calibration = getattr(
                                    model, "duration_calibration", None
                                )
                                calibration_fallback = None
                                if shape_calibration is not None:
                                    applied_bias_ns, calibration_fallback = (
                                        shape_calibration.bias_ns(
                                            predicted, point_features, applied_bias_ns
                                        )
                                    )
                                predicted = max(1, predicted - applied_bias_ns)
                                reserve_ns = timing.seal_estimator.preparation_reserve(
                                    core.scheduler._late_seal_prediction_reserve_us
                                    * 1000
                                )
                                predicted_deadline_ns = (
                                    max(r[3] for r in records) + predicted - reserve_ns
                                )
                                prediction = dict(
                                    predecessor_step=a_step,
                                    gpu_start_upper_ns=max(r[3] for r in records),
                                    predicted_duration_ns=predicted,
                                    raw_scaled_duration_ns=raw_duration_ns,
                                    prediction_bias_us=prediction_bias_us,
                                    applied_prediction_bias_ns=applied_bias_ns,
                                    shape_calibration_enabled=shape_calibration
                                    is not None,
                                    shape_calibration_fallback=calibration_fallback,
                                    deadline_ns=predicted_deadline_ns,
                                    features=timing.seal_estimator.features(
                                        data
                                    ).tolist(),
                                    training_samples=len(timing.seal_estimator.samples),
                                    preparation_samples=len(
                                        timing.seal_estimator.preparation_ns
                                    ),
                                    preparation_reserve_ns=reserve_ns,
                                    prediction_cost_ns=prediction_cost_ns,
                                    predictor_backend=(
                                        getattr(model, "backend", "native")
                                        if prediction_data["native_predictor"]
                                        else "numpy"
                                    )
                                    if point_features is not None
                                    else "legacy",
                                    predictor_target=(
                                        model.target_schema
                                        if model is not None
                                        and point_features is not None
                                        else "cuda_span_max.v1"
                                    ),
                                    predictor_input=(
                                        point_features.tolist()
                                        if point_features is not None
                                        else None
                                    ),
                                    forecast_ns=forecast_ns,
                                    predictor_architecture=getattr(
                                        model, "architecture", "trees"
                                    ),
                                    predictor_excludes_elapsed=bool(
                                        getattr(model, "excludes_elapsed", False)
                                    ),
                                    execution_snapshot=records,
                                )
                                record(
                                    core.scheduler,
                                    "seal_prediction",
                                    step=step,
                                    **prediction,
                                )
                    if (
                        has_compute
                        and prediction
                        and (compute_deadline or compute_deadline_shadow)
                        and compute_seal_deadline_ns is None
                    ):
                        compute_seal_deadline_ns = (
                            prediction["gpu_start_upper_ns"]
                            + prediction["predicted_duration_ns"]
                            - compute_deadline_lead_us * 1000
                        )
                        compute_seal_deadline_armed_ns = time.monotonic_ns()
                        effective_seal_expiry_ns = min(
                            compute_seal_deadline_ns,
                            base_ready_ns + core.scheduler._late_seal_safety_us * 1000,
                        )
                        record(
                            core.scheduler,
                            "compute_seal_deadline_armed",
                            step=step,
                            deadline_ns=compute_seal_deadline_ns,
                            effective_expiry_ns=effective_seal_expiry_ns,
                            armed_ns=compute_seal_deadline_armed_ns,
                            lead_us=compute_deadline_lead_us,
                            shadow=compute_deadline_shadow,
                        )
                    if (
                        refinement_model is not None
                        and timing is not None
                        and prediction is not None
                        and refinement is None
                        and progress_frame is not None
                        and compute_seal_deadline_ns is not None
                        and effective_seal_expiry_ns is not None
                        and time.monotonic_ns() < effective_seal_expiry_ns
                    ):
                        primary, execution = progress_frame
                        if all(p[3] > 0 for p in primary):
                            forecast_ns = time.monotonic_ns()
                            refined_data = dict(
                                timing.steps[a_step],
                                native_predictor=getattr(
                                    core.scheduler, "_late_seal_native_predictor", False
                                ),
                                primary_snapshot=primary,
                                execution_snapshot=execution,
                                forecast_ns=forecast_ns,
                                forecast_load_features=timing.loads.forecast_features(
                                    timing.steps[a_step], forecast_ns
                                ),
                            )
                            begin_ns = time.monotonic_ns()
                            duration = refinement_model.predict(refined_data)
                            cost_ns = time.monotonic_ns() - begin_ns
                            if (
                                duration
                                and time.monotonic_ns() < effective_seal_expiry_ns
                            ):
                                duration = (
                                    duration
                                    * core.scheduler._late_seal_prediction_scale_percent
                                    // 100
                                )
                                raw_duration_ns = duration
                                duration = max(1, duration - prediction_bias_us * 1000)
                                old_expiry = effective_seal_expiry_ns
                                compute_seal_deadline_ns = (
                                    max(r[3] for r in execution)
                                    + duration
                                    - compute_deadline_lead_us * 1000
                                )
                                effective_seal_expiry_ns = min(
                                    compute_seal_deadline_ns,
                                    base_ready_ns
                                    + core.scheduler._late_seal_safety_us * 1000,
                                )
                                predicted_deadline_ns = (
                                    max(r[3] for r in execution)
                                    + duration
                                    - prediction["preparation_reserve_ns"]
                                )
                                refinement = dict(
                                    predecessor_step=a_step,
                                    forecast_ns=forecast_ns,
                                    predicted_duration_ns=duration,
                                    raw_scaled_duration_ns=raw_duration_ns,
                                    prediction_bias_us=prediction_bias_us,
                                    previous_expiry_ns=old_expiry,
                                    effective_expiry_ns=effective_seal_expiry_ns,
                                    primary_snapshot=primary,
                                    execution_snapshot=execution,
                                    predictor_input=refinement_model.input_features(
                                        refined_data
                                    ).tolist(),
                                    prediction_cost_ns=cost_ns,
                                )
                                record(
                                    core.scheduler,
                                    "seal_prediction_refined",
                                    step=step,
                                    **refinement,
                                )
                    if (
                        prediction
                        and timing is not None
                        and getattr(
                            core.scheduler, "_late_seal_condition_preparation", False
                        )
                    ):
                        upcoming_shapes = list(
                            getattr(core.scheduler, "_late_base_request_shapes", [])
                        )
                        for req_id in ready_acked:
                            request = core.scheduler.requests.get(req_id)
                            if request is not None:
                                upcoming_shapes.append(
                                    [
                                        request.num_tokens
                                        - request.num_computed_tokens,
                                        request.num_computed_tokens,
                                    ]
                                )
                        if upcoming_shapes != prediction.get("upcoming_shapes"):
                            reserve_method = timing.seal_estimator.preparation_reserve
                            if getattr(
                                core.scheduler, "_late_seal_recent_preparation", False
                            ):
                                reserve_method = (
                                    timing.seal_estimator.recent_preparation_reserve
                                )
                            reserve_ns = reserve_method(
                                core.scheduler._late_seal_prediction_reserve_us * 1000,
                                {"request_shapes": upcoming_shapes},
                            )
                            predicted_deadline_ns = (
                                prediction["gpu_start_upper_ns"]
                                + prediction["predicted_duration_ns"]
                                - reserve_ns
                            )
                            prediction.update(
                                deadline_ns=predicted_deadline_ns,
                                preparation_reserve_ns=reserve_ns,
                                upcoming_shapes=upcoming_shapes,
                                preparation_conditioned=True,
                                preparation_recent=getattr(
                                    core.scheduler,
                                    "_late_seal_recent_preparation",
                                    False,
                                ),
                            )
                            record(
                                core.scheduler,
                                "seal_preparation_prediction",
                                step=step,
                                upcoming_shapes=upcoming_shapes,
                                preparation_reserve_ns=reserve_ns,
                                deadline_ns=predicted_deadline_ns,
                            )
                    if (
                        not prediction_to_readiness
                        and predicted_deadline_ns
                        and now_ns >= predicted_deadline_ns
                    ):
                        seal_reason = "predicted_compute_deadline"
                        break
                    if has_compute:
                        if (
                            prediction_to_readiness
                            and len(set(ready_acked) & active) >= target
                        ):
                            seal_reason = "candidate_target_ready"
                            break
                        if (
                            seal_when_all_ready
                            and candidates
                            and len(ready_acked) == len(candidates)
                        ):
                            seal_reason = "all_candidates_ready"
                            break
                        if (
                            compute_deadline
                            and compute_seal_deadline_ns is not None
                            and effective_seal_expiry_ns is not None
                            and time.monotonic_ns() >= effective_seal_expiry_ns
                        ):
                            seal_reason = (
                                "compute_seal_deadline"
                                if compute_seal_deadline_ns <= effective_seal_expiry_ns
                                else "adaptive_safety_deadline"
                            )
                            break
                        if (
                            now_ns
                            >= base_ready_ns
                            + core.scheduler._late_seal_safety_us * 1000
                        ):
                            seal_reason = "adaptive_safety_deadline"
                            break
                        adaptive_sleep(predicted_deadline_ns)
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
            time.sleep(getattr(core.scheduler, "_late_seal_poll_us", 25) / 1e6)
        if wakeup is not None and notifications is None:
            wakeup.close()
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
            repeated_intervals=repeated_intervals,
            ready_target=(
                target
                if (policy == "event" or prediction_to_readiness) and candidates
                else None
            ),
            load_only_pending_candidates=len(candidates),
            load_only_ready_percent=load_only_ready_percent,
            load_only_fraction_target=load_only_fraction_target,
            has_compute=(
                has_compute
                if (policy == "event" or prediction_to_readiness) and candidates
                else None
            ),
            prediction=prediction,
            refinement=refinement,
            adaptive_poll_us=adaptive_poll_us,
            progress_polls=progress_polls,
            adaptive_poll_cpu_ns=poll_cpu_ns,
            adaptive_sleep_ns=sleep_ns,
            adaptive_sleeps=adaptive_sleeps,
            adaptive_requested_sleep_ns=requested_sleep_ns,
            hybrid_coarse_sleeps=coarse_sleeps,
            hybrid_wait_us=hybrid_wait_us,
            hybrid_guard_us=hybrid_guard_us,
            prediction_to_readiness=prediction_to_readiness,
            compute_seal_deadline_enabled=compute_deadline,
            compute_seal_deadline_shadow=compute_deadline_shadow,
            compute_seal_deadline_lead_us=compute_deadline_lead_us,
            compute_seal_deadline_ns=compute_seal_deadline_ns,
            compute_seal_deadline_armed_ns=compute_seal_deadline_armed_ns,
            effective_seal_expiry_ns=effective_seal_expiry_ns,
            compute_seal_deadline_overshoot_ns=(
                max(0, sealed_ns - effective_seal_expiry_ns)
                if effective_seal_expiry_ns is not None
                else None
            ),
            hybrid_transition_ns=hybrid_transition_ns,
            hybrid_polling_ns=hybrid_polling_ns,
            full_prediction_wait=full_prediction_wait,
            coalesced_notifications=coalesced,
            progress_bridge_cpu_ns=wakeup.cpu_ns if wakeup else 0,
            progress_bridge_scans=wakeup.scans if wakeup else 0,
            progress_bridge_notifications=wakeup.notifications if wakeup else 0,
            adaptive_waits=waits,
        )
        if getattr(core.scheduler, "_late_cached_requests", False):
            key = (core.scheduler._late_generation, step)
            core.scheduler._late_reference_requests = {r.req_id: (key, r) for r in base}
            core.scheduler._late_reference_requests.update(
                (c.req_id, (key, c))
                for slot, c in enumerate(candidates)
                if all(
                    s == 1 for s in statuses[world * (slot + 1) : world * (slot + 2)]
                )
            )
        return [
            c.req_id
            for slot, c in enumerate(candidates)
            if all(s == 1 for s in statuses[world * (slot + 1) : world * (slot + 2)])
        ]
    finally:
        if notifications_armed:
            assert notifications is not None
            notifications.disarm()
            with suppress(FileNotFoundError):
                os.unlink(path + ".seal")
        elif wakeup is not None and not wakeup.closed:
            wakeup.close()
        os.close(fd)
        os.unlink(path)
        os.unlink(data_path)


def cache_references(scheduler, requests):
    """Strip prompts only for the all-rank ACK snapshot and unchanged data."""
    cached = getattr(scheduler, "_late_reference_requests", {})
    result = []
    for request in requests:
        entry = cached.get(request.req_id)
        if entry is not None:
            key, source = entry
            if (
                request.prompt_token_ids == source.prompt_token_ids
                and request.prefill_token_ids == source.prefill_token_ids
            ):
                request = replace(
                    request,
                    prompt_token_ids=None,
                    prefill_token_ids=None,
                    late_cached_key=key,
                )
        result.append(request)
    references = [r.req_id for r in result if r.late_cached_key is not None]
    if references:
        record(scheduler, "cached_request_references", references=references)
    return result


def resolve_references(output, prepared, generation):
    """Hydrate private prompt data, preserving the final allocation fields."""
    resolved = []
    for request in output.scheduled_new_reqs:
        key = request.late_cached_key
        if key is not None:
            data = prepared.get(request.req_id)
            if (
                key != (generation, output.readiness_timeline_step)
                or data is None
                or data.cache_key != key
                or data.source is None
            ):
                raise RuntimeError("acknowledged cached request is missing or stale")
            request = replace(
                request,
                prompt_token_ids=data.source.prompt_token_ids
                if data.source.prompt_token_ids is not None
                else data.token_ids,
                prefill_token_ids=data.source.prefill_token_ids
                if data.source.prefill_token_ids is not None
                else data.token_ids,
                late_cached_key=None,
            )
        else:
            data = prepared.get(request.req_id)
            if data is not None and data.source is not None:
                source = data.source
                matches = (
                    np.array_equal(request.prompt_token_ids, data.token_ids)
                    and np.array_equal(request.prefill_token_ids, data.token_ids)
                    if source.prefill_token_ids is None
                    else request.prompt_token_ids == source.prompt_token_ids
                    and request.prefill_token_ids == source.prefill_token_ids
                )
                if not matches:
                    prepared.pop(request.req_id)
        resolved.append(request)
    output.scheduled_new_reqs = resolved


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
    for key in ("candidates", "staged", "eligible", "activated", "references"):
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
