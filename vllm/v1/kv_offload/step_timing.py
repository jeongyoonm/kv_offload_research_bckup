# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate, bounded per-rank timing transport for the deferral experiment.

CUDA events are queried on a background thread. Completion is an interval
between the last unsuccessful query and the first successful query, never the
poll timestamp treated as an exact device timestamp. flock protects cross-
process snapshots; a local lock also serializes each writer's threads.
"""

import contextlib
import fcntl
import functools
import hashlib
import json
import mmap
import os
import struct
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from vllm.v1.kv_offload.overlap_predictor import OverlapPredictor
from vllm.v1.kv_offload.scheduling_deferral import Hold, Predictor

CAPACITY = 4096
RECORD = struct.Struct("9Q")
# step, entry, prepared, enqueued, lower, upper, observed, status, rpc_ready
PATH_KEY = "_deferral_timing_paths"
STALE_NS = 10_000_000
# Separate liveness limit for queued/preparing work, not a scheduling budget.
PENDING_NS = 1_000_000_000
REGISTERED = 2


def primary_record_reason(r, dispatched, now):
    """Preserve the primary channel's validity rules and expose their cause."""
    if r is None:
        return "missing_record"
    if not dispatched <= r[6] <= now:
        return "observation_clock"
    pending = 0 <= now - dispatched <= PENDING_NS
    if r[7] == REGISTERED:
        if not pending:
            return "pending_expired"
        return "ok" if not any(r[1:6]) and r[6] == dispatched else "registered_fields"
    if r[7] != 1:
        return "worker_failed"
    if not dispatched <= r[1] <= r[6] or r[8] > r[1]:
        return "entry_order"
    if not r[3]:
        if not pending:
            return "pending_expired"
        return (
            "ok"
            if not any(r[4:6]) and (r[2] == 0 or r[1] <= r[2] <= r[6])
            else "preparation_order"
        )
    if not 0 < r[1] <= r[2] <= r[3] <= r[4] <= r[6]:
        return "completion_order"
    if r[5]:
        return "ok" if r[4] <= r[5] <= r[6] else "completion_interval"
    return "ok" if now - r[6] <= STALE_NS else "observer_stale"


def primary_read_state(records, reasons, previous, step, dispatched, now):
    """Busy reads can retry only against fresh evidence of the same generation."""
    if not records or len(records) != len(reasons):
        return "invalid"
    if any(
        reason == "ok"
        and (not r or r[0] != step or primary_record_reason(r, dispatched, now) != "ok")
        for r, reason in zip(records, reasons)
    ):
        return "invalid"
    if all(r == "ok" for r in reasons):
        return "ready"
    if (
        "lock_busy" not in reasons
        or any(r not in ("ok", "lock_busy") for r in reasons)
        or len(previous) != len(records)
        or any(
            not r or r[0] != step or primary_record_reason(r, dispatched, now) != "ok"
            for r in previous
        )
    ):
        return "invalid"
    return "retry"


class RankFile:
    def __init__(self, path: str, create: bool = False):
        self.fd = os.open(
            path, os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0), 0o600
        )
        if create:
            os.ftruncate(self.fd, CAPACITY * RECORD.size)
        self.mem = mmap.mmap(self.fd, CAPACITY * RECORD.size)
        self.lock = threading.Lock()

    def write(self, record: list[int]) -> None:
        with self.lock:
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            try:
                RECORD.pack_into(self.mem, record[0] % CAPACITY * RECORD.size, *record)
            finally:
                fcntl.flock(self.fd, fcntl.LOCK_UN)

    def read(self, step: int) -> tuple[int, ...] | None:
        return self.read_with_status(step)[0]

    def read_with_status(self, step: int):
        try:
            fcntl.flock(self.fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return None, "lock_busy"
        try:
            record = RECORD.unpack_from(self.mem, step % CAPACITY * RECORD.size)
            if record[0] != step or step <= 0:
                return None, "step_mismatch"
            return record, "ok"
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def close(self):
        self.mem.close()
        os.close(self.fd)


class TimingChannel:
    def __init__(self, config):
        self.directory = Path(os.environ["VLLM_DEFERRAL_TIMING_DIR"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.execution_config = (
            hashlib.sha256(
                (
                    config.compute_hash()
                    + os.environ.get("VLLM_DEFERRAL_EXECUTION_CONFIG", "")
                ).encode()
            ).hexdigest()
            if os.environ.get("VLLM_DEFERRAL_EXECUTION_CONFIG")
            else None
        )
        prefix = f"/dev/shm/vllm-step-{uuid.uuid4().hex}"
        self.paths = [
            f"{prefix}-{rank}" for rank in range(config.parallel_config.world_size)
        ]
        self.ranks = []
        self.execution_ranks: list[RankFile] = []
        try:
            for path in self.paths:
                self.ranks.append(RankFile(path, create=True))
                if os.environ.get("VLLM_DEFERRAL_EXECUTION_TIMING") == "1":
                    self.execution_ranks.append(
                        RankFile(path + ".execution", create=True)
                    )
        except BaseException:
            self.close()
            raise
        config.kv_transfer_config.kv_connector_extra_config[PATH_KEY] = self.paths
        self.events: list[dict] = []
        self.steps: dict[int, dict] = {}
        self.registrations: dict[int, tuple[int, int]] = {}
        self.predictor: Predictor | OverlapPredictor | None = None
        self.hold: Hold | None = None
        from vllm.v1.kv_offload.scheduling_deferral import OutstandingLoads

        self.loads = OutstandingLoads()
        self.pending_fill_results = []
        self.opportunity = 0
        self.cutoff: dict[str, bool] = {}
        self.watches: dict[int, dict] = {}
        self.audit = None
        if os.environ.get("VLLM_DEFERRAL_SHADOW_AUDIT") == "1":
            from vllm.v1.kv_offload.shadow_timing_audit import ShadowTimingAudit

            self.audit = ShadowTimingAudit(self.paths)

    def emit(self, event: str, **fields):
        self.events.append(dict(event=event, time_ns=time.monotonic_ns(), **fields))

    def watch(self, predecessor, opportunity, now, snapshot):
        audit = getattr(self, "audit", None)
        if audit is not None:
            data = self.steps.get(predecessor)
            if data is None:
                registration = self.registrations.get(predecessor % CAPACITY)
                if registration is not None and registration[0] == predecessor:
                    valid, complete, _ = self.read(predecessor, now)
                    if valid and complete:
                        data = dict(
                            step=predecessor,
                            dispatch_ns=registration[1],
                            completed_before_watch=True,
                        )
            if data is None:
                self.emit(
                    "timing_audit_unavailable",
                    opportunity_id=opportunity,
                    reason="predecessor_data_already_collected",
                )
            else:
                audit.begin(opportunity, now, data, snapshot)
        if len(self.watches) >= CAPACITY:
            self.emit("readiness_watch_overflow", opportunity_id=opportunity)
            return
        self.watches[opportunity] = dict(
            predecessor=predecessor,
            pending={key: now for key, ready in snapshot.items() if not ready},
        )

    def observe(self, now, snapshot, live_ids):
        for opportunity, watch in list(self.watches.items()):
            for key, lower in list(watch["pending"].items()):
                reason = None
                if snapshot.get(key):
                    reason = "all_rank_ready"
                elif key not in live_ids:
                    reason = "cancelled_or_finished"
                if reason:
                    self.emit(
                        "readiness_observation",
                        opportunity_id=opportunity,
                        predecessor=watch["predecessor"],
                        request=key,
                        lower_ns=lower,
                        upper_ns=now,
                        reason=reason,
                    )
                    del watch["pending"][key]
                else:
                    watch["pending"][key] = now
            if not watch["pending"]:
                del self.watches[opportunity]

    def observe_selection(self, selected, now, successor):
        for opportunity, watch in list(self.watches.items()):
            for key in set(selected).intersection(watch["pending"]):
                self.emit(
                    "readiness_observation",
                    opportunity_id=opportunity,
                    predecessor=watch["predecessor"],
                    request=key,
                    lower_ns=watch["pending"].pop(key),
                    upper_ns=now,
                    reason="selected_before_observation",
                    successor=successor,
                )
            if not watch["pending"]:
                del self.watches[opportunity]

    def register(self, data: dict) -> bool:
        """Publish pending records before dispatch can reach any worker."""
        step, dispatched = data["step"], data["dispatch_ns"]
        try:
            for rank in self.ranks:
                rank.write([step, 0, 0, 0, 0, 0, dispatched, REGISTERED, 0])
            for rank in getattr(self, "execution_ranks", []):
                rank.write([step, 0, 0, 0, dispatched, REGISTERED, 0, 0, 0])
        except (OSError, ValueError):
            return False
        self.steps[step] = data
        self.registrations[step % CAPACITY] = (step, dispatched)
        return True

    def read(self, step: int, now: int | None = None):
        registration = self.registrations.get(step % CAPACITY)
        if registration is None or registration[0] != step:
            return False, False, []
        dispatched = registration[1]
        try:
            records = [rank.read(step) for rank in self.ranks]
        except (OSError, ValueError):
            return False, False, []

        if now is None:
            now = time.monotonic_ns()

        valid = all(primary_record_reason(r, dispatched, now) == "ok" for r in records)
        return valid, valid and all(r is not None and r[5] for r in records), records

    def read_for_deferral(self, step, previous):
        registration = self.registrations.get(step % CAPACITY)
        if registration is None or registration[0] != step:
            return "invalid", False, [], ["registration_mismatch"]
        records, reasons = [], []
        for rank in self.ranks:
            try:
                record, reason = rank.read_with_status(step)
            except (OSError, ValueError):
                record, reason = None, "read_error"
            records.append(record)
            reasons.append(reason)
        now = time.monotonic_ns()
        reasons = [
            primary_record_reason(r, registration[1], now) if s == "ok" else s
            for r, s in zip(records, reasons)
        ]
        state = primary_read_state(
            records, reasons, previous, step, registration[1], now
        )
        complete = state == "ready" and all(r[5] for r in records)
        return state, complete, records, reasons

    def collect(self):
        audit = getattr(self, "audit", None)
        if audit is not None:
            self.events.extend(audit.drain())
        now = time.monotonic_ns()
        for step, data in list(self.steps.items()):
            valid, complete, records = self.read(step)
            execution = self.read_execution(step)
            if (
                complete
                and execution
                and not all(r and r[8] for r in execution)
                and now - max(r[5] for r in records) < 100_000_000
            ):
                continue
            if complete or now - data["dispatch_ns"] > 60_000_000_000:
                self.emit(
                    "step_completion",
                    **data,
                    valid=bool(valid and complete),
                    ranks=records,
                    execution_ranks=execution,
                )
                for opportunity, watch in list(getattr(self, "watches", {}).items()):
                    if watch["predecessor"] == step:
                        self.emit(
                            "readiness_watch_end",
                            opportunity_id=opportunity,
                            predecessor=step,
                            unobserved=watch["pending"],
                            completion_valid=bool(valid and complete),
                        )
                        del self.watches[opportunity]
                del self.steps[step]

    def read_execution(self, step):
        try:
            return [r.read(step) for r in getattr(self, "execution_ranks", [])]
        except (OSError, ValueError):
            return []

    def read_execution_with_status(self, step):
        records, statuses = [], []
        for rank in getattr(self, "execution_ranks", []):
            try:
                record, status = rank.read_with_status(step)
            except (OSError, ValueError):
                record, status = None, "read_error"
            records.append(record)
            statuses.append(status)
        return records, statuses

    def flush(self):
        self.collect()
        audit = getattr(self, "audit", None)
        if audit is not None:
            self.events.extend(audit.flush())
        path = self.directory / f"engine-{os.getpid()}.jsonl"
        with path.open("a") as stream:
            for event in self.events:
                stream.write(json.dumps(event) + "\n")
        self.events.clear()

    def close(self):
        audit = getattr(self, "audit", None)
        if audit is not None:
            audit.close()
        for rank in self.ranks:
            rank.close()
        for path in self.paths:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
        for rank in getattr(self, "execution_ranks", []):
            rank.close()
        for path in self.paths:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path + ".execution")


def timing_only(fn):
    """Telemetry errors disable this writer without masking model/KV errors."""

    @functools.wraps(fn)
    def wrapped(self, *args):
        if getattr(self, "failed", False):
            if fn.__name__ == "begin":
                with contextlib.suppress(Exception):
                    now = time.monotonic_ns()
                    self.rank.write([args[0], now, 0, 0, 0, 0, now, 0, 0])
            return
        try:
            return fn(self, *args)
        except Exception:
            self.failed = True
            records = [item[2] for item in getattr(self, "pending", ())]
            if getattr(self, "current", None) is not None:
                records.append(self.current)
            for record in records:
                record[7] = 0
                with contextlib.suppress(Exception):
                    self.rank.write(record)
            self.current = None

    return wrapped


class WorkerTiming:
    def __init__(self, path: str, device):
        from vllm.v1.kv_offload.execution_timing import (
            ExecutionObserver,
            NativeExecutionObserver,
            UnavailableExecutionObserver,
        )

        self.rank = RankFile(path)
        self.device = device
        self.pending: deque[tuple[Any, Any, list[int]]] = deque()
        self.wakeup = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.current: list[int] | None = None
        self.execution: (
            ExecutionObserver
            | NativeExecutionObserver
            | UnavailableExecutionObserver
            | None
        ) = None
        if os.path.exists(path + ".execution"):
            try:
                library = os.environ.get("VLLM_DEFERRAL_NATIVE_OBSERVER")
                self.execution = (
                    NativeExecutionObserver(path + ".execution", device, library)
                    if library
                    else ExecutionObserver(path + ".execution", device)
                )
            except Exception:
                self.execution = UnavailableExecutionObserver(path + ".execution")
        self.thread.start()

    @timing_only
    def begin(self, step: int):
        now = time.monotonic_ns()
        self.current = [step, now, 0, 0, 0, 0, now, 1, getattr(self, "rpc_ready_ns", 0)]
        self.rank.write(self.current)

    @timing_only
    def prepared(self):
        if self.current is not None:
            self.current[2] = self.current[6] = time.monotonic_ns()
            self.rank.write(self.current)
            if self.execution is not None:
                self.execution.start(self.current[0])

    @timing_only
    def finish(self, output):
        import torch

        record = self.current
        if record is None:
            return
        # Join required output-copy work on a separate stream; never introduce
        # a timing-induced dependency on the model's compute stream.
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        copy_event = getattr(output, "async_copy_ready_event", None)
        if copy_event is None:
            copy_event = getattr(output, "copy_event", None)
        if copy_event is not None:
            stream.wait_event(copy_event)
        event = torch.cuda.Event(enable_timing=self.execution is not None)
        record[3] = record[4] = time.monotonic_ns()
        event.record(stream)
        if self.execution is not None:
            self.execution.finish(record[0], event, record[3], stream=stream)
        record[6] = time.monotonic_ns()
        self.rank.write(record)
        self.pending.append((event, stream, record))
        self.current = None
        self.wakeup.set()

    @timing_only
    def _poll(self):
        import torch

        torch.accelerator.set_device_index(self.device)
        while True:
            self.wakeup.wait()
            while self.pending:
                event, stream, record = self.pending[0]
                before = time.monotonic_ns()
                try:
                    complete = event.query()
                except Exception:
                    record[7] = 0
                    complete = True
                after = time.monotonic_ns()
                if complete:
                    record[5] = after
                else:
                    record[4] = before
                record[6] = after
                self.rank.write(record)
                if complete:
                    self.pending.popleft()
                else:
                    time.sleep(0.0005)
            self.wakeup.clear()
            if self.pending:
                self.wakeup.set()
