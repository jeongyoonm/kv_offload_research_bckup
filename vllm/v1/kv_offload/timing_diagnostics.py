# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicitly enabled, drained-engine timing interference experiment.

Markers are common to the ablations, bounded, and exported outside replay.
Diagnostic failures invalidate a cell; they never change KV completion semantics.
"""

import ctypes
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from vllm.v1.kv_offload.step_timing import WorkerTiming

MODES = {"legacy", "baseline", "host", "record", "full", "full_retained"}
MARKER_CAPACITY = 1_000_000
EVENT_CAPACITY = 4096
_active = False
_events: list[dict[str, Any]] = []
_lock = threading.Lock()
_generation = 0
_overflow = False


def validate_mode(options):
    if options.get("load_only_wait_audit"):
        if os.environ.get("VLLM_LOAD_ONLY_WAIT_AUDIT") != "1":
            raise ValueError("load-only wait audit requires development opt-in")
        if options["timing_diagnostic_mode"] != "legacy":
            raise ValueError("load-only wait audit excludes timing ablation modes")
    mode = options["timing_diagnostic_mode"]
    action = options["timing_replay_action"]
    if action not in ("none", "capture", "save", "replay"):
        raise ValueError("invalid controlled replay action")
    if action != "none" and mode == "legacy":
        raise ValueError("controlled replay requires diagnostic mode")
    if action in ("capture", "save") and mode != "baseline":
        raise ValueError("controlled capture requires baseline mode")
    if mode not in MODES:
        raise ValueError("invalid timing diagnostic mode")
    if mode == "legacy":
        if options["timing_diagnostic_markers"]:
            raise ValueError("diagnostic markers require a diagnostic mode")
        return
    if os.environ.get("VLLM_TIMING_DIAGNOSIS") != "1":
        raise ValueError("timing diagnosis requires explicit development opt-in")
    expected = "disabled" if mode == "baseline" else "timing"
    if options["deferral_policy"] != expected:
        raise ValueError("timing diagnosis excludes holds and mismatched policies")
    if options["optimistic_cap"] or options["background_prepare"]:
        raise ValueError("timing diagnosis excludes other experimental scheduling")


def enabled():
    return _active


def emit(event, **fields):
    global _overflow
    if not _active:
        return
    record = dict(
        event=event,
        time_ns=time.monotonic_ns(),
        pid=os.getpid(),
        tid=threading.get_native_id(),
        generation=_generation,
        **fields,
    )
    with _lock:
        if len(_events) >= MARKER_CAPACITY:
            _overflow = True
        else:
            _events.append(record)


def configure(markers, generation):
    global _active, _generation
    if _events or _overflow:
        raise RuntimeError("diagnostic markers must be exported before reconfiguration")
    _generation = generation
    _active = markers
    emit("collector_activation")


def export():
    global _events, _overflow
    emit("export_begin")
    with _lock:
        records, _events = _events, []
        overflow, _overflow = _overflow, False
    directory = Path(os.environ["VLLM_DEFERRAL_TIMING_DIR"]) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{os.getpid()}-{_generation}-{time.monotonic_ns()}.json"
    result = dict(
        generation=_generation,
        pid=os.getpid(),
        overflow=overflow,
        export_ns=time.monotonic_ns(),
        events=records,
    )
    with path.open("x") as stream:
        json.dump(result, stream)
    if overflow:
        raise RuntimeError(f"diagnostic marker overflow: {path}")
    return str(path)


class DiagnosticWorkerTiming(WorkerTiming):
    """Same primary event/stream operations as WorkerTiming, selectable queries."""

    def __init__(self, path, device, mode):
        from vllm.v1.kv_offload.execution_timing import NativeExecutionObserver
        from vllm.v1.kv_offload.step_timing import RankFile

        self.rank = RankFile(path)
        self.device, self.mode = device, mode
        self.pending: deque[tuple[Any, Any, list[int]]] = deque()
        self.retained = []
        self.records = []
        self.current = None
        self.failed = False
        self.stopping = False
        self.wakeup = threading.Event()
        self.poller: threading.Thread | None = None
        self.execution = None
        self.query_count = 0
        self.first_sample_ns = self.last_sample_ns = self.max_gap_ns = 0
        self.first_latency_ns = []
        self.pool = []
        self.pool_index = 0
        if mode != "host":
            # Construct through the existing wrapper, replacing its handle only
            # for the explicitly selected bounded-retention diagnostic variants.
            self.execution = NativeExecutionObserver(
                path + ".execution", device, os.environ["VLLM_DEFERRAL_NATIVE_OBSERVER"]
            )
            lib = self.execution.lib
            lib.timing_create_mode.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.timing_create_mode.restype = ctypes.c_void_p
            lib.timing_drain.argtypes = [ctypes.c_void_p]
            lib.timing_drain.restype = ctypes.c_int
            lib.timing_snapshot.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
            ]
            lib.timing_snapshot.restype = ctypes.c_int
            lib.timing_audit_copy.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
            ]
            lib.timing_audit_copy.restype = ctypes.c_size_t
            if mode in ("record", "full_retained"):
                self.execution.close()
                self.execution.handle = lib.timing_create_mode(
                    (path + ".execution").encode(),
                    device.index,
                    1 if mode == "record" else 2,
                )
                if not self.execution.handle:
                    raise RuntimeError("native diagnostic initialization failed")
                import torch

                self.pool = [
                    torch.cuda.Event(enable_timing=True) for _ in range(EVENT_CAPACITY)
                ]
                # PyTorch events initialize lazily; force allocation before replay.
                for event in self.pool:
                    event.record(torch.cuda.current_stream(device))
                torch.accelerator.synchronize(device)
        if mode in ("full", "full_retained"):
            self.poller = threading.Thread(target=self._poll, daemon=True)
            self.poller.start()

    def begin(self, step):
        if self.failed or self.current is not None:
            raise RuntimeError("failed or overlapping diagnostic step")
        if len(self.records) >= EVENT_CAPACITY:
            raise RuntimeError("diagnostic event capacity exceeded")
        now = time.monotonic_ns()
        self.current = [step, now, 0, 0, 0, 0, now, 1, getattr(self, "rpc_ready_ns", 0)]
        self.records.append(self.current)
        self.rank.write(self.current)

    def prepared(self):
        if self.current is not None:
            self.current[2] = self.current[6] = time.monotonic_ns()
            self.rank.write(self.current)
            if self.execution is not None:
                self.execution.start(self.current[0])

    def finish(self, output):
        import torch

        record = self.current
        if record is None:
            return
        if self.mode == "host":
            record[3] = record[6] = time.monotonic_ns()
            self.rank.write(record)
        else:
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            copy_event = getattr(output, "async_copy_ready_event", None)
            if copy_event is None:
                copy_event = getattr(output, "copy_event", None)
            if copy_event is not None:
                stream.wait_event(copy_event)
            if self.pool:
                event = self.pool[self.pool_index]
                self.pool_index += 1
            else:
                event = torch.cuda.Event(enable_timing=True)
            record[3] = record[4] = time.monotonic_ns()
            event.record(stream)
            assert self.execution is not None
            self.execution.finish(record[0], event, record[3], stream=stream)
            record[6] = time.monotonic_ns()
            self.rank.write(record)
            item = (event, stream, record)
            if self.mode in ("record", "full_retained"):
                self.retained.append(item)
            if self.mode != "record":
                self.pending.append(item)
                self.wakeup.set()
        self.current = None

    def _poll(self):
        import torch

        torch.accelerator.set_device_index(self.device)
        try:
            while not self.stopping:
                self.wakeup.wait()
                while self.pending and not self.stopping:
                    event, stream, record = self.pending[0]
                    before = time.monotonic_ns()
                    if not self.first_sample_ns:
                        self.first_sample_ns = before
                    if self.last_sample_ns:
                        self.max_gap_ns = max(
                            self.max_gap_ns, before - self.last_sample_ns
                        )
                    if record[4] == record[3]:
                        self.first_latency_ns.append(before - record[3])
                    self.last_sample_ns = before
                    self.query_count += 1
                    complete = event.query()
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
        except Exception:
            self.failed = True

    def drain_close(self):
        import torch

        torch.accelerator.synchronize(self.device)
        deadline = time.monotonic() + 10
        while self.pending and not self.failed and time.monotonic() < deadline:
            self.wakeup.set()
            time.sleep(0.001)
        self.stopping = True
        self.wakeup.set()
        if self.poller is not None:
            self.poller.join(timeout=10)
            if self.poller.is_alive():
                raise RuntimeError("diagnostic poller failed to stop")
        native = None
        if self.execution is not None:
            from vllm.v1.kv_offload.execution_timing import NativeExecutionObserver

            assert isinstance(self.execution, NativeExecutionObserver)
            lib, handle = self.execution.lib, self.execution.handle
            stats = (ctypes.c_uint64 * 8)()
            if self.mode != "record":
                while time.monotonic() < deadline:
                    lib.timing_snapshot(handle, stats)
                    if not stats[5] or stats[4]:
                        break
                    time.sleep(0.001)
                if stats[5]:
                    self.failed = True
            success = lib.timing_drain(handle)
            success &= lib.timing_snapshot(handle, stats)
            native = dict(
                zip(
                    (
                        "queries",
                        "first_sample_ns",
                        "last_sample_ns",
                        "max_gap_ns",
                        "failed",
                        "pending",
                        "retained",
                        "reusable",
                    ),
                    stats,
                )
            )
            native["offline_drain"] = self.mode == "record"
            audit = (ctypes.c_uint64 * (22 * EVENT_CAPACITY))()
            count = lib.timing_audit_copy(handle, audit, len(audit))
            native["audit"] = [list(audit[i * 22 : (i + 1) * 22]) for i in range(count)]
            if count != len(self.records):
                self.failed = True
            self.execution.close()
            if not success:
                self.failed = True
        result = dict(
            mode=self.mode,
            records=self.records,
            native=native,
            queries=self.query_count,
            first_sample_ns=self.first_sample_ns,
            max_gap_ns=self.max_gap_ns,
            first_latency_ns=self.first_latency_ns,
            failed=self.failed,
            pending=len(self.pending),
        )
        self.rank.close()
        self.retained.clear()
        self.pool.clear()
        if self.failed or self.pending or self.current is not None:
            raise RuntimeError("diagnostic timing failed or did not drain")
        return result


def batch_fields(output, future=None, execute_future=None):
    """Describe existing queue state without probing or resolving futures."""
    metadata = output.kv_connector_metadata
    loads = list(getattr(metadata, "load_jobs", {}))
    stores = list(getattr(metadata, "store_jobs", {}))
    tokens = output.total_num_scheduled_tokens
    kind = (
        "compute_transfers"
        if tokens and (loads or stores)
        else (
            "compute"
            if tokens
            else "load_store"
            if loads and stores
            else ("load_only" if loads else "store_only" if stores else "empty")
        )
    )
    return dict(
        step=output.readiness_timeline_step,
        kind=kind,
        scheduled_tokens=tokens,
        load_jobs=loads,
        store_jobs=stores,
        future_rpc=getattr(future, "diagnostic_rpc", None),
        execute_rpc=getattr(execute_future, "diagnostic_rpc", None),
    )


def queue_event(event, queue, capacity, **fields):
    if enabled():
        emit(
            event,
            capacity=capacity,
            occupancy=len(queue),
            queue=[
                batch_fields(output, future, execute)
                for future, output, execute in queue
            ],
            **fields,
        )
