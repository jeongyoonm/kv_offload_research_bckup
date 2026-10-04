# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in GPU execution boundary transport, independent of KV and step timing."""

import contextlib
import threading
import time

# RankFile words: step, enqueued, start_lower, start_upper, observed,
# status (0 failed, 1 active, 2 registered), elapsed_ns, end_lower, end_upper.
MAX_START_UNCERTAINTY_NS = 2_000_000


def start_snapshot_state(records, step, now, ranks):
    """Distinguish a healthy pending start from failed or inconsistent telemetry."""
    if len(records) != ranks:
        return "invalid"
    pending = False
    for r in records:
        if r is None or len(r) != 9 or r[0] != step:
            return "invalid"
        if r[5] == 2:
            if any(r[k] for k in (1, 2, 3, 6, 7, 8)) or not 0 < r[4] <= now:
                return "invalid"
            pending = True
        elif r[5] == 1 and not r[3]:
            if not 0 < r[1] <= r[2] <= r[4] <= now or r[8]:
                return "invalid"
            pending = True
        elif not valid_snapshot([r], step, now):
            return "invalid"
    return "pending" if pending else "ready"


def start_read_state(records, statuses, step, now, ranks):
    """Retry only explicit contention; another rank's failure takes precedence."""
    if len(records) != ranks or len(statuses) != ranks or ranks <= 0:
        return "invalid"
    states = []
    for record, status in zip(records, statuses):
        if status == "lock_busy" and record is None:
            states.append("busy")
        elif status != "ok":
            return "invalid"
        else:
            states.append(start_snapshot_state([record], step, now, 1))
    if "invalid" in states:
        return "invalid"
    if "busy" in states:
        return "busy"
    return "pending" if "pending" in states else "ready"


def valid_snapshot(records, step, now, *, completed=False):
    if not records:
        return False
    for r in records:
        if r is None or len(r) != 9 or r[0] != step or r[5] != 1:
            return False
        if not 0 < r[1] <= r[2] <= r[3] <= r[4] <= now:
            return False
        if r[3] - r[2] > MAX_START_UNCERTAINTY_NS:
            return False
        if completed and not (r[6] > 0 and r[2] <= r[7] <= r[8] <= r[4]):
            return False
    return True


class ExecutionObserver:
    """Observe CUDA events without making the worker wait for GPU progress."""

    def __init__(self, path, device):
        from vllm.v1.kv_offload.step_timing import RankFile

        self.rank = RankFile(path)
        self.device = device
        self.jobs = {}
        self.lock = threading.Lock()
        self.wakeup = threading.Event()
        self.stopped = threading.Event()
        self.failed = False
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def _fail(self):
        self.failed = True
        for job in self.jobs.values():
            job["record"][5] = 0
            with contextlib.suppress(Exception):
                self.rank.write(job["record"])
        self.jobs.clear()

    def start(self, step):
        import torch

        with self.lock:
            if self.failed:
                return
            try:
                from vllm.v1.kv_offload.step_timing import CAPACITY

                if len(self.jobs) >= CAPACITY or step in self.jobs:
                    raise ValueError("execution timing overflow or duplicate step")
                event = torch.cuda.Event(enable_timing=True)
                before = time.monotonic_ns()
                event.record(torch.cuda.current_stream(self.device))
                record = [step, before, before, 0, time.monotonic_ns(), 1, 0, 0, 0]
                self.jobs[step] = dict(start=event, end=None, record=record)
                self.rank.write(record)
                self.wakeup.set()
            except Exception:
                self._fail()

    def finish(self, step, event, enqueued_ns, stream=None):
        with self.lock:
            if self.failed:
                return
            try:
                job = self.jobs[step]
                job["end"] = event
                job["record"][7] = enqueued_ns
                self.wakeup.set()
            except Exception:
                self._fail()

    def _poll_once(self):
        with self.lock:
            try:
                for step, job in list(self.jobs.items()):
                    r = job["record"]
                    if not r[3]:
                        before = time.monotonic_ns()
                        if job["start"].query():
                            r[3] = time.monotonic_ns()
                        else:
                            r[2] = before
                    if r[3] and job["end"] is not None:
                        before = time.monotonic_ns()
                        if job["end"].query():
                            r[8] = time.monotonic_ns()
                            r[6] = round(job["start"].elapsed_time(job["end"]) * 1e6)
                        else:
                            r[7] = before
                    r[4] = time.monotonic_ns()
                    self.rank.write(r)
                    if r[8]:
                        del self.jobs[step]
            except Exception:
                self._fail()
            return bool(self.jobs)

    def _poll(self):
        import torch

        try:
            torch.accelerator.set_device_index(self.device)
            while not self.stopped.is_set():
                self.wakeup.wait(0.1)
                if self.stopped.is_set():
                    break
                self.wakeup.clear()
                while self._poll_once() and not self.stopped.wait(0.0005):
                    pass
        except Exception:
            with self.lock:
                self._fail()

    def close(self):
        self.stopped.set()
        self.wakeup.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("execution observer did not stop")
        self.rank.close()


class NativeExecutionObserver:
    """Explicitly loaded experimental observer; never silently use another backend."""

    def __init__(self, path, device, library):
        import ctypes
        import os
        from pathlib import Path

        self.lib = ctypes.CDLL(library)
        if Path(path + ".seal").exists():
            abi = getattr(self.lib, "timing_seal_notification_abi", None)
            if abi is None or abi() != 1:
                raise ValueError("native timing observer lacks seal notifications")
        self.lib.timing_abi_version.argtypes = []
        self.lib.timing_abi_version.restype = ctypes.c_int
        if self.lib.timing_abi_version() != 2:
            raise ValueError("unsupported native timing ABI")
        self.lib.timing_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
        ]
        self.lib.timing_create.restype = ctypes.c_void_p
        for name in ("timing_start", "timing_finish"):
            fn = getattr(self.lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
            fn.restype = ctypes.c_int
        self.lib.timing_destroy.argtypes = [ctypes.c_void_p]
        self.lib.timing_destroy.restype = None
        self.device = device
        audit = os.environ.get("VLLM_DEFERRAL_NATIVE_AUDIT_DIR")
        audit_path = b""
        if audit:
            directory = Path(audit)
            directory.mkdir(parents=True, exist_ok=True)
            audit_path = str(
                directory / f"native-{os.getpid()}-{device.index}.bin"
            ).encode()
        self.handle = self.lib.timing_create(path.encode(), device.index, audit_path)
        if not self.handle:
            raise RuntimeError("native execution observer initialization failed")

    def start(self, step):
        import torch

        if not self.handle or type(step) is not int or not 0 < step < 2**64:
            raise ValueError("invalid native timing step or closed observer")
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if not self.lib.timing_start(self.handle, step, stream):
            raise RuntimeError("native execution observation failed")

    def finish(self, step, event, enqueued_ns, stream=None):
        if (
            not self.handle
            or stream is None
            or not self.lib.timing_finish(self.handle, step, stream.cuda_stream)
        ):
            raise RuntimeError("native execution completion observation failed")

    def close(self):
        if self.handle:
            self.lib.timing_destroy(self.handle)
            self.handle = None


class UnavailableExecutionObserver:
    """Publish initialization failure without disabling primary completion timing."""

    def __init__(self, path):
        from vllm.v1.kv_offload.step_timing import RankFile

        self.rank = RankFile(path)

    def start(self, step):
        self.rank.write([step, 0, 0, 0, time.monotonic_ns(), 0, 0, 0, 0])

    def finish(self, step, event, enqueued_ns, stream=None):
        pass

    def close(self):
        self.rank.close()
