# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, buffered host timeline for KV readiness diagnostics."""

import functools
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

from vllm.v1.kv_offload import timing_diagnostics

DIRECTORY = os.environ.get("VLLM_KV_READINESS_TIMELINE_DIR")
GPU_TRACE = os.environ.get("VLLM_KV_READINESS_GPU_TRACE") == "1"
LOAD_AUDIT = os.environ.get("VLLM_DEFERRAL_LOAD_AUDIT") == "1"
_events: list[dict] = []
_lock = threading.Lock()
_started = False
P = ParamSpec("P")
R = TypeVar("R")


def trace_execution(fn: Callable[P, R]) -> Callable[P, R]:
    """Correlate forward CUDA activity with a scheduler step when tracing."""

    @functools.wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        output = args[1] if len(args) > 1 else kwargs.get("scheduler_output")
        step = getattr(output, "readiness_timeline_step", None)
        if enabled() and step is not None:
            from vllm.distributed import get_tensor_model_parallel_rank

            emit(
                "worker_entry",
                step=step,
                opportunity_id=getattr(output, "readiness_opportunity_id", None),
                rank=get_tensor_model_parallel_rank(),
            )
        if not GPU_TRACE or step is None:
            return fn(*args, **kwargs)
        import torch

        cast(Any, args[0])._readiness_timeline_step = step
        torch.cuda.nvtx.range_push(f"kvexecute|{step}")
        try:
            return fn(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    return wrapped


def trace_sampling(fn: Callable[P, R]) -> Callable[P, R]:
    """Include sampling kernels in the step that produced their hidden states."""

    @functools.wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        step = getattr(args[0], "_readiness_timeline_step", None)
        if not GPU_TRACE or step is None:
            return fn(*args, **kwargs)
        import torch

        torch.cuda.nvtx.range_push(f"kvexecute|{step}")
        try:
            return fn(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    return wrapped


def enabled() -> bool:
    return DIRECTORY is not None or timing_diagnostics.enabled()


def emit(event: str, **fields) -> None:
    """Buffer an observation; completion times describe polling, not CUDA events."""
    global _started
    if timing_diagnostics.enabled():
        timing_diagnostics.emit(event, **fields)
    if DIRECTORY is None:
        return
    record = {
        "event": event,
        "time_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "tid": threading.get_native_id(),
        **fields,
    }
    if GPU_TRACE:
        import torch

        torch.cuda.nvtx.mark(f"kvclock|{record['time_ns']}")
        record["nvtx_after_ns"] = time.monotonic_ns()
    with _lock:
        _events.append(record)
        if not _started:
            _started = True
            threading.Thread(target=_flush_loop, daemon=True).start()


def _flush_loop() -> None:
    global _events
    assert DIRECTORY is not None
    directory = Path(DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    last = 0
    while True:
        time.sleep(0.5)
        trigger = directory / "flush.request"
        try:
            stamp = trigger.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        if stamp == last:
            continue
        with _lock:
            batch, _events = _events, []
        output = directory / f"{os.getpid()}.{stamp}.jsonl"
        with output.open("w") as stream:
            for record in batch:
                stream.write(json.dumps(record) + "\n")
        (directory / f"{os.getpid()}.flushed").write_text(str(stamp))
        last = stamp
