# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse experiment records, exported only across drained configurations."""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

CAPACITY = 65536
active = False
generation = 0
rank = -1
device_index: int | None = None
records: list[tuple[str, int, int, dict[str, Any]]] = []
overflow = False
period_begin_ns = time.monotonic_ns()
lock = threading.Lock()


def emit(event, **fields):
    global overflow
    if not active:
        return
    # Format common fields only after drain, not on worker/engine hot paths.
    record = (event, time.monotonic_ns(), generation, fields)
    with lock:
        if len(records) == CAPACITY:
            overflow = True
        else:
            records.append(record)


def configure(
    enabled, next_generation, worker_rank=-1, counters=None, worker_device=None
):
    global active, generation, rank, records, overflow, period_begin_ns, device_index
    directory = Path(os.environ["VLLM_DEFERRAL_TIMING_DIR"]) / "queue_stages"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{os.getpid()}-{generation}-{time.monotonic_ns()}.json"
    with lock:
        process_id = os.getpid()
        exported = [
            dict(
                event=event,
                time_ns=timestamp,
                generation=event_generation,
                rank=rank,
                pid=process_id,
                **fields,
            )
            for event, timestamp, event_generation, fields in records
        ]
        with path.open("x") as stream:
            json.dump(
                dict(
                    generation=generation,
                    rank=rank,
                    overflow=overflow,
                    events=exported,
                    device_index=device_index,
                    counters=counters,
                    begin_ns=period_begin_ns,
                    end_ns=time.monotonic_ns(),
                ),
                stream,
            )
        if overflow:
            raise RuntimeError(f"queue stage overflow: {path}")
        records = []
        period_begin_ns = time.monotonic_ns()
        device_index = worker_device
        generation, rank, active = next_generation, worker_rank, enabled
    return dict(rank=worker_rank, generation=generation, export=str(path))
