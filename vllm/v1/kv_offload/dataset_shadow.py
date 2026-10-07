# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe initial execution snapshots without opening a seal transaction."""

import os
import queue
import threading
import time

from vllm.v1.kv_offload.execution_timing import valid_snapshot


def collection_compute_held(request):
    """Keep diagnostic transfer requests out of authoritative compute membership."""
    if request.num_computed_tokens <= 0:
        return False
    prefix = os.environ.get("VLLM_DATASET_BACKGROUND_PREFIX")
    if prefix and prefix in request.request_id:
        return True
    gate = os.environ.get("VLLM_DATASET_TARGET_GATE")
    return bool(
        gate and "coverage-main-" in request.request_id and os.path.exists(gate)
    )


class DatasetShadow:
    def __init__(self, timing):
        self.timing = timing
        self.events: queue.SimpleQueue[dict] = queue.SimpleQueue()
        self.seen = set()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def sample(self):
        for step, data in list(self.timing.steps.items()):
            if step in self.seen or not data.get("prefill_only"):
                continue
            ranks = self.timing.read_execution(step)
            now = time.monotonic_ns()
            if len(ranks) != len(self.timing.execution_ranks):
                continue
            if not valid_snapshot(ranks, step, now):
                continue
            if all(r[8] for r in ranks):
                self.seen.add(step)
                continue
            self.events.put(
                dict(
                    event="dataset_forecast",
                    time_ns=now,
                    predecessor_step=step,
                    gpu_start_upper_ns=max(r[3] for r in ranks),
                    execution_snapshot=ranks,
                )
            )
            self.seen.add(step)
        self.seen.intersection_update(list(self.timing.steps))

    def _run(self):
        while not self.stop.wait(0.001):
            try:
                self.sample()
            except (OSError, ValueError, RuntimeError) as error:
                self.events.put(
                    dict(
                        event="dataset_observer_error",
                        time_ns=time.monotonic_ns(),
                        error=repr(error),
                    )
                )

    def drain(self):
        result = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except queue.Empty:
                return result

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("dataset observer did not stop")
