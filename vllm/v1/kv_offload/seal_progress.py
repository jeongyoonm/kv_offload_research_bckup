# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe physical predecessor completion without retiring its result."""

import time

from vllm.v1.kv_offload.execution_timing import valid_snapshot


def compute_progress(core, queued):
    """Return whether compute exists and whether every queued compute ended."""
    productive = [
        (future, output)
        for future, output, _ in queued
        if getattr(output, "total_num_scheduled_tokens", 0) > 0
    ]
    timing = getattr(core, "deferral_timing", None)
    for future, output in productive:
        if future.poll():
            continue
        step = getattr(output, "readiness_timeline_step", None)
        records = timing.read_execution(step) if timing and step else []
        now = time.monotonic_ns()
        world = core.vllm_config.parallel_config.world_size
        if len(records) != world or not valid_snapshot(
            records, step, now, completed=True
        ):
            return bool(productive), False
    return bool(productive), True
