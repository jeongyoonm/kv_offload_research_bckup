# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment-only destination dependencies for direct sharded CPU loads."""

import torch

from vllm.platforms import current_platform
from vllm.v1.kv_offload import queue_stages


class BlockReadiness:
    """Retain the last compute/zero/copy fence for each physical GPU block."""

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.events = {}
        self.requests = {}
        self.stream = current_platform.Stream()
        self.initial = torch.Event()
        self.initial.record()

    def begin(self, output):
        for req_id in output.finished_req_ids | (output.preempted_req_ids or set()):
            self.requests.pop(req_id, None)
        for request in output.scheduled_new_reqs:
            self.requests[request.req_id] = tuple(list(g) for g in request.block_ids)
        cached = output.scheduled_cached_reqs
        for req_id, groups in zip(cached.req_ids, cached.new_block_ids):
            if groups is None:
                continue
            if req_id in cached.resumed_req_ids:
                self.requests[req_id] = tuple(list(g) for g in groups)
            else:
                existing = self.requests[req_id]
                assert len(existing) == len(groups)
                for old, new in zip(existing, groups):
                    old.extend(new)

    def record(self, block_ids, kind):
        blocks = set(map(int, block_ids))
        if not blocks:
            return
        if min(blocks) < 0 or max(blocks) >= self.num_blocks:
            raise RuntimeError("invalid physical block readiness identity")
        event = torch.Event()
        event.record()
        for block in blocks:
            self.events[block] = event
        if queue_stages.active:
            queue_stages.emit("block_readiness_record", kind=kind, blocks=len(blocks))

    def record_compute(self, output):
        blocks = (
            block
            for req_id in output.num_scheduled_tokens
            for group in self.requests[req_id]
            for block in group
        )
        self.record(blocks, "compute")

    def dependencies(self, block_ids):
        blocks = set(map(int, block_ids))
        if blocks and (min(blocks) < 0 or max(blocks) >= self.num_blocks):
            raise RuntimeError("invalid H2D destination block")
        events = {self.events.get(block, self.initial) for block in blocks}
        for event in events:
            self.stream.wait_event(event)
        if queue_stages.active:
            queue_stages.emit(
                "load_block_dependencies", blocks=len(blocks), events=len(events)
            )
        return self.stream
