# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic connector: audit completed transfers before releasing blocks.

This synchronizes and copies cache bytes; do not use its runs for timing.
"""

import bisect
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)


class AuditedOffloadingConnector(OffloadingConnector):
    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        worker = self.connector_worker.worker
        self._audit_handler(worker._load_handler, "H2D")
        self._audit_handler(worker._store_handler, "D2H")

    def _audit_handler(self, handler, direction):
        original = handler.get_finished
        gpu_views = handler.dst_tensors if direction == "H2D" else handler.src_tensors
        cpu_views = handler.src_tensors if direction == "H2D" else handler.dst_tensors
        ordered = sorted((tensor.data_ptr(), i) for i, tensor in enumerate(gpu_views))
        bases = [base for base, _ in ordered]
        output = Path(os.environ["CACHE_INIT_DMA_AUDIT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        rank = torch.accelerator.current_device_index()

        def get_finished():
            pending = {
                transfer.job_id: transfer
                for transfer in list(handler._transfers)
                + list(handler._async_transfers)
            }
            completed = original()
            for result in completed:
                transfer = pending[result.job_id]
                grouped = defaultdict(list)
                described_bytes = 0
                for source, target, size in zip(
                    transfer.batch_src.numpy(),
                    transfer.batch_dst.numpy(),
                    transfer.batch_sizes.numpy(),
                ):
                    if described_bytes == transfer.num_bytes:
                        break
                    source, target, size = int(source), int(target), int(size)
                    gpu_ptr, cpu_ptr = (
                        (target, source) if direction == "H2D" else (source, target)
                    )
                    _, index = ordered[bisect.bisect_right(bases, gpu_ptr) - 1]
                    gpu, cpu = gpu_views[index], cpu_views[index]
                    assert size == gpu.shape[1] == cpu.shape[1]
                    gpu_block, gpu_offset = divmod(
                        gpu_ptr - gpu.data_ptr(), gpu.stride(0)
                    )
                    cpu_block, cpu_offset = divmod(
                        cpu_ptr - cpu.data_ptr(), cpu.stride(0)
                    )
                    assert gpu_offset == cpu_offset == 0
                    grouped[index].append((cpu_block, gpu_block))
                    described_bytes += size
                assert described_bytes == transfer.num_bytes
                mismatched_bytes = 0
                for index, blocks in grouped.items():
                    cpu_ids, gpu_ids = zip(*blocks)
                    expected = cpu_views[index].index_select(
                        0, torch.tensor(cpu_ids, dtype=torch.long)
                    )
                    actual = (
                        gpu_views[index]
                        .index_select(
                            0,
                            torch.tensor(
                                gpu_ids,
                                dtype=torch.long,
                                device=gpu_views[index].device,
                            ),
                        )
                        .cpu()
                    )
                    mismatched_bytes += int((expected != actual).sum().item())
                with (output / f"rank{rank}.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "direction": direction,
                                "job_id": result.job_id,
                                "verified_bytes": described_bytes,
                                "mismatched_bytes": mismatched_bytes,
                            }
                        )
                        + "\n"
                    )
                assert mismatched_bytes == 0, (rank, direction, result.job_id)
            return completed

        handler.get_finished = get_finished
