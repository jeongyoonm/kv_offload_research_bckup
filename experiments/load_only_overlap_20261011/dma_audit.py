# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer-byte correctness diagnostics; never enable in timing cells."""

import bisect
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


class AuditedOffloadingConnector(OffloadingConnector):
    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        worker = self.connector_worker.worker
        self._stored_fingerprints = {}
        self._loaded_blocks = {}
        self._previous_store = {}
        self._audit_handler(worker._load_handler, "H2D")
        self._audit_handler(worker._store_handler, "D2H")
        original = ActiveKVConnector.pre_forward

        def checked(instance, scheduler_output):
            original(instance, scheduler_output)
            if scheduler_output.total_num_scheduled_tokens:
                self._audit_before_forward()

        ActiveKVConnector.pre_forward = checked

    def _audit_handler(self, handler, direction):
        original = handler.get_finished
        gpu_views = handler.dst_tensors if direction == "H2D" else handler.src_tensors
        cpu_views = handler.src_tensors if direction == "H2D" else handler.dst_tensors
        ordered = sorted((tensor.data_ptr(), i) for i, tensor in enumerate(gpu_views))
        bases = [base for base, _ in ordered]
        output = Path(os.environ["CACHE_INIT_DMA_AUDIT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        rank = torch.accelerator.current_device_index()
        submit = handler.transfer_async

        def transfer_async(job_id, src_spec, dst_spec, **kwargs):
            assert handler.src_blocks_per_chunk == handler.dst_blocks_per_chunk == 1
            expected = []
            offset = 0
            gpu_spec = dst_spec if direction == "H2D" else src_spec
            for size, refs in zip(gpu_spec.group_sizes, handler.layer_refs_per_group):
                for ref in refs:
                    for source, destination in zip(
                        src_spec.block_ids[offset : offset + size],
                        dst_spec.block_ids[offset : offset + size],
                    ):
                        src = handler.src_tensors[ref.tensor_idx]
                        dst = handler.dst_tensors[ref.tensor_idx]
                        expected.append(
                            (
                                src.data_ptr() + int(source) * src.stride(0),
                                dst.data_ptr() + int(destination) * dst.stride(0),
                                ref.page_size_bytes,
                            )
                        )
                offset += size
            result = submit(job_id, src_spec, dst_spec, **kwargs)
            transfer = (handler._async_transfers or handler._transfers)[-1]
            actual = list(
                zip(
                    transfer.batch_src.numpy()[: len(expected)].tolist(),
                    transfer.batch_dst.numpy()[: len(expected)].tolist(),
                    transfer.batch_sizes.numpy()[: len(expected)].tolist(),
                )
            )
            assert actual == expected, (rank, direction, job_id, "mapping")
            return result

        handler.transfer_async = transfer_async

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
                changed_cpu_blocks = 0
                fingerprints = []
                cold_differences = []
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
                    if direction == "D2H":
                        previous = self._previous_store.get(index)
                        if previous is not None and previous.shape == expected.shape:
                            changed = expected != previous
                            cold_differences.append(
                                {
                                    "tensor_index": index,
                                    "changed_bytes": int(changed.sum().item()),
                                    "total_bytes": expected.numel(),
                                    "first_block_changed_bytes": int(
                                        changed[0].sum().item()
                                    ),
                                }
                            )
                        self._previous_store[index] = expected
                    for row, cpu_block in zip(expected.numpy(), cpu_ids):
                        digest = hashlib.sha256(row.tobytes()).hexdigest()
                        key = (index, cpu_block)
                        if direction == "D2H":
                            self._stored_fingerprints[key] = digest
                        else:
                            assert key in self._stored_fingerprints, key
                            changed_cpu_blocks += int(
                                self._stored_fingerprints[key] != digest
                            )
                        fingerprints.append([index, cpu_block, digest])
                    if direction == "H2D":
                        for cpu_block, gpu_block in blocks:
                            self._loaded_blocks[index, gpu_block] = cpu_block
                with (output / f"rank{rank}.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "direction": direction,
                                "job_id": result.job_id,
                                "verified_bytes": described_bytes,
                                "mismatched_bytes": mismatched_bytes,
                                "changed_cpu_blocks": changed_cpu_blocks,
                                "fingerprints": fingerprints,
                                "previous_store_differences": cold_differences,
                            }
                        )
                        + "\n"
                    )
                assert mismatched_bytes == 0, (rank, direction, result.job_id)
                assert changed_cpu_blocks == 0, (rank, direction, result.job_id)
            return completed

        handler.get_finished = get_finished

    def _audit_before_forward(self):
        # Check completed loads after input preparation, immediately before forward.
        if self._loaded_blocks:
            worker = self.connector_worker.worker
            grouped = defaultdict(list)
            for (index, gpu_block), cpu_block in self._loaded_blocks.items():
                grouped[index].append((cpu_block, gpu_block))
            mismatched_bytes = 0
            verified_bytes = 0
            for index, blocks in grouped.items():
                cpu_ids, gpu_ids = zip(*blocks)
                cpu = worker._load_handler.src_tensors[index]
                gpu = worker._load_handler.dst_tensors[index]
                expected = cpu.index_select(0, torch.tensor(cpu_ids, dtype=torch.long))
                actual = gpu.index_select(
                    0, torch.tensor(gpu_ids, dtype=torch.long, device=gpu.device)
                ).cpu()
                verified_bytes += expected.numel()
                mismatched_bytes += int((actual != expected).sum().item())
            rank = torch.accelerator.current_device_index()
            output = Path(os.environ["CACHE_INIT_DMA_AUDIT_DIR"])
            with (output / f"rank{rank}.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "direction": "before_forward",
                            "verified_bytes": verified_bytes,
                            "mismatched_bytes": mismatched_bytes,
                        }
                    )
                    + "\n"
                )
            assert mismatched_bytes == 0, (rank, "before_forward")
            self._loaded_blocks.clear()
