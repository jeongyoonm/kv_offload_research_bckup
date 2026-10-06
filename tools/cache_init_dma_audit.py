# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic connector: audit completed transfers before releasing blocks.

This synchronizes and copies cache bytes; do not use its runs for timing.
"""

import bisect
import hashlib
import json
import os
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)


def _stable_indices(logits, starts, ends, topk):
    """Diagnostic oracle: descending scores, ties resolved by token index."""
    positions = torch.arange(logits.shape[1], device=logits.device)
    valid = (positions >= starts.reshape(-1, 1)) & (positions < ends.reshape(-1, 1))
    scores = logits.masked_fill(~valid, float("-inf"))
    selected = scores.argsort(dim=1, descending=True, stable=True)[:, :topk]
    selected_valid = valid.gather(1, selected)
    relative = selected - starts.reshape(-1, 1)
    result = torch.where(selected_valid, relative, -1).to(torch.int32)
    if result.shape[1] < topk:
        result = torch.nn.functional.pad(result, (0, topk - result.shape[1]), value=-1)
    return result


def _install_stable_topk():
    from vllm import _custom_ops as ops

    def prefill(logits, starts, ends, indices, num_rows, stride0, stride1, topk):
        indices[:num_rows].copy_(_stable_indices(logits, starts, ends, topk))

    def cooperative(logits, lengths, indices, workspace, topk, max_seq_len):
        ends = lengths.reshape(-1)
        indices.copy_(_stable_indices(logits, torch.zeros_like(ends), ends, topk))

    ops.top_k_per_row_prefill = prefill
    torch.ops._C.cooperative_topk = cooperative
    torch.ops._C.persistent_topk = cooperative


class StableTopkOffloadingConnector(OffloadingConnector):
    """Deterministic selection control without transfer auditing or CPU copies."""

    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        _install_stable_topk()


class AuditedOffloadingConnector(OffloadingConnector):
    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        worker = self.connector_worker.worker
        self._stored_fingerprints = {}
        self._loaded_blocks = {}
        self._previous_store = {}
        self._audit_handler(worker._load_handler, "H2D")
        self._audit_handler(worker._store_handler, "D2H")
        output = Path(os.environ["CACHE_INIT_DMA_AUDIT_DIR"])
        rank = torch.accelerator.current_device_index()
        info = [
            {
                "tensor_index": index,
                "page_bytes": tensor.shape[1],
                "layers": [
                    name
                    for name, original in kv_caches.items()
                    if original.data_ptr() == tensor.data_ptr()
                ],
            }
            for index, tensor in enumerate(worker._load_handler.dst_tensors)
        ]
        (output / f"rank{rank}_tensors.json").write_text(
            json.dumps(info, indent=2) + "\n"
        )
        if os.environ.get("CACHE_INIT_MOE_REPLAY") == "1":
            import flashinfer

            original_moe = flashinfer.fused_moe.trtllm_fp4_block_scale_moe
            replayed = False

            def replay_moe(*args, **kwargs):
                nonlocal replayed
                if replayed or kwargs["hidden_states"].shape[0] != 4160:
                    return original_moe(*args, **kwargs)
                replayed = True
                inputs = {
                    key: kwargs[key].clone()
                    for key in (
                        "hidden_states",
                        "hidden_states_scale",
                        "routing_logits",
                    )
                }
                first = original_moe(*args, **kwargs)[0].clone()
                differences = []
                for _ in range(3):
                    other = original_moe(*args, **kwargs)[0]
                    difference = (first.float() - other.float()).abs()
                    differences.append(
                        {
                            "changed_elements": int((first != other).sum().item()),
                            "total_elements": first.numel(),
                            "max_abs": float(difference.max().item()),
                            "mean_abs": float(difference.mean().item()),
                        }
                    )
                unchanged = {
                    key: bool(
                        (tensor.view(torch.uint8) == kwargs[key].view(torch.uint8))
                        .all()
                        .item()
                    )
                    for key, tensor in inputs.items()
                }
                (output / f"rank{rank}_moe_replay.json").write_text(
                    json.dumps(
                        {"inputs_unchanged": unchanged, "differences": differences},
                        indent=2,
                    )
                    + "\n"
                )
                assert all(unchanged.values())
                return original_moe(*args, **kwargs)

            flashinfer.fused_moe.trtllm_fp4_block_scale_moe = replay_moe
        from vllm.v1.worker.kv_connector_model_runner_mixin import (
            KVConnectorModelRunnerMixin,
        )

        original = KVConnectorModelRunnerMixin._get_kv_connector_output

        @contextmanager
        def checked_forward(scheduler_output, **kwargs):
            with original(scheduler_output, **kwargs) as output:
                if scheduler_output.total_num_scheduled_tokens:
                    self._audit_before_forward()
                yield output

        KVConnectorModelRunnerMixin._get_kv_connector_output = staticmethod(
            checked_forward
        )
        from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

        original_pre_forward = ActiveKVConnector.pre_forward

        def checked_pre_forward(instance, scheduler_output):
            original_pre_forward(instance, scheduler_output)
            if scheduler_output.total_num_scheduled_tokens:
                self._audit_before_forward()

        ActiveKVConnector.pre_forward = checked_pre_forward
        if os.environ.get("CACHE_INIT_TOPK_REPLAY") == "1":
            from vllm import _custom_ops as ops

            original_topk = ops.top_k_per_row_prefill
            replayed_topk = False

            def replay_topk(logits, starts, ends, indices, *args):
                nonlocal replayed_topk
                result = original_topk(logits, starts, ends, indices, *args)
                if replayed_topk or logits.shape[0] != 4160:
                    return result
                replayed_topk = True
                first = indices.clone()
                first_logits = logits[-1].clone()
                comparisons = []
                for _ in range(3):
                    original_topk(logits, starts, ends, indices, *args)
                    comparisons.append(
                        {
                            "ordered_changed_elements": int(
                                (first != indices).sum().item()
                            ),
                            "membership_changed_elements": int(
                                (first.sort(dim=1).values != indices.sort(dim=1).values)
                                .sum()
                                .item()
                            ),
                            "last_row_membership_changed": int(
                                (first[-1].sort().values != indices[-1].sort().values)
                                .sum()
                                .item()
                            ),
                        }
                    )
                unchanged = bool((first_logits == logits[-1]).all().item())
                torch.save(
                    {
                        "logits": logits[-1].cpu(),
                        "start": int(starts[-1].item()),
                        "end": int(ends[-1].item()),
                        "indices": first[-1].cpu(),
                    },
                    output / f"rank{rank}_topk_row.pt",
                )
                (output / f"rank{rank}_topk_replay.json").write_text(
                    json.dumps(
                        {"logits_unchanged": unchanged, "comparisons": comparisons},
                        indent=2,
                    )
                    + "\n"
                )
                assert unchanged
                return result

            ops.top_k_per_row_prefill = replay_topk
        if os.environ.get("CACHE_INIT_STABLE_TOPK") == "1":
            _install_stable_topk()

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

        def transfer_async(job_id, src_spec, dst_spec):
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
            result = submit(job_id, src_spec, dst_spec)
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
