# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure native loads with actual block-major and tensor-major CPU mappings."""

import argparse
import json
import os
import statistics
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    assert not args.output.exists(), "Keep previous evidence"
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    assert not occupied, f"GPUs occupied: {occupied}"
    torch.set_num_threads(2)
    pages = [36864] * 78 + [8448] * 21
    rows, transferred = 1040, 912
    tensors = [
        CanonicalKVCacheTensor(
            torch.empty((rows, page), dtype=torch.int8, device="cuda:0"), page
        )
        for page in pages
    ]
    refs = [[CanonicalKVCacheRef(i, page) for i, page in enumerate(pages)]]
    records = []
    original = gpu_worker._coalesce_bounded_descriptors
    effective_count = transferred * len(pages)

    def capture(*values, **options):
        nonlocal effective_count
        effective_count = original(*values, **options)
        return effective_count

    gpu_worker._coalesce_bounded_descriptors = capture
    try:
        for layout in ("private", "private_tensor"):
            region = SharedOffloadRegion(
                engine_id=str(uuid.uuid4()),
                num_blocks=rows,
                rank=0,
                kv_bytes_per_block=round_up(
                    sum(pages), SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
                ),
                cpu_page_size=sum(pages),
                memory_layout=layout,
            )
            worker = CPUOffloadingWorker(
                CanonicalKVCaches(tensors, refs),
                blocks_per_chunk=1,
                num_cpu_blocks=rows,
                mmap_region=region,
                async_h2d_submission=True,
                async_h2d_chunk_bytes=8 * 1024**2,
            )
            job_id = 0
            try:
                for index, source in enumerate(worker._load_handler.src_tensors):
                    values = (torch.arange(rows) + index).remainder(127).to(torch.int8)
                    source.copy_(values[:, None])
                for gpu_order in ("ascending", "descending"):
                    ids = np.arange(transferred)
                    if gpu_order == "descending":
                        ids = ids[::-1].copy()
                    gpu_spec = GPULoadStoreSpec(
                        ids, group_sizes=(transferred,), block_indices=(0,)
                    )
                    cpu_spec = CPULoadStoreSpec(np.arange(transferred))
                    for mode in ("off", "sorted", "preserved"):
                        worker.configure_experiment(
                            dict(
                                copy_pipeline_depth=2,
                                load_descriptor_coalescing=mode != "off",
                                load_descriptor_packet_alignment=mode != "off",
                                descriptor_preserve_order=mode == "preserved",
                            )
                        )
                        for tensor in tensors:
                            tensor.tensor.fill_(-1)
                        torch.accelerator.synchronize()
                        samples = []
                        for repeat in range(args.repeats + 2):
                            job_id += 1
                            effective_count = transferred * len(pages)
                            start = time.perf_counter_ns()
                            assert worker.submit_load(job_id, cpu_spec, gpu_spec)
                            submitted = time.perf_counter_ns()
                            worker.wait({job_id})
                            finished = time.perf_counter_ns()
                            completed = worker.get_finished()
                            assert len(completed) == 1 and completed[0].job_id == job_id
                            assert not worker.get_finished()
                            assert completed[0].transfer_time is not None
                            if repeat >= 2:
                                samples.append(
                                    dict(
                                        cpu_submit_ms=(submitted - start) / 1e6,
                                        wall_ms=(finished - start) / 1e6,
                                        gpu_ms=completed[0].transfer_time * 1000,
                                    )
                                )
                        for tensor, source in zip(
                            tensors, worker._load_handler.src_tensors
                        ):
                            copied = tensor.tensor.cpu()
                            assert torch.equal(copied[ids], source[:transferred])
                            assert torch.all(copied[transferred:] == -1)
                        sizes = worker._load_handler._buffer_pool[-1][2].numpy()
                        chunks, used = 0, 0
                        for size in sizes[:effective_count]:
                            if used and used + size > 8 * 1024**2:
                                chunks += 1
                                used = 0
                            used += int(size)
                        chunks += bool(used)
                        row = dict(
                            layout=layout,
                            gpu_order=gpu_order,
                            mode=mode,
                            raw_descriptors=transferred * len(pages),
                            effective_descriptors=effective_count,
                            native_chunks=chunks,
                            bytes_validated=transferred * sum(pages),
                            samples=samples,
                            medians={
                                key: statistics.median(
                                    sample[key] for sample in samples
                                )
                                for key in samples[0]
                            },
                        )
                        records.append(row)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(
                            json.dumps(dict(complete=False, records=records), indent=2)
                            + "\n"
                        )
                        print(
                            json.dumps(
                                {k: v for k, v in row.items() if k != "samples"}
                            ),
                            flush=True,
                        )
            finally:
                worker.shutdown()
        result = dict(
            complete=True,
            records=records,
            owner_pid=os.getpid(),
            bytes_per_load=transferred * sum(pages),
            depth=2,
            chunk_bytes=8 * 1024**2,
            definition="GPU zero, private tensors and actual mmap layouts. "
            "Each case excludes two warmups, retires every native job exactly "
            "once and checks all copied bytes and untouched rows. CPU submission "
            "includes descriptor work; GPU time is the native transfer envelope. "
            "No model or concurrent compute is present. Descriptor group order "
            "changes do not identify a specific GPU hardware mechanism.",
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        gpu_worker._coalesce_bounded_descriptors = original


if __name__ == "__main__":
    main()
