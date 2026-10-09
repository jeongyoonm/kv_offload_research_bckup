# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure CPU submission and DMA separately for representative GLM D2H jobs."""

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    other = {int(pid) for pid in occupied.splitlines()} - {os.getpid()}
    assert not other, f"Other GPU workloads must be stopped: {other}"
    assert not args.output.exists(), "Use a fresh output path"
    torch.set_num_threads(4)
    torch.accelerator.set_device_index(args.device)
    pages = [36864] * 78 + [8448] * 21
    rows = 48
    tensors = [
        torch.arange(rows, dtype=torch.int8, device=f"cuda:{args.device}")[:, None]
        .expand(-1, page)
        .contiguous()
        for page in pages
    ]
    worker = CPUOffloadingWorker(
        kv_caches=CanonicalKVCaches(
            tensors=[
                CanonicalKVCacheTensor(tensor=t, page_size_bytes=p)
                for t, p in zip(tensors, pages)
            ],
            group_data_refs=[
                [
                    CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=p)
                    for i, p in enumerate(pages)
                ]
            ],
        ),
        blocks_per_chunk=1,
        num_cpu_blocks=rows,
    )
    handler = worker._store_handler
    cpu = handler.dst_tensors
    producer = current_platform.current_stream()
    records = []
    job_id = 0
    try:
        for layout in ("contiguous", "reverse_source"):
            src_ids = list(range(rows))
            if layout == "reverse_source":
                src_ids.reverse()
            modes = ["batch_stream", "coalesced_stream", "prefetch_if_ready"]
            if layout == "contiguous":
                modes.append("whole_layer_copies")
            for mode in modes:
                worker.configure_experiment(
                    {"store_descriptor_coalescing": mode == "coalesced_stream"}
                )
                for sleep_cycles in (0, 25_000_000, 100_000_000):
                    samples = []
                    for repeat in range(args.repeats + 2):
                        torch.accelerator.synchronize()
                        torch.cuda._sleep(sleep_cycles)
                        operator = {}

                        def copy(
                            src,
                            dst,
                            sizes,
                            is_src_access_order_any=False,
                            *,
                            mode=mode,
                            operator=operator,
                        ):
                            ready = producer.query()
                            access_any = mode == "prefetch_if_ready" and ready
                            begin = time.perf_counter_ns()
                            if mode == "whole_layer_copies":
                                for host, gpu in zip(cpu, tensors):
                                    host.copy_(gpu, non_blocking=True)
                            else:
                                ops.swap_blocks_batch(
                                    src,
                                    dst,
                                    sizes,
                                    is_src_access_order_any=access_any,
                                )
                            operator.update(
                                cpu_operator_ms=(time.perf_counter_ns() - begin) / 1e6,
                                descriptors=len(src),
                                producer_ready=ready,
                                source_access_any=access_any,
                            )

                        handler._swap_blocks_batch = copy
                        job_id += 1
                        begin = time.perf_counter_ns()
                        assert worker.submit_store(
                            job_id,
                            GPULoadStoreSpec(src_ids, (rows,), (0,)),
                            CPULoadStoreSpec(list(range(rows))),
                        )
                        cpu_submit_ms = (time.perf_counter_ns() - begin) / 1e6
                        transfer = handler._transfers[-1]
                        completed_at_return = transfer.end_event.query()
                        worker.wait({job_id})
                        done = worker.get_finished()
                        assert len(done) == 1 and done[0].job_id == job_id
                        assert done[0].transfer_time is not None
                        if repeat in (0, args.repeats + 1):
                            expected = np.asarray(src_ids, dtype=np.int8)[:, None]
                            assert all(np.all(t.numpy() == expected) for t in cpu)
                        if repeat >= 2:
                            samples.append(
                                dict(
                                    cpu_submit_ms=cpu_submit_ms,
                                    gpu_transfer_ms=done[0].transfer_time * 1000,
                                    completed_at_return=completed_at_return,
                                    **operator,
                                )
                            )
                    records.append(
                        dict(
                            layout=layout,
                            mode=mode,
                            sleep_cycles=sleep_cycles,
                            byte_validation=True,
                            samples=samples,
                            cpu_submit_median_ms=statistics.median(
                                r["cpu_submit_ms"] for r in samples
                            ),
                            cpu_operator_median_ms=statistics.median(
                                r["cpu_operator_ms"] for r in samples
                            ),
                            gpu_transfer_median_ms=statistics.median(
                                r["gpu_transfer_ms"] for r in samples
                            ),
                            complete_at_return_ratio=sum(
                                r["completed_at_return"] for r in samples
                            )
                            / len(samples),
                        )
                    )
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(
                            dict(
                                complete=False,
                                bytes_per_job=sum(pages) * rows,
                                raw_descriptors=len(pages) * rows,
                                descriptor_cap=os.environ.get(
                                    "VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS", "default"
                                ),
                                records=records,
                            ),
                            indent=2,
                        )
                        + "\n"
                    )
                    print(
                        json.dumps(
                            {k: v for k, v in records[-1].items() if k != "samples"}
                        ),
                        flush=True,
                    )
        result = json.loads(args.output.read_text())
        result["complete"] = True
        result["scope"] = (
            "Isolated private GPU/CPU tensors matching a 146.5 MB GLM store. "
            "ANY is used only when this isolated source producer is complete; "
            "it is not enabled for live model stores. Source-ready delays, "
            "contiguous and reverse layouts distinguish API submission and DMA."
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        worker.shutdown()


if __name__ == "__main__":
    main()
