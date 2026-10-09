# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate same-context H2D/D2H CPU submission interference with exact bytes."""

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import nvtx
import torch

from vllm import _custom_ops as ops
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
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    assert not ({int(pid) for pid in occupied.splitlines()} - {os.getpid()}), occupied
    assert not args.output.exists(), "Use a fresh output path"
    torch.set_num_threads(4)
    torch.accelerator.set_device_index(args.device)
    pages = [36864] * 78 + [8448] * 21
    store_rows, load_rows = 48, 912
    rows = store_rows + load_rows
    tensors = [
        torch.arange(rows, dtype=torch.int16, device=f"cuda:{args.device}")
        .remainder(127)
        .to(torch.int8)[:, None]
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
        async_h2d_submission=True,
        async_h2d_chunk_bytes=8 * 1024 * 1024,
    )
    store, load = worker._store_handler, worker._load_handler
    cpu = store.dst_tensors
    for host in cpu:
        host[store_rows:].fill_(31)
    torch.accelerator.synchronize()
    load_ids = list(range(store_rows, rows))
    job_id = 0
    records = []
    try:
        for layout in ("contiguous", "reverse_source"):
            src_ids = list(range(store_rows))
            if layout == "reverse_source":
                src_ids.reverse()
            for coalesced in (False, True):
                worker.configure_experiment(
                    dict(
                        copy_pipeline_depth=2,
                        store_descriptor_coalescing=coalesced,
                    )
                )
                for background_jobs in (0, 2):
                    samples = []
                    for repeat in range(args.repeats + 2):
                        torch.accelerator.synchronize()
                        jobs = []
                        with nvtx.annotate(
                            f"h2d_queue/layout={layout}/coalesced={coalesced}/"
                            f"background={background_jobs}/repeat={repeat}"
                        ):
                            for _ in range(background_jobs):
                                job_id += 1
                                jobs.append(job_id)
                                assert worker.submit_load(
                                    job_id,
                                    CPULoadStoreSpec(load_ids),
                                    GPULoadStoreSpec(load_ids, (load_rows,), (0,)),
                                )
                        operator = {}

                        def copy(
                            src,
                            dst,
                            sizes,
                            is_src_access_order_any=False,
                            *,
                            operator=operator,
                        ):
                            begin = time.perf_counter_ns()
                            with nvtx.annotate("d2h_operator"):
                                ops.swap_blocks_batch(
                                    src,
                                    dst,
                                    sizes,
                                    is_src_access_order_any=is_src_access_order_any,
                                )
                            operator.update(
                                cpu_operator_ms=(time.perf_counter_ns() - begin) / 1e6,
                                descriptors=len(src),
                            )

                        store._swap_blocks_batch = copy
                        job_id += 1
                        begin = time.perf_counter_ns()
                        with nvtx.annotate(
                            f"d2h_submit/layout={layout}/coalesced={coalesced}/"
                            f"background={background_jobs}/repeat={repeat}"
                        ):
                            assert worker.submit_store(
                                job_id,
                                GPULoadStoreSpec(src_ids, (store_rows,), (0,)),
                                CPULoadStoreSpec(list(range(store_rows))),
                            )
                        cpu_submit_ms = (time.perf_counter_ns() - begin) / 1e6
                        completed_at_return = store._transfers[-1].end_event.query()
                        store.wait({job_id})
                        done = store.get_finished()
                        assert len(done) == 1 and done[0].job_id == job_id
                        load.wait(set(jobs))
                        loaded = load.get_finished()
                        assert [result.job_id for result in loaded] == jobs
                        if repeat in (0, args.repeats + 1):
                            with nvtx.annotate("byte_validation"):
                                expected = np.asarray(src_ids, dtype=np.int8)[:, None]
                                assert all(
                                    np.all(host[:store_rows].numpy() == expected)
                                    for host in cpu
                                )
                                if background_jobs:
                                    assert all(
                                        np.all(tensor[store_rows:].cpu().numpy() == 31)
                                        for tensor in tensors
                                    )
                        if repeat >= 2:
                            samples.append(
                                dict(
                                    cpu_submit_ms=cpu_submit_ms,
                                    gpu_transfer_ms=done[0].transfer_time * 1000,
                                    completed_at_return=completed_at_return,
                                    background_load_ms=[
                                        r.transfer_time * 1000 for r in loaded
                                    ],
                                    **operator,
                                )
                            )
                    record = dict(
                        layout=layout,
                        coalesced=coalesced,
                        background_jobs=background_jobs,
                        byte_validation=True,
                        samples=samples,
                        cpu_operator_median_ms=statistics.median(
                            r["cpu_operator_ms"] for r in samples
                        ),
                        cpu_submit_median_ms=statistics.median(
                            r["cpu_submit_ms"] for r in samples
                        ),
                        gpu_transfer_median_ms=statistics.median(
                            r["gpu_transfer_ms"] for r in samples
                        ),
                    )
                    records.append(record)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(
                            dict(
                                complete=False,
                                store_bytes=sum(pages) * store_rows,
                                load_bytes=sum(pages) * load_rows,
                                chunk_bytes=8 * 1024 * 1024,
                                depth=2,
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
                        json.dumps({k: v for k, v in record.items() if k != "samples"}),
                        flush=True,
                    )
        result = json.loads(args.output.read_text())
        result.update(
            complete=True,
            scope="Isolated private tensors, same GPU CUDA context, exact-byte "
            "validation. H2D destinations and CPU sources are disjoint from D2H "
            "sources and CPU destinations. Pending background loads do not prove "
            "CPU API overlap: use NVTX and Nsight API records to verify overlap. "
            "This probe has no model or full TP8 serving workload.",
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        worker.shutdown()


if __name__ == "__main__":
    main()
