# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure native H2D ordering using isolated buffers and synthetic compute."""

import argparse
import json
import time
from pathlib import Path

import torch

from vllm import _custom_ops as ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.accelerator.set_device_index(0)
    size = 64 * 1024**2
    source = torch.full((size,), 73, dtype=torch.uint8, pin_memory=True)
    destination = torch.zeros(size, dtype=torch.uint8, device="cuda")
    independent = torch.cuda.Stream()
    offsets = range(0, size, 1024**2)

    def descriptors(values):
        return torch.tensor(list(values), dtype=torch.int64, pin_memory=True)

    src = descriptors(source.data_ptr() + offset for offset in offsets)
    dst = descriptors(destination.data_ptr() + offset for offset in offsets)
    sizes = descriptors(1024**2 for _ in offsets)
    handle = ops.create_async_batch_memcpy_submitter(0, 8 * 1024**2)
    ops.configure_async_batch_memcpy_pipeline(handle, 2)
    results = []
    try:
        for job in range(7):
            torch.accelerator.synchronize()
            mode = "compute_stream" if job % 2 else "independent_idle_stream"
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            torch.cuda._sleep(150_000_000)
            end.record()
            start = time.monotonic_ns()
            stream = torch.cuda.current_stream() if job % 2 else independent
            with torch.cuda.stream(stream):
                ops.submit_async_batch_memcpy(handle, job, src, dst, sizes, True)
            submitted = time.monotonic_ns()
            ops.wait_async_batch_memcpy(handle, [job])
            ready = time.monotonic_ns()
            compute_done = end.query()
            ids, durations = ops.poll_async_batch_memcpy(handle)
            assert ids == [job]
            end.synchronize()
            assert torch.equal(source, destination.cpu())
            results.append(
                dict(
                    job=job,
                    mode=mode,
                    warmup=job == 0,
                    host_submit_us=(submitted - start) / 1e3,
                    submit_to_ready_ms=(ready - start) / 1e6,
                    synthetic_compute_ms=begin.elapsed_time(end),
                    compute_done_at_copy_completion=compute_done,
                    native_copy_ms=durations[0],
                    bytes_verified=size,
                )
            )
    finally:
        ops.destroy_async_batch_memcpy_submitter(handle)
        torch.accelerator.synchronize()
    measured = results[1:]
    assert all(
        r["compute_done_at_copy_completion"] == (r["mode"] == "compute_stream")
        for r in measured
    )
    args.output.write_text(
        json.dumps(
            dict(
                isolated_buffers=True,
                production_dependency_changed=False,
                chunk_bytes=8 * 1024**2,
                depth=2,
                gpu=torch.cuda.get_device_name(),
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(measured), flush=True)


if __name__ == "__main__":
    main()
