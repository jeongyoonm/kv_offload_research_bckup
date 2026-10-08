# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrate native async H2D chunks on concurrent ranks without model compute."""

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import subprocess
import time
import traceback
import uuid
from pathlib import Path

PAGE_SIZES = [36864] * 78 + [8448] * 21
ROW_BYTES = sum(PAGE_SIZES)
PROFILES = {"4k": 64, "16k": 243, "60k": 912}


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def chunks(sizes, budget, limit):
    result = []
    first = 0
    while first < len(sizes):
        end, total = first, 0
        while end < min(len(sizes), first + limit):
            value = int(sizes[end])
            if end > first and value > budget - total:
                break
            total += value
            end += 1
        result.append([end - first, total])
        first = end
    return result


def worker(rank, args, plan, engine, barrier, output):
    region = None
    handle = ring = None
    try:
        from tools.benchmark_cpu_cache_init import _bind, _thp_policy

        thp = _thp_policy("disabled")
        binding = _bind(rank, args.ranks, "local")
        # Match the serving allocation's disjoint 16-core + SMT CPU sets.
        local = rank % 4
        base = 0 if rank < 4 else 64
        affinity = set(range(base + local * 16, base + (local + 1) * 16))
        affinity |= {cpu + 128 for cpu in affinity.copy()}
        affinity &= os.sched_getaffinity(0)
        os.sched_setaffinity(0, affinity)
        binding.update(cpus=sorted(affinity), thp_disable=thp)
        import numpy as np
        import torch

        from vllm import _custom_ops as ops
        from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region
        from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.accelerator.set_device_index(rank)
        region = SharedOffloadRegion(
            engine_id=engine,
            num_blocks=2048,
            rank=rank,
            kv_bytes_per_block=(args.ranks * ROW_BYTES + 4095) // 4096 * 4096,
            cpu_page_size=ROW_BYTES,
            barrier=lambda: barrier.wait(300),
            population_mode="contiguous",
            population_workers=args.ranks,
            num_workers=args.ranks,
            process_rank=rank,
            memory_layout="private",
        )
        host = region._base.numpy().view(np.uint8).reshape(2048, ROW_BYTES)
        block_tags = (np.arange(2048, dtype=np.uint16) + rank * 19) % 256
        offset = 0
        for index, page in enumerate(PAGE_SIZES):
            pattern = (np.arange(page, dtype=np.uint16) + index * 7) % 256
            np.add(
                block_tags[:, None],
                pattern[None, :],
                out=host[:, offset : offset + page],
                casting="unsafe",
            )
            offset += page
        pin_mmap_region(region)
        assert region.is_pinned
        gpu = torch.empty(1024 * ROW_BYTES, dtype=torch.uint8, device="cuda")
        rng = np.random.default_rng(8100 + rank)
        source_ids = rng.permutation(2048)
        destination_ids = rng.permutation(1024)
        descriptors = {}
        host_flat = host.reshape(-1)
        for name, blocks in PROFILES.items():
            sources, destinations, sizes = [], [], []
            source_samples, destination_samples = [], []
            source_offset = destination_offset = 0
            for page in PAGE_SIZES:
                source = source_ids[:blocks] * ROW_BYTES + source_offset
                destination = destination_ids[:blocks] * page + destination_offset
                sources.append(source + region._base.data_ptr())
                destinations.append(destination + gpu.data_ptr())
                sizes.append(np.full(blocks, page, dtype=np.int64))
                points = np.r_[np.arange(32), np.arange(page - 32, page)]
                picks = [0, blocks // 2, blocks - 1]
                source_samples.append((source[picks, None] + points).reshape(-1))
                destination_samples.append(
                    (destination[picks, None] + points).reshape(-1)
                )
                source_offset += page
                destination_offset += 1024 * page

            def pinned(parts):
                return torch.from_numpy(np.concatenate(parts)).pin_memory()

            descriptors[name] = dict(
                src=pinned(sources),
                dst=pinned(destinations),
                sizes=pinned(sizes),
                expected=host_flat[np.concatenate(source_samples)].copy(),
                samples=torch.from_numpy(np.concatenate(destination_samples)).to(
                    "cuda"
                ),
            )
        torch.accelerator.synchronize()
        output.put(
            dict(
                event="ready",
                rank=rank,
                pid=os.getpid(),
                binding=binding,
                init=region.init_stats,
                registered_bytes=region.registration_size_bytes,
                gpu_bytes=gpu.numel(),
                gpu_name=torch.cuda.get_device_name(),
                capability=torch.cuda.get_device_capability(),
                source_numa_maps=[
                    line
                    for line in Path("/proc/self/numa_maps").read_text().splitlines()
                    if engine in line
                ],
            )
        )
        ring_path = str(args.output / f"completion-rank-{rank}")
        session = 8109
        ring = ops.create_async_batch_memcpy_completion_ring(ring_path, session, 4096)
        job = 0
        for cell in plan:
            desc = descriptors[cell["profile"]]
            chunk = cell["chunk_mib"] * 1024**2
            handle = ops.create_async_batch_memcpy_submitter(
                rank, chunk, 4096, ring_path, session
            )
            ops.configure_async_batch_memcpy_pipeline(handle, cell["depth"])
            if cell["repeat"] == 0:
                for _ in range(2):
                    job += 1
                    ops.submit_async_batch_memcpy(
                        handle, job, desc["src"], desc["dst"], desc["sizes"], True
                    )
                ops.wait_async_batch_memcpy(handle, [job])
                assert ops.poll_async_batch_memcpy(handle)[0] == [job - 1, job]
                ops.poll_async_batch_memcpy_completion_ring(ring)
            barrier.wait(300)
            begin = time.monotonic_ns()
            cpu_begin = time.process_time_ns()
            ids = []
            for _ in range(args.jobs):
                job += 1
                ids.append(job)
                ops.submit_async_batch_memcpy(
                    handle, job, desc["src"], desc["dst"], desc["sizes"], True
                )
            submitted = time.monotonic_ns()
            ops.wait_async_batch_memcpy(handle, ids)
            end = time.monotonic_ns()
            cpu_end = time.process_time_ns()
            finished, durations = ops.poll_async_batch_memcpy(handle)
            sessions, jobs, stamps, statuses, overflow, failure = (
                ops.poll_async_batch_memcpy_completion_ring(ring)
            )
            assert finished == ids == jobs
            assert sessions == [session] * args.jobs
            assert statuses == [1] * args.jobs and not overflow and not failure
            assert stamps == sorted(stamps) and begin <= stamps[0] <= stamps[-1] <= end
            actual = gpu[desc["samples"]].cpu().numpy()
            assert np.array_equal(actual, desc["expected"]), "sample byte mismatch"
            torch.accelerator.synchronize()
            chunk_plan = chunks(desc["sizes"].numpy(), chunk, 4096)
            output.put(
                dict(
                    event="cell",
                    rank=rank,
                    **cell,
                    begin_ns=begin,
                    end_ns=end,
                    submit_ns=submitted - begin,
                    cpu_ns=cpu_end - cpu_begin,
                    completion_ns=stamps,
                    cuda_duration_ms=durations,
                    bytes_per_job=PROFILES[cell["profile"]] * ROW_BYTES,
                    descriptors_per_job=desc["sizes"].numel(),
                    chunks_per_job=len(chunk_plan),
                    chunk_bytes=[item[1] for item in chunk_plan],
                    validated_bytes=len(actual),
                )
            )
            ops.destroy_async_batch_memcpy_submitter(handle)
            handle = None
        barrier.wait(300)
        del desc, descriptors, host, host_flat, actual
        region.cleanup()
        region = None
        ops.destroy_async_batch_memcpy_completion_ring(ring)
        ring = None
        output.put(dict(event="done", rank=rank))
    except BaseException:
        output.put(dict(event="error", rank=rank, traceback=traceback.format_exc()))
        barrier.abort()
    finally:
        if handle is not None:
            ops.destroy_async_batch_memcpy_submitter(handle)
        if ring is not None:
            ops.destroy_async_batch_memcpy_completion_ring(ring)
        if region is not None:
            region.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=8, choices=[8])
    parser.add_argument(
        "--chunks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512]
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--configs", nargs="+", help="Explicit chunkMiB:depth pairs")
    parser.add_argument("--seed", type=int, default=8109)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1 or args.repeats < 1:
        parser.error("jobs and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "plan.json").exists():
        parser.error("Use a fresh output directory")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    ).strip()
    if apps:
        parser.error(f"GPU processes are already active: {apps}")
    plan = []
    configs = (
        [tuple(map(int, value.split(":"))) for value in args.configs]
        if args.configs
        else [(chunk, depth) for chunk in args.chunks for depth in args.depths]
    )
    if any(chunk < 1 or not 1 <= depth <= 8 for chunk, depth in configs):
        parser.error("chunk must be positive and depth must be 1 through 8")
    for repeat in range(args.repeats):
        order = list(configs)
        random.Random(args.seed + repeat).shuffle(order)
        for chunk, depth in order:
            for profile in PROFILES:
                plan.append(
                    dict(
                        cell=len(plan),
                        repeat=repeat,
                        chunk_mib=chunk,
                        depth=depth,
                        profile=profile,
                    )
                )
    save(
        args.output / "plan.json",
        dict(
            ranks=args.ranks,
            jobs=args.jobs,
            cells=plan,
            page_sizes=PAGE_SIZES,
            profiles=PROFILES,
            row_bytes=ROW_BYTES,
            host_blocks=2048,
            gpu_blocks=1024,
            max_descriptors=4096,
            seed=args.seed,
            scope=(
                "Native transfer-only; no model compute or concurrent "
                "compute-copy interference"
            ),
            validation=(
                "First/middle/last block, first/last 32 bytes of each of 99 KV tensors"
            ),
        ),
    )
    context = mp.get_context("spawn")
    barrier, output = context.Barrier(args.ranks), context.Queue()
    engine = "h2dcal_" + uuid.uuid4().hex
    processes = [
        context.Process(target=worker, args=(r, args, plan, engine, barrier, output))
        for r in range(args.ranks)
    ]
    started = time.monotonic()
    errors, ready, done, counts = [], [], set(), {}
    try:
        for process in processes:
            process.start()
        save(args.output / "ownership.json", dict(pids=[p.pid for p in processes]))
        with (args.output / "events.jsonl").open("w") as log:
            while len(done) < args.ranks:
                try:
                    event = output.get(timeout=30)
                except queue.Empty:
                    if errors or not all(
                        p.is_alive()
                        for rank, p in enumerate(processes)
                        if rank not in done
                    ):
                        raise RuntimeError(
                            "Worker exited without completing calibration"
                        ) from None
                    if time.monotonic() - started > 1800:
                        raise TimeoutError("Calibration exceeded 30 minutes") from None
                    continue
                log.write(json.dumps(event) + "\n")
                log.flush()
                if event["event"] == "error":
                    errors.append(event)
                    raise RuntimeError(event["traceback"])
                if event["event"] == "ready":
                    ready.append(event)
                    print(
                        json.dumps(dict(event="rank_ready", rank=event["rank"])),
                        flush=True,
                    )
                elif event["event"] == "done":
                    done.add(event["rank"])
                elif event["event"] == "cell":
                    index = event["cell"]
                    counts[index] = counts.get(index, 0) + 1
                    if counts[index] == args.ranks:
                        save(
                            args.output / "progress.json",
                            dict(
                                completed=sum(v == args.ranks for v in counts.values()),
                                expected=len(plan),
                                latest=plan[index],
                            ),
                        )
                        if index % 12 == 0:
                            print(
                                json.dumps(dict(event="cell_done", **plan[index])),
                                flush=True,
                            )
        for process in processes:
            process.join(120)
            if process.exitcode != 0:
                raise RuntimeError(f"Worker exitcode {process.exitcode}")
        save(
            args.output / "summary.json",
            dict(
                complete=True,
                ready=ready,
                cells=len(plan),
                seconds=time.monotonic() - started,
            ),
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(30)
            if process.is_alive():
                process.kill()
                process.join(10)
        if errors:
            save(args.output / "errors.json", errors)


if __name__ == "__main__":
    main()
