# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure shared CPU-cache startup without loading a model.

Run from the source checkout with .venv/bin/python. Every cell uses a fresh
mapping and releases its CUDA registrations and shared memory before exiting.
"""

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import queue
import subprocess
import time
import traceback
import uuid
from pathlib import Path


def _cpu_list(value: str) -> list[int]:
    result = []
    for part in value.strip().split(","):
        bounds = [int(x) for x in part.split("-")]
        result.extend(range(bounds[0], bounds[-1] + 1))
    return result


def _bind(rank: int, workers: int, policy: str) -> dict:
    nodes = sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))
    node = nodes[min(rank * len(nodes) // workers, len(nodes) - 1)]
    cpus = set(_cpu_list((node / "cpulist").read_text()))
    cpus &= os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    if policy != "first-touch":
        lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
        lib.set_mempolicy.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
        ]
        lib.set_mempolicy.restype = ctypes.c_int
        node_id = int(node.name.removeprefix("node"))
        mask = (1 << node_id) if policy == "local" else (1 << len(nodes)) - 1
        value = ctypes.c_ulong(mask)
        # MPOL_BIND=2, MPOL_INTERLEAVE=3; policy is local to this thread.
        mode = 2 if policy == "local" else 3
        if lib.set_mempolicy(mode, ctypes.byref(value), len(nodes) + 1):
            raise OSError(ctypes.get_errno(), "set_mempolicy failed")
    return {"node": node.name, "cpus": sorted(cpus), "policy": policy}


def _mapping_snapshot(engine_id: str) -> dict:
    numa = [
        line
        for line in Path("/proc/self/numa_maps").read_text().splitlines()
        if engine_id in line
    ]
    fields = {}
    selected = False
    for line in Path("/proc/self/smaps").read_text().splitlines():
        if line and "-" in line.split()[0]:
            selected = engine_id in line
        elif selected and ":" in line:
            key, value = line.split(":", 1)
            if key in (
                "Size",
                "Rss",
                "Pss",
                "Locked",
                "ShmemPmdMapped",
                "KernelPageSize",
                "MMUPageSize",
                "AnonHugePages",
            ):
                fields[key] = fields.get(key, 0) + int(value.split()[0])
    return {"numa_maps": numa, "smaps_kib": fields}


def _external_gpu_pids(allowed: set[int]) -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return [
        int(line)
        for line in result.stdout.splitlines()
        if line.strip().isdigit() and int(line) not in allowed
    ]


def _worker(rank, args, engine_id, num_blocks, row_bytes, barrier, semaphore, out):
    region = None
    try:
        binding = _bind(rank, args.workers, args.numa)
        import torch

        from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if args.cuda:
            torch.accelerator.set_device_index(rank)
            torch.empty(1, device="cuda")
            torch.accelerator.synchronize()
        barrier.wait(args.timeout)
        start_ns = time.monotonic_ns()
        region = SharedOffloadRegion(
            engine_id=engine_id,
            num_blocks=num_blocks,
            rank=rank,
            kv_bytes_per_block=row_bytes,
            cpu_page_size=args.worker_block_bytes,
            barrier=lambda: barrier.wait(args.timeout),
            population_mode=args.population,
            population_workers=args.population_workers,
            population_chunk_bytes=args.chunk_mib * 1024**2,
            num_workers=args.workers,
            process_rank=rank,
            population_barrier=args.synchronize_population,
        )
        before = _mapping_snapshot(engine_id)
        if args.cuda:
            from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region

            wait_start = time.perf_counter()
            with semaphore:
                region.init_stats["registration_queue_seconds"] = (
                    time.perf_counter() - wait_start
                )
                pin_mmap_region(region)
            if not region.is_pinned:
                raise RuntimeError("CUDA registration failed")
        ready_ns = time.monotonic_ns()
        after = _mapping_snapshot(engine_id)
        out.put({"event": "ready", "rank": rank, "stats": region.init_stats})
        barrier.wait(args.timeout)

        # Exercise strided first/middle/last block views before measuring DMA.
        view = region.create_next_worker_view(args.worker_block_bytes)
        selected = [0, num_blocks // 2, num_blocks - 1]
        for block in selected:
            view[block, :16] = rank + 1
        barrier.wait(args.timeout)
        for block in selected:
            assert bool(torch.all(view[block, :16] == rank + 1))
        del view
        barrier.wait(args.timeout)
        bandwidth = {}
        if args.cuda:
            # Disjoint contiguous transfer ranges, one per rank. Rank layout
            # uses the same boundaries; all ranges are inside registrations.
            transfer_bytes = min(64 * 1024**2, region.total_size_bytes // args.workers)
            offset = rank * (region.total_size_bytes // args.workers)
            host = region._base[offset : offset + transfer_bytes]
            source = torch.full(
                (transfer_bytes,), rank + 17, dtype=torch.int8, device="cuda"
            )
            restored = torch.empty_like(source)
            for direction, destination, origin in (
                ("d2h", host, source),
                ("h2d", restored, host),
            ):
                destination.copy_(origin, non_blocking=True)
                torch.accelerator.synchronize()
                begin = time.perf_counter()
                for _ in range(8):
                    destination.copy_(origin, non_blocking=True)
                torch.accelerator.synchronize()
                bandwidth[direction + "_gbps"] = (
                    8 * transfer_bytes / (time.perf_counter() - begin) / 1e9
                )
            assert torch.equal(source, restored)
            del host, source, restored, destination, origin
        barrier.wait(args.timeout)
        stats = dict(region.init_stats)
        cleanup_start = time.perf_counter()
        region.cleanup()
        region = None
        out.put(
            {
                "event": "done",
                "rank": rank,
                "start_ns": start_ns,
                "ready_ns": ready_ns,
                "stats": stats,
                "binding": binding,
                "before_registration": before,
                "after_registration": after,
                "bandwidth": bandwidth,
                "cleanup_seconds": time.perf_counter() - cleanup_start,
            }
        )
    except BaseException:
        out.put({"event": "error", "rank": rank, "traceback": traceback.format_exc()})
        barrier.abort()
        if region is not None:
            region.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--capacity-gib", type=float, default=16)
    size.add_argument("--capacity-bytes", type=int)
    parser.add_argument("--workers", type=int, default=8)
    # GLM-like mixed-page geometry; override with measured model geometry.
    parser.add_argument("--worker-block-bytes", type=int, default=3052800)
    parser.add_argument("--population", choices=("rank", "contiguous"), default="rank")
    parser.add_argument("--population-workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=256)
    parser.add_argument("--synchronize-population", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--registration-concurrency", type=int, default=8)
    parser.add_argument(
        "--numa", choices=("first-touch", "local", "interleave"), default="local"
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or args.worker_block_bytes < 16:
        parser.error("workers and worker-block-bytes must be positive")
    if not 1 <= args.registration_concurrency <= args.workers:
        parser.error("registration-concurrency must be within workers")
    capacity = args.capacity_bytes or int(args.capacity_gib * 1024**3)
    row_bytes = (args.workers * args.worker_block_bytes + 4095) // 4096 * 4096
    num_blocks = capacity // row_bytes
    total_bytes = num_blocks * row_bytes
    if num_blocks < 1:
        parser.error("capacity must hold at least one block")
    shm = os.statvfs("/dev/shm")
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if total_bytes > min(shm.f_bavail * shm.f_frsize, available * 0.8):
        parser.error("capacity exceeds free shared memory or 80% of available RAM")
    other = _external_gpu_pids(set())
    if other:
        parser.error(f"other GPU work is running: {other}; leaving it undisturbed")
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.workers)
    semaphore = ctx.Semaphore(args.registration_concurrency)
    out = ctx.Queue()
    engine_id = "initbench_" + uuid.uuid4().hex
    processes = [
        ctx.Process(
            target=_worker,
            args=(
                rank,
                args,
                engine_id,
                num_blocks,
                row_bytes,
                barrier,
                semaphore,
                out,
            ),
        )
        for rank in range(args.workers)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    events = []
    run_start = time.monotonic()
    try:
        for process in processes:
            process.start()
        last_check = time.monotonic()
        while len(records) < args.workers:
            if time.monotonic() - run_start > args.timeout:
                raise TimeoutError("initialization harness timed out")
            if time.monotonic() - last_check > 15:
                other = _external_gpu_pids({p.pid for p in processes})
                if other:
                    raise RuntimeError(
                        f"other GPU work started: {other}; stopping cell"
                    )
                last_check = time.monotonic()
            try:
                event = out.get(timeout=1)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes):
                    raise RuntimeError("worker exited unexpectedly") from None
                continue
            events.append(event)
            print(
                json.dumps(
                    {
                        key: event[key]
                        for key in ("event", "rank", "stats", "bandwidth", "traceback")
                        if key in event
                    }
                ),
                flush=True,
            )
            if event["event"] == "error":
                raise RuntimeError(event["traceback"])
            if event["event"] == "done":
                records.append(event)
        for process in processes:
            process.join(30)
            if process.exitcode != 0:
                raise RuntimeError(f"worker {process.pid} failed to exit")
        start_ns = min(r["start_ns"] for r in records)
        populate_end = max(r["stats"]["populate_end_ns"] for r in records)
        summary = {
            "population_wall_seconds": (populate_end - start_ns) / 1e9,
            "ready_wall_seconds": (max(r["ready_ns"] for r in records) - start_ns)
            / 1e9,
            "registration_coverage_bytes": sum(
                r["stats"].get("register_bytes", 0) for r in records
            ),
            "population_calls": sum(r["stats"]["populate_calls"] for r in records),
            "correctness": "passed",
        }
        payload = {
            "args": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "actual_capacity_bytes": total_bytes,
            "num_blocks": num_blocks,
            "row_bytes": row_bytes,
            "summary": summary,
            "workers": sorted(records, key=lambda r: r["rank"]),
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"summary": summary}), flush=True)
    except BaseException:
        args.output.with_suffix(".failed.json").write_text(
            json.dumps(
                {"traceback": traceback.format_exc(), "events": events}, indent=2
            )
        )
        raise
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join(10)
        Path(f"/dev/shm/vllm_offload_{engine_id}.mmap").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
