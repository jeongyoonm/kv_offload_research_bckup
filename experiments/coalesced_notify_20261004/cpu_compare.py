# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched eight-process replay through the production preparation/seal loop."""

import ctypes
import json
import mmap
import multiprocessing as mp
import queue
import struct
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))

from vllm.v1.kv_offload.late_admission_runtime import (  # noqa: E402
    CandidateNotice,
    signal_prepared,
    stage_rpc,
)
from vllm.v1.kv_offload.seal_notifications import (  # noqa: E402
    NotificationQueue,
    SealNotifications,
)
from vllm.v1.kv_offload.step_timing import RankFile  # noqa: E402

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_coalesced_notify_20261004/cpu-comparison")
LIBRARY = ROOT.parent / "build/libseal_native_harness.so"
CANDIDATES = 10


def worker(rank, commands, results, ring_path):
    lib = ctypes.CDLL(str(LIBRARY))
    lib.seal_open.argtypes = [ctypes.c_char_p]
    lib.seal_open.restype = ctypes.c_void_p
    lib.seal_close.argtypes = [ctypes.c_void_p]
    lib.seal_kv.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
    handle = lib.seal_open(ring_path.encode())
    assert handle
    with open(ring_path, "r+b") as stream:
        ring = mmap.mmap(stream.fileno(), 64 + CANDIDATES * 8)
    results.put((rank, "ready"))
    try:
        while (command := commands.get()) is not None:
            control, step, begin, spacing_ns, skew_ns = command
            cpu = time.process_time_ns()
            signal_prepared(control, step, rank, 1)
            for slot in range(CANDIDATES):
                signal_prepared(control, step, rank, 1, slot)
            for slot in range(CANDIDATES):
                target = begin + 5_000_000 + slot * spacing_ns + rank * skew_ns
                remaining = target - time.monotonic_ns()
                if remaining > 0:
                    time.sleep(remaining / 1e9)
                completed = time.monotonic_ns()
                struct.pack_into("Q", ring, 64 + slot * 8, completed)
                struct.pack_into("Q", ring, 24, slot + 1)
                assert lib.seal_kv(handle, step * 32 + slot, 1)
            results.put((rank, time.process_time_ns() - cpu))
    finally:
        ring.close()
        lib.seal_close(handle)


class HeldResult:
    def poll(self):
        return False


class Executor:
    def __init__(self, commands, spacing_ns, skew_ns):
        self.commands = commands
        self.spacing_ns, self.skew_ns = spacing_ns, skew_ns

    def publish_late_preparation(self, path):
        import pickle

        with open(path, "rb") as stream:
            _, step, _, _, control = pickle.load(stream)
        begin = time.monotonic_ns() + 2_000_000
        for command in self.commands:
            command.put((control, step, begin, self.spacing_ns, self.skew_ns))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    channel = SealNotifications(str(ROOT / "notifications"), 123, 8)
    paths, rings, ranks = [], [], []
    context = mp.get_context("spawn")
    commands = [context.Queue() for _ in range(8)]
    results = context.Queue()
    workers = []
    rows = []
    try:
        for rank in range(8):
            path = str(ROOT / f"ring-{rank}")
            Path(path).write_bytes(bytes(64 + CANDIDATES * 8))
            channel.descriptor(path, rank)
            paths.append(path)
            with open(path, "r+b") as stream:
                rings.append(mmap.mmap(stream.fileno(), 64 + CANDIDATES * 8))
            ranks.append(RankFile(str(ROOT / f"execution-{rank}"), create=True))
            process = context.Process(
                target=worker, args=(rank, commands[rank], results, path)
            )
            process.start()
            workers.append(process)
        assert {results.get(timeout=60) for _ in workers} == {
            (rank, "ready") for rank in range(8)
        }
        step = 10
        for skew_ns in (0, 1_000_000, 2_000_000):
            for repeat in range(6):
                modes = (False, True) if repeat % 2 == 0 else (True, False)
                for coalesced in modes:
                    step += 2
                    for ring in rings:
                        ring[:] = bytes(len(ring))
                    start = time.monotonic_ns()
                    for rank in ranks:
                        rank.write([step - 1, start, start, start, start, 1, 0, 0, 0])
                    scheduler = NS(
                        _late_generation=1,
                        _late_seal_policy="predictive",
                        _late_seal_full_prediction_wait=True,
                        _late_seal_coalesced_notifications=coalesced,
                        _late_seal_prediction_to_readiness=True,
                        _late_seal_condition_preparation=True,
                        _late_seal_hybrid_guard_us=50_000,
                        _late_seal_adaptive_poll_us=2000,
                        _late_seal_target_candidates=16,
                        _late_seal_repeat_us=25_000,
                        _late_seal_prediction_scale_percent=100,
                        _late_seal_prediction_reserve_us=10_000,
                        _late_seal_safety_us=2_000_000,
                        _late_base_request_shapes=[],
                        _late_seal_wait_trace=True,
                        finished_recving_kv_req_ids=set(),
                        requests={},
                        connector=NS(
                            connector_scheduler=NS(
                                _req_status={
                                    f"c{slot}": NS(transfer_jobs={step * 32 + slot})
                                    for slot in range(CANDIDATES)
                                }
                            )
                        ),
                    )
                    readiness = NS(
                        paths=paths,
                        seal_notifications=channel,
                        _released_jobs=set(),
                        _rank_reports={},
                    )

                    def refresh(
                        step=step, readiness=readiness, scheduler=scheduler, **kwargs
                    ):
                        for slot in range(CANDIDATES):
                            reports = {
                                rank
                                for rank, ring in enumerate(rings)
                                if struct.unpack_from("Q", ring, 64 + slot * 8)[0]
                            }
                            job = step * 32 + slot
                            readiness._rank_reports[job] = reports
                            if len(reports) == 8:
                                scheduler.finished_recving_kv_req_ids.add(f"c{slot}")
                                readiness._released_jobs.add(job)

                    core = NS(
                        scheduler=scheduler,
                        vllm_config=NS(parallel_config=NS(world_size=8)),
                        model_executor=Executor(commands, 2_500_000, skew_ns),
                        early_load_readiness=readiness,
                        aborts_queue=NotificationQueue(channel)
                        if coalesced
                        else queue.Queue(),
                        batch_queue=deque(
                            [
                                (
                                    HeldResult(),
                                    NS(
                                        total_num_scheduled_tokens=1024,
                                        readiness_timeline_step=step - 1,
                                    ),
                                    None,
                                )
                            ]
                        ),
                        _refresh_early_load_readiness=refresh,
                        deferral_timing=NS(
                            execution_ranks=ranks,
                            steps={step - 1: {"request_shapes": [[1024, 8192]]}},
                            read_execution=lambda current: [
                                r.read(current) for r in ranks
                            ],
                            seal_estimator=NS(
                                predict=lambda data: 800_000_000,
                                preparation_reserve=lambda extra, upcoming=None: extra,
                                features=lambda data: np.ones(4),
                                samples=[0] * 8,
                                preparation_ns=[],
                            ),
                        ),
                    )
                    cpu = time.process_time_ns()
                    admitted = stage_rpc(
                        core,
                        step,
                        [],
                        [CandidateNotice(f"c{slot}") for slot in range(CANDIDATES)],
                    )
                    engine_cpu_ns = time.process_time_ns() - cpu
                    producer_cpu = [results.get(timeout=10) for _ in workers]
                    assert {rank for rank, _ in producer_cpu} == set(range(8))
                    assert len(admitted) == CANDIDATES
                    transport = scheduler._late_transport
                    all_ready_ns = max(
                        struct.unpack_from("Q", ring, 64 + slot * 8)[0]
                        for ring in rings
                        for slot in range(CANDIDATES)
                    )
                    row = dict(
                        mode="coalesced" if coalesced else "bridge",
                        rank_skew_ms=skew_ns / 1e6,
                        repeat=repeat,
                        engine_cpu_ms=engine_cpu_ns / 1e6,
                        worker_cpu_ms=sum(value for _, value in producer_cpu) / 1e6,
                        ready_to_seal_ms=(transport["seal_ns"] - all_ready_ns) / 1e6,
                        progress_polls=transport["progress_polls"],
                        coarse_wakes=sum(
                            w["phase"] == "coarse" and w["reason"] == "progress"
                            for w in transport["adaptive_waits"]
                        ),
                        bridge_scans=transport["progress_bridge_scans"],
                        admissions=len(admitted),
                    )
                    row["total_cpu_ms"] = row["engine_cpu_ms"] + row["worker_cpu_ms"]
                    rows.append(row)
                    (ROOT / "cells.json").write_text(json.dumps(rows, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
    finally:
        for command in commands:
            command.put(None)
        for process in workers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for rank in ranks:
            rank.close()
        for ring in rings:
            ring.close()
        channel.close()


if __name__ == "__main__":
    main()
