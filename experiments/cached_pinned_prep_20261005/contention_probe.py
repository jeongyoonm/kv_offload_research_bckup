# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate preparation-thread interference with CUDA submission, without a model."""

import argparse
import contextlib
import dataclasses
import json
import pickle
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch

from vllm import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_offload import late_admission_runtime as runtime


class MeasuredLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows = []
        self.entered = 0

    def __enter__(self):
        start = time.perf_counter_ns()
        self.lock.acquire()
        self.entered = time.perf_counter_ns()
        self.rows.append([threading.current_thread().name, self.entered - start, 0])
        return self

    def __exit__(self, *_):
        self.rows[-1][2] = time.perf_counter_ns() - self.entered
        self.lock.release()


def describe(values):
    values = sorted(values)
    return dict(
        mean=statistics.mean(values),
        median=statistics.median(values),
        p95=values[min(len(values) - 1, int(len(values) * 0.95))],
        max=max(values),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--launches", type=int, default=2048)
    parser.add_argument("--prequeue", action="store_true")
    parser.add_argument("--launch-modes", default="eager,graph")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.accelerator.set_device_index(0)
    device = torch.device("cuda:0")
    tensor = torch.zeros(65536, device=device)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(32):
            tensor.add_(1)
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(32):
            tensor.add_(1)
    torch.accelerator.synchronize()
    requests = []
    length, count = 61440, 10
    for i in range(count):
        tokens = ((np.arange(length) * 17 + i * 100) % 200000).tolist()
        requests.append(
            NewRequestData(
                f"r{i}",
                tokens,
                [],
                SamplingParams(max_tokens=1),
                None,
                (list(range(length // 64)),),
                length - 3072,
                None,
                prefill_token_ids=tokens.copy(),
            )
        )
    array_requests = [
        dataclasses.replace(
            r,
            prompt_token_ids=np.asarray(r.prompt_token_ids, dtype=np.int32),
            prefill_token_ids=np.asarray(r.prefill_token_ids, dtype=np.int32),
        )
        for r in requests
    ]
    pool = [torch.empty(length, dtype=torch.int32, pin_memory=True) for _ in requests]
    state = dict(mode="idle", stages=[], pool_index=0)
    original_load, original_array = pickle.load, np.asarray
    original_copy, original_empty = np.copyto, torch.empty

    @contextlib.contextmanager
    def measured(name):
        start = time.perf_counter_ns()
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
            state["stages"].append(
                dict(stage=name, begin_ns=start, end_ns=time.perf_counter_ns())
            )

    def load(*a, **kw):
        with measured("prep_pickle_load"):
            return original_load(*a, **kw)

    def array(*a, **kw):
        with measured("prep_list_to_numpy"):
            return original_array(*a, **kw)

    def copy(*a, **kw):
        with measured("prep_numpy_copy"):
            return original_copy(*a, **kw)

    def empty(*a, **kw):
        if not kw.get("pin_memory"):
            return original_empty(*a, **kw)
        with measured("prep_pinned_allocation"):
            if state["mode"] == "pinned_reuse":
                result = pool[state["pool_index"]]
                state["pool_index"] += 1
                return result
            return original_empty(*a, **kw)

    runtime.signal_prepared = lambda *a, **kw: None
    done = threading.Event()
    lane = runtime.PreparationLane(
        0,
        1,
        candidate_slots=count,
        cached_requests=True,
        pinned_packing=True,
        device=device,
    )
    lane.lock = MeasuredLock()
    original_prepare = lane._prepare_path

    def prepare(path):
        try:
            with measured("preparation_packet"):
                if state["mode"] in ["decode", "decode_arrays"]:
                    with open(path, "rb") as f:
                        load(f)
                else:
                    original_prepare(path)
        finally:
            done.set()

    lane._prepare_path = prepare
    pickle.load, np.asarray, np.copyto, torch.empty = load, array, copy, empty
    rows = []
    modes = [
        "idle",
        "notice",
        "decode",
        "decode_arrays",
        "references",
        "pinned",
        "pinned_reuse",
    ]
    launch_modes = args.launch_modes.split(",")
    try:
        for launch_mode in launch_modes:
            for repeat in range(args.repeats + 2):
                order = modes if repeat % 2 == 0 else list(reversed(modes))
                for mode in order:
                    step = len(rows) + 1
                    candidates = (
                        [runtime.CandidateNotice(r.req_id) for r in requests]
                        if mode == "notice"
                        else requests
                    )
                    if mode == "decode_arrays":
                        candidates = array_requests
                    path = args.root / "packet.pkl"
                    path.write_bytes(
                        pickle.dumps(
                            (1, step, [], candidates, None),
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    )
                    state.update(mode=mode, stages=[], pool_index=0)
                    lane.notice_only = mode == "notice"
                    lane.pinned_packing = mode in ["pinned", "pinned_reuse"]
                    lane.lock.rows.clear()
                    done.clear()
                    torch.accelerator.synchronize()
                    start_event, end_event = [
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    ]
                    torch.cuda.nvtx.range_push(f"{launch_mode}:{mode}:{repeat}")
                    start = time.perf_counter_ns()
                    start_event.record()
                    if mode != "idle" and not args.prequeue:
                        os_path = str(path) + "\n"
                        import os

                        os.write(lane.pipe, os_path.encode())
                    latencies = []
                    submit = (
                        graph.replay
                        if launch_mode == "graph"
                        else lambda: tensor.add_(1)
                    )
                    for _ in range(args.launches):
                        before = time.perf_counter_ns()
                        submit()
                        latencies.append((time.perf_counter_ns() - before) / 1e6)
                    submitted = time.perf_counter_ns()
                    end_event.record()
                    active_at_prep_start = None
                    if mode != "idle" and args.prequeue:
                        active_at_prep_start = not end_event.query()
                        import os

                        os.write(lane.pipe, (str(path) + "\n").encode())
                    end_event.synchronize()
                    finished = time.perf_counter_ns()
                    torch.cuda.nvtx.range_pop()
                    if mode != "idle":
                        assert done.wait(5), "preparation did not complete"
                    before_consume = time.perf_counter_ns()
                    adopted = lane.consume(step, {r.req_id for r in requests})
                    consumed = time.perf_counter_ns()
                    if mode not in ["idle", "decode", "decode_arrays"]:
                        assert len(adopted) == count
                        assert all(p.ready.is_set() for p in adopted.values())
                    row = dict(
                        launch_mode=launch_mode,
                        mode=mode,
                        repeat=repeat,
                        measured=repeat >= 2,
                        prequeued=args.prequeue,
                        gpu_active_at_prep_start=active_at_prep_start,
                        submit_ms=(submitted - start) / 1e6,
                        gpu_span_ms=start_event.elapsed_time(end_event),
                        wall_ms=(finished - start) / 1e6,
                        call_ms=describe(latencies),
                        consume_ms=(consumed - before_consume) / 1e6,
                        mutex_wait_ms=sum(x[1] for x in lane.lock.rows) / 1e6,
                        mutex_hold_ms=sum(x[2] for x in lane.lock.rows) / 1e6,
                        stages=list(state["stages"]),
                    )
                    rows.append(row)
                    del adopted
            print("FINISHED", launch_mode, flush=True)
    finally:
        lane.close()
        pickle.load, np.asarray, np.copyto, torch.empty = (
            original_load,
            original_array,
            original_copy,
            original_empty,
        )
    (args.root / "probe_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = []
    for launch_mode in launch_modes:
        for mode in modes:
            selected = [
                r
                for r in rows
                if r["measured"]
                and r["mode"] == mode
                and r["launch_mode"] == launch_mode
            ]
            item = dict(launch_mode=launch_mode, mode=mode, repeats=len(selected))
            for key in [
                "submit_ms",
                "gpu_span_ms",
                "wall_ms",
                "consume_ms",
                "mutex_wait_ms",
                "mutex_hold_ms",
            ]:
                item[key] = describe([r[key] for r in selected])
            item["worst_submit_call_ms"] = describe(
                [r["call_ms"]["max"] for r in selected]
            )
            summary.append(item)
    (args.root / "probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
