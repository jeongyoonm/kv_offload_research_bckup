# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in same-process native H2D alongside unchanged real model forwards."""

import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from benchmarks.kernels.calibrate_async_h2d import PAGE_SIZES, PROFILES, ROW_BYTES
from vllm import _custom_ops as ops
from vllm.distributed import get_tp_group
from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.execution_timing import NativeExecutionObserver
from vllm.v1.kv_offload.step_timing import RankFile


class Calibration:
    def __init__(self, worker, root):
        self.worker = worker
        self.root = Path(root)
        self.rank = worker.rank
        self.cell = None
        self.handle = None
        self.job = 0
        self.records = []
        self.current = None
        self.forward_count = 0
        self.ids = []
        self.session = 1009
        self.device = worker.device
        assert worker.use_v2_model_runner
        self.runner = worker.model_runner
        path = str(self.root / f"execution-{self.rank}")
        self.timing = RankFile(path, create=True)
        self.observer = NativeExecutionObserver(
            path,
            self.device,
            "/mnt/localnvme/glm53_dataset_coverage_20261007/"
            "libseal_execution_observer.so",
        )
        self.ring_path = str(self.root / f"copy-{self.rank}")
        self.ring = ops.create_async_batch_memcpy_completion_ring(
            self.ring_path, self.session, 4096
        )
        self.region = SharedOffloadRegion(
            engine_id="h2dinterference_" + uuid.uuid4().hex,
            num_blocks=2048,
            rank=self.rank,
            kv_bytes_per_block=(8 * ROW_BYTES + 4095) // 4096 * 4096,
            cpu_page_size=ROW_BYTES,
            barrier=get_tp_group().barrier,
            population_mode="contiguous",
            population_workers=8,
            num_workers=8,
            process_rank=self.rank,
            memory_layout="private",
        )
        host = self.region._base.numpy().view(np.uint8).reshape(2048, ROW_BYTES)
        tags = (np.arange(2048, dtype=np.uint16) + self.rank * 19) % 256
        offset = 0
        for index, page in enumerate(PAGE_SIZES):
            pattern = (np.arange(page, dtype=np.uint16) + index * 7) % 256
            np.add(
                tags[:, None],
                pattern[None, :],
                out=host[:, offset : offset + page],
                casting="unsafe",
            )
            offset += page
        pin_mmap_region(self.region)
        assert self.region.is_pinned
        self.gpu = torch.empty(1024 * ROW_BYTES, dtype=torch.uint8, device=self.device)
        rng = np.random.default_rng(8100 + self.rank)
        source_ids, destination_ids = rng.permutation(2048), rng.permutation(1024)
        self.descriptors = {}
        for name, blocks in PROFILES.items():
            sources, destinations, sizes, src_samples, dst_samples = [], [], [], [], []
            source_offset = destination_offset = 0
            for page in PAGE_SIZES:
                src = source_ids[:blocks] * ROW_BYTES + source_offset
                dst = destination_ids[:blocks] * page + destination_offset
                sources.append(src + self.region._base.data_ptr())
                destinations.append(dst + self.gpu.data_ptr())
                sizes.append(np.full(blocks, page, dtype=np.int64))
                points = np.r_[np.arange(32), np.arange(page - 32, page)]
                picks = [0, blocks // 2, blocks - 1]
                src_samples.append((src[picks, None] + points).reshape(-1))
                dst_samples.append((dst[picks, None] + points).reshape(-1))
                source_offset += page
                destination_offset += 1024 * page
            self.descriptors[name] = dict(
                src=torch.from_numpy(np.concatenate(sources)).pin_memory(),
                dst=torch.from_numpy(np.concatenate(destinations)).pin_memory(),
                sizes=torch.from_numpy(np.concatenate(sizes)).pin_memory(),
                samples=torch.from_numpy(np.concatenate(dst_samples)).to(self.device),
                expected=host.reshape(-1)[np.concatenate(src_samples)].copy(),
            )
        self.original_pre = self.runner.kv_connector.pre_forward
        self.original_execute = self.runner.execute_model
        self.runner.kv_connector.pre_forward = self.pre_forward
        self.runner.execute_model = self.execute
        torch.accelerator.synchronize()

    def submit(self):
        desc = self.descriptors[self.cell["profile"]]
        begin = time.monotonic_ns()
        self.ids = []
        for _ in range(self.cell["jobs"]):
            self.job += 1
            self.ids.append(self.job)
            ops.submit_async_batch_memcpy(
                self.handle, self.job, desc["src"], desc["dst"], desc["sizes"], True
            )
        self.current.update(io_begin_ns=begin, io_enqueued_ns=time.monotonic_ns())

    def pre_forward(self, scheduler_output):
        result = self.original_pre(scheduler_output)
        if self.cell is not None:
            assert self.current is None, "multiple forwards in one calibration cell"
            self.forward_count += 1
            self.current = dict(
                host_begin_ns=time.monotonic_ns(),
                tokens=dict(scheduler_output.num_scheduled_tokens),
            )
            torch.cuda.nvtx.range_push(
                f"calibration/{self.cell['mode']}/c{self.cell['chunk_mib']}/"
                f"d{self.cell['depth']}/{self.cell['profile']}/{self.cell['shape']}"
            )
            self.observer.start(self.cell["id"])
            if self.cell["mode"] == "concurrent":
                self.submit()
            self.current["compute_submit_begin_ns"] = time.monotonic_ns()
        return result

    def execute(self, *args, **kwargs):
        before = getattr(self, "forward_count", 0)
        result = self.original_execute(*args, **kwargs)
        if self.cell is not None and self.forward_count > before:
            now = time.monotonic_ns()
            self.current["compute_enqueued_ns"] = now
            self.observer.finish(
                self.cell["id"],
                None,
                now,
                stream=torch.cuda.current_stream(self.device),
            )
            batch = self.runner.execute_model_state.input_batch
            self.current.update(
                suffix=batch.num_scheduled_tokens.tolist(),
                context=batch.num_computed_tokens_np.tolist(),
                padded_tokens=batch.num_tokens_after_padding,
            )
            if self.cell["mode"] == "after_enqueued":
                self.submit()
            torch.cuda.nvtx.range_pop()
        return result

    def arm(self, cell):
        assert self.cell is None and self.handle is None
        self.cell, self.current, self.forward_count, self.ids = cell, None, 0, []
        if cell["mode"] != "compute_only":
            assert cell["depth"] in (1, 2)
            self.handle = ops.create_async_batch_memcpy_submitter(
                self.rank,
                cell["chunk_mib"] * 1024**2,
                4096,
                self.ring_path,
                self.session,
            )
            ops.configure_async_batch_memcpy_pipeline(self.handle, cell["depth"])
            # Untimed native warmup precedes every cell, including confirmations.
            self.current = {}
            self.submit()
            ops.wait_async_batch_memcpy(self.handle, self.ids)
            finished, _ = ops.poll_async_batch_memcpy(self.handle)
            ring = ops.poll_async_batch_memcpy_completion_ring(self.ring)
            assert finished == self.ids == ring[1]
            assert ring[3] == [1] * cell["jobs"] and not ring[4] and not ring[5]
            self.current, self.ids = None, []
        torch.accelerator.synchronize()
        get_tp_group().barrier()
        return dict(rank=self.rank, armed=cell["id"])

    def copy_only(self):
        assert self.cell["mode"] == "copy_only"
        get_tp_group().barrier()
        self.current = dict(host_begin_ns=time.monotonic_ns())
        self.submit()

    def drain(self):
        assert self.cell is not None and self.current is not None
        cell, row = self.cell, self.current
        if cell["mode"] != "copy_only":
            assert self.forward_count == 1
            deadline = time.monotonic() + 120
            while True:
                execution = self.timing.read(cell["id"])
                if execution is not None and execution[8]:
                    assert execution[5] == 1 and execution[6] > 0
                    break
                assert time.monotonic() < deadline, "native compute timing timeout"
                time.sleep(0.001)
            row["execution"] = execution
        if self.ids:
            ops.wait_async_batch_memcpy(self.handle, self.ids)
            finished, durations = ops.poll_async_batch_memcpy(self.handle)
            session, jobs, stamps, status, overflow, failure = (
                ops.poll_async_batch_memcpy_completion_ring(self.ring)
            )
            assert finished == self.ids == jobs
            assert session == [self.session] * cell["jobs"]
            assert status == [1] * cell["jobs"] and not overflow and not failure
            assert stamps == sorted(stamps)
            row.update(
                completion_ns=stamps,
                copy_cuda_ms=durations,
                bytes_per_job=PROFILES[cell["profile"]] * ROW_BYTES,
            )
        torch.accelerator.synchronize()
        get_tp_group().barrier()
        if self.ids:
            desc = self.descriptors[cell["profile"]]
            actual = self.gpu[desc["samples"]].cpu().numpy()
            assert np.array_equal(actual, desc["expected"]), "KV sample mismatch"
            row["validated_bytes"] = len(actual)
        get_tp_group().barrier()
        if self.handle is not None:
            ops.destroy_async_batch_memcpy_submitter(self.handle)
            self.handle = None
        self.cell, self.current = None, None
        return dict(rank=self.rank, **cell, **row)

    def close(self):
        assert self.cell is None and self.handle is None
        self.runner.kv_connector.pre_forward = self.original_pre
        self.runner.execute_model = self.original_execute
        self.observer.close()
        self.timing.close()
        ops.destroy_async_batch_memcpy_completion_ring(self.ring)
        self.descriptors.clear()
        self.gpu = None
        self.region.cleanup()


def setup(worker, root):
    from tools.benchmark_cpu_cache_init import _thp_policy

    assert _thp_policy("disabled") == 1
    worker._h2d_calibration = Calibration(worker, root)
    return dict(
        rank=worker.rank,
        pid=os.getpid(),
        cpus=sorted(os.sched_getaffinity(0)),
        init=worker._h2d_calibration.region.init_stats,
    )


def arm(worker, cell):
    return worker._h2d_calibration.arm(cell)


def copy_only(worker):
    worker._h2d_calibration.copy_only()


def drain(worker):
    return worker._h2d_calibration.drain()


def close(worker):
    worker._h2d_calibration.close()
    return dict(rank=worker.rank, closed=True)


class CalibrationExtension:
    """Named control RPCs; calibration does not require callable serialization."""

    def calibration_setup(self, root):
        return setup(self, root)

    def calibration_arm(self, cell):
        return arm(self, cell)

    def calibration_copy_only(self):
        return copy_only(self)

    def calibration_drain(self):
        return drain(self)

    def calibration_close(self):
        return close(self)

    def calibration_profile_begin(self):
        torch.cuda.profiler.start()

    def calibration_profile_end(self):
        torch.cuda.profiler.stop()
