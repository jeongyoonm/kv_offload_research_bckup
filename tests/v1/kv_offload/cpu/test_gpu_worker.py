# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import concurrent.futures
import logging
import random
import threading
import time
import uuid
from unittest.mock import MagicMock

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    TransferResult,
)
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

NUM_GPU_BLOCKS = [64]
NUM_CPU_BLOCKS = [256]
GPU_PAGE_SIZES = [512, 1024]
BLOCKS_PER_CHUNK_VALUES = [1, 3]
NUM_TENSORS = [4]
SEEDS = [0]
DEVICE_TYPE = current_platform.device_type
DEVICES = [f"{DEVICE_TYPE}:0"]
NUM_MAPPINGS = [3]
NUM_MAPPINGS_PER_GROUP = [2]


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific test")
def test_rocm_cpu_to_gpu_uses_dma(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_worker, "HAS_TRITON", True)
    monkeypatch.setattr(gpu_worker.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(gpu_worker.current_platform, "is_rocm", lambda: True)

    refs = [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=512)]]
    assert gpu_worker._select_swap_blocks_fn(refs, gpu_to_cpu=False) is (
        ops.swap_blocks_batch
    )


def test_worker_shutdown_releases_region_and_runs_both_handlers() -> None:
    """Both directions drain before the worker releases its shared region."""
    worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    calls: list[str] = []

    def record_store_shutdown() -> bool:
        calls.append("store")
        return True

    def record_load_shutdown() -> bool:
        calls.append("load")
        return True

    store_handler = MagicMock()
    load_handler = MagicMock()
    store_handler.shutdown.side_effect = record_store_shutdown
    load_handler.shutdown.side_effect = record_load_shutdown

    def record_region_cleanup(**_: bool) -> bool:
        calls.append("region")
        return True

    mmap_region = MagicMock()
    mmap_region.cleanup.side_effect = record_region_cleanup
    worker._store_handler = store_handler
    worker._load_handler = load_handler
    worker._mmap_region = mmap_region

    worker.shutdown()

    assert calls == ["store", "load", "region"]
    store_handler.shutdown.assert_called_once_with()
    load_handler.shutdown.assert_called_once_with()
    mmap_region.cleanup.assert_called_once_with()
    assert worker._mmap_region is None


def test_store_reuse_fence_enqueues_only_pending_store_events(monkeypatch) -> None:
    """Stream ordering must not synchronize or retire the store on the host."""
    worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    event = MagicMock()
    worker._store_handler = MagicMock()
    worker._store_handler._transfer_events = {1: event}
    worker._store_handler._async_submitter_handle = None
    stream = MagicMock()
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "current_stream", lambda: stream)

    worker.fence_store_reuse({1, 2})

    stream.wait_event.assert_called_once_with(event)
    event.synchronize.assert_not_called()
    assert worker._store_handler._transfer_events == {1: event}


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA stream fence test")
@torch.inference_mode()
@pytest.mark.parametrize("async_h2d_submission", [False, True])
@pytest.mark.parametrize("overwrite", ["forward", "load"])
def test_gpu_store_reuse_fence_orders_forward_and_independent_h2d(
    default_vllm_config, async_h2d_submission, overwrite
) -> None:
    """Overwriting a store's source must preserve the old CPU bytes."""
    page_size = 128 * 1024
    gpu_tensor = torch.full((2, page_size), 0x11, dtype=torch.int8, device=DEVICES[0])
    worker = CPUOffloadingWorker(
        kv_caches=CanonicalKVCaches(
            tensors=[
                CanonicalKVCacheTensor(tensor=gpu_tensor, page_size_bytes=page_size)
            ],
            group_data_refs=[
                [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_size)]
            ],
        ),
        blocks_per_chunk=1,
        num_cpu_blocks=2,
        async_h2d_submission=async_h2d_submission,
        async_h2d_chunk_bytes=8192,
    )
    worker.configure_experiment(
        {"copy_pipeline_depth": 2 if async_h2d_submission else 1}
    )
    cpu_tensor = worker._load_handler.src_tensors[0]
    cpu_tensor[1].fill_(0x5A)
    spec = GPULoadStoreSpec([0], group_sizes=(1,), block_indices=(0,))
    try:
        torch.accelerator.synchronize()
        assert worker.submit_store(1, spec, CPULoadStoreSpec([0]))
        transfer = worker._store_handler._transfers[-1]
        # Extend the store with a delayed DMA read, independently of whether
        # the driver's batched copy API blocks during the initial submission.
        with current_platform.stream(transfer.stream):
            torch.cuda._sleep(100_000_000)
            cpu_tensor[0].copy_(gpu_tensor[0], non_blocking=True)
            transfer.end_event.record(transfer.stream)
        worker.fence_store_reuse({1})
        assert not transfer.end_event.query(), "GPU fence unexpectedly waited on CPU"

        if overwrite == "forward":
            gpu_tensor[0].fill_(0x5A)
        else:
            assert worker.submit_load(2, CPULoadStoreSpec([1]), spec)

        worker.wait({1, 2})
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            cpu_tensor[0], torch.full((page_size,), 0x11, dtype=torch.int8)
        )
        torch.testing.assert_close(
            gpu_tensor[0].cpu(), torch.full((page_size,), 0x5A, dtype=torch.int8)
        )
        finished = worker.get_finished()
        assert {result.job_id for result in finished} == (
            {1} if overwrite == "forward" else {1, 2}
        )
        assert not worker.get_finished()
    finally:
        worker.shutdown()


def test_native_store_reuse_waits_for_completion_without_early_retirement():
    """An unrecorded native completion cannot be used as a CUDA stream fence."""
    worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    worker._store_handler = MagicMock()
    worker._store_handler._async_submitter_handle = 17
    worker.fence_store_reuse({3, 4})
    worker._store_handler.wait.assert_called_once_with({3, 4})
    worker._store_handler.get_finished.assert_not_called()


def test_native_wait_cannot_skip_a_job_during_registration(monkeypatch):
    """A concurrent waiter must not consult native jobs before insertion finishes."""
    handler = gpu_worker.SingleDirectionOffloadingHandler.__new__(
        gpu_worker.SingleDirectionOffloadingHandler
    )
    handler._async_submission = True
    handler._async_submitter_handle = 17
    handler._async_transfer_lock = threading.Lock()
    entered, registered, called = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def native_wait(handle, jobs):
        called.set()
        assert registered.is_set(), "Wait skipped an incompletely registered job"
        assert handle == 17 and jobs == [3]

    def wait_task():
        entered.set()
        handler.wait({3})

    monkeypatch.setattr(ops, "wait_async_batch_memcpy", native_wait)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        with handler._async_transfer_lock:
            future = executor.submit(wait_task)
            assert entered.wait(1)
            assert not called.wait(0.1)
            registered.set()
        future.result(timeout=1)
    assert called.is_set()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Native D2H requires CUDA")
@torch.inference_mode()
@pytest.mark.parametrize("overwrite", ["forward", "load"])
def test_native_store_orders_delayed_producer_and_source_reuse(
    default_vllm_config, overwrite
):
    """Capture the source producer before native submission and retain old bytes."""
    page_size = 128 * 1024
    tensor = torch.zeros((2, page_size), dtype=torch.int8, device=DEVICES[0])
    worker = CPUOffloadingWorker(
        kv_caches=CanonicalKVCaches(
            tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_size)],
            group_data_refs=[
                [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_size)]
            ],
        ),
        blocks_per_chunk=1,
        num_cpu_blocks=2,
        async_h2d_submission=True,
        async_h2d_chunk_bytes=8192,
        async_d2h_submission=True,
        async_d2h_chunk_bytes=8192,
    )
    worker.configure_experiment({"copy_pipeline_depth": 2})
    cpu_tensor = worker._load_handler.src_tensors[0]
    cpu_tensor[1].fill_(0x5A)
    spec = GPULoadStoreSpec([0], group_sizes=(1,), block_indices=(0,))
    try:
        torch.accelerator.synchronize()
        torch.cuda._sleep(500_000_000)
        tensor[0].fill_(0x11)
        producer = torch.Event()
        producer.record(current_platform.current_stream())
        assert worker.submit_store(1, spec, CPULoadStoreSpec([0]))
        assert not producer.query(), "Store submission waited for its producer"
        assert not worker.get_finished(), "Store retired before producer completion"
        worker.fence_store_reuse({1})
        assert producer.query()
        torch.testing.assert_close(
            cpu_tensor[0], torch.full((page_size,), 0x11, dtype=torch.int8)
        )
        if overwrite == "forward":
            tensor[0].fill_(0x5A)
        else:
            assert worker.submit_load(2, CPULoadStoreSpec([1]), spec)
        worker.wait({1, 2})
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            cpu_tensor[0], torch.full((page_size,), 0x11, dtype=torch.int8)
        )
        torch.testing.assert_close(
            tensor[0].cpu(), torch.full((page_size,), 0x5A, dtype=torch.int8)
        )
        assert {result.job_id for result in worker.get_finished()} == (
            {1} if overwrite == "forward" else {1, 2}
        )
        assert not worker.get_finished()
    finally:
        worker.shutdown()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Native D2H requires CUDA")
@torch.inference_mode()
def test_native_store_fifo_and_shutdown_preserve_overlapping_cpu_writes(
    default_vllm_config,
):
    """Stores to one CPU page remain FIFO; shutdown drains the final native job."""
    page_size = 64 * 1024
    tensor = torch.empty((2, page_size), dtype=torch.int8, device=DEVICES[0])
    tensor[0].fill_(0x11)
    tensor[1].fill_(0x33)
    worker = CPUOffloadingWorker(
        kv_caches=CanonicalKVCaches(
            tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_size)],
            group_data_refs=[
                [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_size)]
            ],
        ),
        blocks_per_chunk=1,
        num_cpu_blocks=2,
        async_d2h_submission=True,
        async_d2h_chunk_bytes=8192,
    )
    cpu = worker._store_handler.dst_tensors[0]
    try:
        for job in (1, 2):
            spec = GPULoadStoreSpec([job - 1], group_sizes=(1,), block_indices=(0,))
            assert worker.submit_store(job, spec, CPULoadStoreSpec([0]))
        worker.wait({1, 2})
        assert [result.job_id for result in worker.get_finished()] == [1, 2]
        assert not worker.get_finished()
        torch.testing.assert_close(
            cpu[0], torch.full((page_size,), 0x33, dtype=torch.int8)
        )
        torch.cuda._sleep(100_000_000)
        tensor[1].fill_(0x55)
        spec = GPULoadStoreSpec([1], group_sizes=(1,), block_indices=(0,))
        assert worker.submit_store(3, spec, CPULoadStoreSpec([1]))
    finally:
        worker.shutdown()
    torch.testing.assert_close(cpu[1], torch.full((page_size,), 0x55, dtype=torch.int8))
    assert worker._store_handler._async_submitter_handle is None


@pytest.mark.parametrize("failing_handler", ["store", "load"])
def test_worker_logs_handler_error_and_cleans_region(
    caplog_vllm, monkeypatch: pytest.MonkeyPatch, failing_handler
) -> None:
    worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    calls: list[str] = []
    store_handler = MagicMock()
    load_handler = MagicMock()
    if failing_handler == "store":

        def fail_store_shutdown() -> bool:
            calls.append("store")
            raise RuntimeError("transfer did not drain")

        store_handler.shutdown.side_effect = fail_store_shutdown
    else:

        def fail_load_shutdown() -> bool:
            calls.append("load")
            raise RuntimeError("transfer did not drain")

        load_handler.shutdown.side_effect = fail_load_shutdown

    def record_other_shutdown() -> bool:
        calls.append("load" if failing_handler == "store" else "store")
        return True

    if failing_handler == "store":
        load_handler.shutdown.side_effect = record_other_shutdown
    else:
        store_handler.shutdown.side_effect = record_other_shutdown

    def record_device_sync() -> None:
        calls.append("sync")

    monkeypatch.setattr(torch.accelerator, "synchronize", record_device_sync)

    def record_region_cleanup() -> None:
        calls.append("region")

    mmap_region = MagicMock()
    mmap_region.cleanup.side_effect = record_region_cleanup
    worker._store_handler = store_handler
    worker._load_handler = load_handler
    worker._mmap_region = mmap_region

    with caplog_vllm.at_level(
        logging.ERROR, logger="vllm.v1.kv_offload.cpu.gpu_worker"
    ):
        worker.shutdown()

    other = "load" if failing_handler == "store" else "store"
    getattr(worker, f"_{other}_handler").shutdown.assert_called_once_with()
    mmap_region.cleanup.assert_called_once_with()
    assert worker._mmap_region is None
    assert (
        f"Failed to shut down {failing_handler} offloading handler" in caplog_vllm.text
    )
    assert calls[-2:] == ["sync", "region"]


def test_handler_shutdown_skips_transfers_after_event_sync_failure() -> None:
    handler = gpu_worker.SingleDirectionOffloadingHandler.__new__(
        gpu_worker.SingleDirectionOffloadingHandler
    )
    handler._async_submitter_handle = None
    failed_event = MagicMock()
    failed_event.synchronize.side_effect = RuntimeError("device lost")
    skipped_event = MagicMock()
    handler._transfers = gpu_worker.deque(
        [
            MagicMock(end_event=failed_event),
            MagicMock(end_event=skipped_event),
        ]
    )
    handler._transfer_events = {1: failed_event, 2: skipped_event}
    handler._stream_pool = [MagicMock()]
    handler._event_pool = [MagicMock()]
    handler._buffer_pool = [(MagicMock(), MagicMock(), MagicMock())]
    handler.src_tensors = [MagicMock()]
    handler.dst_tensors = [MagicMock()]

    with pytest.raises(RuntimeError, match="device lost"):
        handler.shutdown()
    failed_event.synchronize.assert_called_once_with()
    skipped_event.synchronize.assert_not_called()
    assert not handler._transfers
    assert not handler._transfer_events
    assert not handler._stream_pool
    assert not handler._event_pool
    assert not handler._buffer_pool
    assert not handler.src_tensors
    assert not handler.dst_tensors


@pytest.mark.parametrize("device_sync_fails", [False, True])
def test_worker_syncs_before_cleanup_after_handler_failure(
    caplog_vllm, monkeypatch: pytest.MonkeyPatch, device_sync_fails: bool
) -> None:
    worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    calls: list[str] = []
    store_handler = MagicMock()

    def fail_store_shutdown() -> None:
        raise RuntimeError("device lost")

    store_handler.shutdown.side_effect = fail_store_shutdown
    load_handler = MagicMock()

    def record_device_sync() -> None:
        calls.append("sync")
        if device_sync_fails:
            raise RuntimeError("device lost")

    def record_region_cleanup() -> None:
        calls.append("region")

    monkeypatch.setattr(torch.accelerator, "synchronize", record_device_sync)
    mmap_region = MagicMock()
    mmap_region.cleanup.side_effect = record_region_cleanup
    worker._store_handler = store_handler
    worker._load_handler = load_handler
    worker._mmap_region = mmap_region

    with caplog_vllm.at_level(
        logging.WARNING, logger="vllm.v1.kv_offload.cpu.gpu_worker"
    ):
        worker.shutdown()

    assert calls == ["sync", "region"]
    mmap_region.cleanup.assert_called_once_with()
    if device_sync_fails:
        assert "Device sync before mmap cleanup failed" in caplog_vllm.text


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("gpu_page_size_bytes", GPU_PAGE_SIZES)
@pytest.mark.parametrize("blocks_per_chunk", BLOCKS_PER_CHUNK_VALUES)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_blocks", NUM_CPU_BLOCKS)
@pytest.mark.parametrize("num_tensors", NUM_TENSORS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("use_shared_memory", "replicated_layout"),
    [(False, False), (True, False), (True, True)],
)
@torch.inference_mode()
@pytest.mark.parametrize("async_h2d_submission", [False, True])
def test_transfer(
    default_vllm_config,
    async_h2d_submission,
    gpu_to_cpu: bool,
    num_mappings: int,
    gpu_page_size_bytes: int,
    blocks_per_chunk: int,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    num_tensors: int,
    seed: int,
    device: str,
    use_shared_memory: bool,
    replicated_layout: bool,
) -> None:
    set_random_seed(seed)

    # build CanonicalKVCacheTensor list: one per tensor
    kv_cache_tensors: list[CanonicalKVCacheTensor] = []
    for i in range(num_tensors):
        gpu_tensor = torch.zeros(
            (num_gpu_blocks, gpu_page_size_bytes),
            dtype=torch.int8,
            device=device,
        )
        kv_cache_tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        )

    # one group containing all tensors, one data ref per tensor
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [
            CanonicalKVCacheRef(
                tensor_idx=i,
                page_size_bytes=gpu_page_size_bytes,
            )
            for i in range(num_tensors)
        ]
    ]

    kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors,
        group_data_refs=kv_cache_groups_data_refs,
    )

    mmap_region: SharedOffloadRegion | None = None
    if use_shared_memory:
        cpu_page_size = round_up(
            gpu_page_size_bytes * num_tensors * blocks_per_chunk,
            SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT,
        )
        simulated_world_size = 2
        kv_bytes_per_block = (
            cpu_page_size if replicated_layout else cpu_page_size * simulated_world_size
        )
        mmap_region = SharedOffloadRegion(
            engine_id=str(uuid.uuid4()),
            num_blocks=num_cpu_blocks,
            rank=0,
            kv_bytes_per_block=kv_bytes_per_block,
            cpu_page_size=cpu_page_size,
        )

    worker = CPUOffloadingWorker(
        kv_caches=kv_caches,
        blocks_per_chunk=blocks_per_chunk,
        num_cpu_blocks=num_cpu_blocks,
        async_h2d_submission=async_h2d_submission,
        async_h2d_chunk_bytes=8192,
        mmap_region=mmap_region,
    )

    # select block mappings
    gpu_blocks = random.sample(range(num_gpu_blocks), num_mappings * blocks_per_chunk)
    cpu_blocks = random.sample(range(num_cpu_blocks), num_mappings)

    # expand cpu blocks to gpu-page granularity for uniform comparison:
    # each cpu block maps to blocks_per_chunk consecutive sub-blocks
    cpu_blocks_expanded = [
        cpu_block * blocks_per_chunk + j
        for cpu_block in cpu_blocks
        for j in range(blocks_per_chunk)
    ]

    # maybe skip some GPU blocks to test reading/writing from the middle of a CPU block
    blocks_to_skip = blocks_per_chunk - 1
    if blocks_to_skip > 0:
        gpu_blocks = gpu_blocks[blocks_to_skip:]
        cpu_blocks_expanded = cpu_blocks_expanded[blocks_to_skip:]

    # set transfer direction
    if gpu_to_cpu:
        handler = worker._store_handler
        src_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=(len(gpu_blocks),), block_indices=(blocks_to_skip,)
        )
        dst_spec = CPULoadStoreSpec(cpu_blocks)
        dst_to_src = dict(zip(cpu_blocks_expanded, gpu_blocks))
        num_dst_sub_blocks = num_gpu_blocks
    else:
        handler = worker._load_handler
        src_spec = CPULoadStoreSpec(cpu_blocks)
        dst_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=(len(gpu_blocks),), block_indices=(blocks_to_skip,)
        )
        dst_to_src = dict(zip(gpu_blocks, cpu_blocks_expanded))
        num_dst_sub_blocks = num_gpu_blocks

    # randomize src and dst tensors before transfer
    for tensor in handler.src_tensors:
        tensor.random_()
    for tensor in handler.dst_tensors:
        tensor.random_()

    # clone src and dst tensors before transfer
    orig_src_tensors = [x.clone() for x in handler.src_tensors]
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    # call transfer function via public API
    start_time = time.time()
    if gpu_to_cpu:
        assert worker.submit_store(1, src_spec, dst_spec)
    else:
        assert worker.submit_load(1, src_spec, dst_spec)
    pending = list(handler._transfers) + list(handler._async_transfers)
    assert {x.job_id for x in pending} == {1}

    # wait for transfer to complete
    end_time = time.time() + 10
    while time.time() < end_time:
        finished = worker.get_finished()
        if finished:
            assert finished[0].job_id == 1
            assert finished[0].success
            assert finished[0].transfer_size == (
                len(gpu_blocks)
                * sum([x.page_size_bytes for x in handler.layer_refs_per_group[0]])
            )
            assert finished[0].transfer_time > 0
            assert finished[0].transfer_time < (time.time() - start_time)
            break
        time.sleep(0.1)

    # verify src tensors did not change
    for orig_tensor, tensor in zip(orig_src_tensors, handler.src_tensors):
        assert torch.equal(orig_tensor, tensor)

    # verify dst tensors at gpu-page granularity.
    for src_tensor, dst_tensor, orig_dst_tensor in zip(
        handler.src_tensors,
        handler.dst_tensors,
        orig_dst_tensors,
    ):
        # view both GPU and CPU tensors as (n, gpu_page_size_bytes) for comparison.
        src_view = src_tensor.reshape(-1, gpu_page_size_bytes)
        dst_view = dst_tensor.reshape(-1, gpu_page_size_bytes)
        orig_dst_view = orig_dst_tensor.reshape(-1, gpu_page_size_bytes)
        for dst_sub_block in range(num_dst_sub_blocks):
            src_sub_block = dst_to_src.get(dst_sub_block)
            if src_sub_block is not None:
                expected = src_view[src_sub_block]
            else:
                expected = orig_dst_view[dst_sub_block]
            torch.testing.assert_close(dst_view[dst_sub_block].cpu(), expected.cpu())

    # Drop loop-variable refs so mmap_obj has no exported buffers at cleanup.
    del orig_tensor, tensor, src_tensor, dst_tensor, orig_dst_tensor
    del src_view, dst_view, orig_dst_view, expected

    worker.shutdown()


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings_per_group", NUM_MAPPINGS_PER_GROUP)
@pytest.mark.parametrize("gpu_page_size_bytes", GPU_PAGE_SIZES)
@pytest.mark.parametrize("blocks_per_chunk", BLOCKS_PER_CHUNK_VALUES)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_blocks", NUM_CPU_BLOCKS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
@pytest.mark.parametrize("async_h2d_submission", [False, True])
def test_transfer_multi_group(
    default_vllm_config,
    async_h2d_submission,
    gpu_to_cpu: bool,
    num_mappings_per_group: int,
    gpu_page_size_bytes: int,
    blocks_per_chunk: int,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    seed: int,
    device: str,
) -> None:
    """Test transfers with three KV cache groups:
    - Group 0: aligned transfer with num_mappings_per_group blocks
    - Group 1: zero blocks (empty group)
    - Group 2: unaligned CPU->GPU transfer (logical_offset=blocks_per_chunk-1,
      causing the implementation to skip source sub-blocks) with
      num_mappings_per_group blocks
    """
    set_random_seed(seed)

    # 3 groups, each with 2 tensors
    num_groups = 3
    tensors_per_group = 2
    num_tensors = num_groups * tensors_per_group
    kv_cache_tensors: list[CanonicalKVCacheTensor] = []
    for _ in range(num_tensors):
        gpu_tensor = torch.zeros(
            (num_gpu_blocks, gpu_page_size_bytes),
            dtype=torch.int8,
            device=device,
        )
        kv_cache_tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        )

    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [
            CanonicalKVCacheRef(
                tensor_idx=g * tensors_per_group + i,
                page_size_bytes=gpu_page_size_bytes,
            )
            for i in range(tensors_per_group)
        ]
        for g in range(num_groups)
    ]

    canonical_kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors, group_data_refs=kv_cache_groups_data_refs
    )

    worker = CPUOffloadingWorker(
        kv_caches=canonical_kv_caches,
        blocks_per_chunk=blocks_per_chunk,
        num_cpu_blocks=num_cpu_blocks,
        async_h2d_submission=async_h2d_submission,
        async_h2d_chunk_bytes=8192,
    )

    # group 0: aligned, group 1: empty, group 2: unaligned on CPU->GPU
    group_sizes_in_cpu_blocks = [num_mappings_per_group, 0, num_mappings_per_group]

    total_cpu_blocks = sum(group_sizes_in_cpu_blocks)
    total_gpu_blocks_needed = total_cpu_blocks * blocks_per_chunk
    gpu_blocks_all = random.sample(range(num_gpu_blocks), total_gpu_blocks_needed)
    cpu_blocks_all = random.sample(range(num_cpu_blocks), total_cpu_blocks)

    # split gpu/cpu blocks per group
    gpu_blocks_per_group: list[list[int]] = []
    cpu_blocks_per_group: list[list[int]] = []
    gpu_offset = 0
    cpu_offset = 0
    for size in group_sizes_in_cpu_blocks:
        gpu_count = size * blocks_per_chunk
        gpu_blocks_per_group.append(gpu_blocks_all[gpu_offset : gpu_offset + gpu_count])
        cpu_blocks_per_group.append(cpu_blocks_all[cpu_offset : cpu_offset + size])
        gpu_offset += gpu_count
        cpu_offset += size

    # expand cpu blocks to gpu-page granularity
    cpu_blocks_expanded_per_group = [
        [
            cpu_block * blocks_per_chunk + j
            for cpu_block in cpu_blocks
            for j in range(blocks_per_chunk)
        ]
        for cpu_blocks in cpu_blocks_per_group
    ]

    # skip sub-blocks from group 2 to test unaligned transfers.
    sub_blocks_to_skip = blocks_per_chunk - 1  # e.g. 2 when blocks_per_chunk=3
    if sub_blocks_to_skip > 0:
        gpu_blocks_per_group[2] = gpu_blocks_per_group[2][
            sub_blocks_to_skip:-sub_blocks_to_skip
        ]
        cpu_blocks_expanded_per_group[2] = cpu_blocks_expanded_per_group[2][
            sub_blocks_to_skip:-sub_blocks_to_skip
        ]

    # build flat gpu_blocks list and group_sizes in GPU blocks
    gpu_blocks: list[int] = []
    group_sizes: list[int] = []
    for gpu_blks in gpu_blocks_per_group:
        gpu_blocks.extend(gpu_blks)
        group_sizes.append(len(gpu_blks))

    # build flat cpu_blocks list
    cpu_blocks = []
    for cpu_blks in cpu_blocks_per_group:
        cpu_blocks.extend(cpu_blks)

    # block_indices: only relevant for unaligned transfers
    block_indices: list[int] = [0, 0, sub_blocks_to_skip]

    if gpu_to_cpu:
        handler = worker._store_handler
        src_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=group_sizes, block_indices=block_indices
        )
        dst_spec = CPULoadStoreSpec(cpu_blocks)
        # per-group mapping: cpu sub-block -> gpu sub-block
        dst_to_src_per_group = [
            dict(zip(expanded, gpu_blks))
            for expanded, gpu_blks in zip(
                cpu_blocks_expanded_per_group, gpu_blocks_per_group
            )
        ]
        num_dst_sub_blocks = num_cpu_blocks * blocks_per_chunk
    else:
        handler = worker._load_handler
        src_spec = CPULoadStoreSpec(cpu_blocks)
        dst_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=group_sizes, block_indices=block_indices
        )
        # per-group mapping: gpu sub-block -> cpu sub-block
        dst_to_src_per_group = [
            dict(zip(gpu_blks, expanded))
            for gpu_blks, expanded in zip(
                gpu_blocks_per_group, cpu_blocks_expanded_per_group
            )
        ]
        num_dst_sub_blocks = num_gpu_blocks

    # randomize src and dst tensors before transfer
    for tensor in handler.src_tensors:
        tensor.random_()
    for tensor in handler.dst_tensors:
        tensor.random_()

    orig_src_tensors = [x.clone() for x in handler.src_tensors]
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    if gpu_to_cpu:
        assert worker.submit_store(1, src_spec, dst_spec)
    else:
        assert worker.submit_load(1, src_spec, dst_spec)
    pending = list(handler._transfers) + list(handler._async_transfers)
    assert {x.job_id for x in pending} == {1}

    end_time = time.time() + 10
    while time.time() < end_time:
        finished = worker.get_finished()
        if finished:
            assert finished[0].job_id == 1
            assert finished[0].success
            expected_bytes = sum(
                group_size * sum([x.page_size_bytes for x in data_refs])
                for group_size, data_refs in zip(
                    group_sizes, handler.layer_refs_per_group
                )
            )
            assert finished[0].transfer_size == expected_bytes
            break
        time.sleep(0.1)

    # verify src tensors did not change
    for orig_tensor, tensor in zip(orig_src_tensors, handler.src_tensors):
        assert torch.equal(orig_tensor, tensor)

    # verify dst tensors at gpu-page granularity
    for group_idx, dst_to_src in enumerate(dst_to_src_per_group):
        group_tensor_offset = group_idx * tensors_per_group
        for tensor_idx in range(tensors_per_group):
            src_tensor = handler.src_tensors[group_tensor_offset + tensor_idx]
            dst_tensor = handler.dst_tensors[group_tensor_offset + tensor_idx]
            orig_dst_tensor = orig_dst_tensors[group_tensor_offset + tensor_idx]
            src_view = src_tensor.view(-1, gpu_page_size_bytes)
            dst_view = dst_tensor.view(-1, gpu_page_size_bytes)
            orig_dst_view = orig_dst_tensor.view(-1, gpu_page_size_bytes)
            for dst_sub_block in range(num_dst_sub_blocks):
                src_sub_block = dst_to_src.get(dst_sub_block)
                if src_sub_block is not None:
                    expected = src_view[src_sub_block]
                else:
                    expected = orig_dst_view[dst_sub_block]
                torch.testing.assert_close(
                    dst_view[dst_sub_block].cpu(), expected.cpu()
                )

    worker.shutdown()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="stream ordering test requires a CUDA-like platform",
)
@torch.inference_mode()
@pytest.mark.parametrize(
    "async_h2d_submission,pipeline_depth", [(False, 1), (True, 1), (True, 2), (True, 4)]
)
def test_load_waits_for_pending_compute_stream_writes(
    default_vllm_config, async_h2d_submission, pipeline_depth
) -> None:
    """A CPU load must land after pending writes to its GPU destination."""
    device = DEVICES[0]
    page_size_bytes = 128 * 1024
    num_blocks = 64
    loaded_blocks = list(range(32))
    sentinel = 0x5A

    gpu_tensor = torch.zeros(
        (num_blocks, page_size_bytes), dtype=torch.int8, device=device
    )
    loaded_block_ids = torch.tensor(loaded_blocks, dtype=torch.long, device=device)
    worker = CPUOffloadingWorker(
        kv_caches=CanonicalKVCaches(
            tensors=[
                CanonicalKVCacheTensor(
                    tensor=gpu_tensor, page_size_bytes=page_size_bytes
                )
            ],
            group_data_refs=[
                [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_size_bytes)]
            ],
        ),
        blocks_per_chunk=1,
        num_cpu_blocks=num_blocks,
        async_h2d_submission=async_h2d_submission,
        async_h2d_chunk_bytes=128 * 1024,
    )
    worker._load_handler.src_tensors[0].fill_(sentinel)
    worker.configure_experiment({"copy_pipeline_depth": pipeline_depth})
    expected = torch.full((page_size_bytes,), sentinel, dtype=torch.int8)

    try:
        for trial in range(3):
            gpu_tensor.fill_(0x11)
            torch.accelerator.synchronize()

            # Model a delayed zero of a freshly allocated KV block. Without a
            # compute-stream dependency, the DMA can finish during the sleep
            # and this later fill wipes out the loaded cache contents.
            torch.cuda._sleep(50_000_000)
            gpu_tensor.index_fill_(0, loaded_block_ids, 0)

            assert worker.submit_load(
                trial + 1,
                CPULoadStoreSpec(loaded_blocks),
                GPULoadStoreSpec(
                    loaded_blocks,
                    group_sizes=(len(loaded_blocks),),
                    block_indices=(0,),
                ),
            )
            deadline = time.time() + 10
            finished: list[TransferResult] = []
            while time.time() < deadline and not finished:
                finished = worker.get_finished()
                if not finished:
                    time.sleep(0.001)
            assert finished and finished[0].success, f"load {trial} did not finish"

            torch.accelerator.synchronize()
            for block_id in loaded_blocks:
                torch.testing.assert_close(gpu_tensor[block_id].cpu(), expected)
    finally:
        worker.shutdown()


@pytest.mark.parametrize("depth", [0, 9, -1, True])
def test_invalid_copy_pipeline_depth_rejected(depth):
    from vllm.v1.kv_offload.early_readiness import validate_experiment_options

    with pytest.raises(ValueError):
        validate_experiment_options({"copy_pipeline_depth": depth})


@pytest.mark.skipif(not current_platform.is_cuda(), reason="native CUDA H2D")
@pytest.mark.parametrize("dependency", ["independent", "reader", "zero", "store"])
@torch.inference_mode()
def test_block_readiness_overlaps_only_independent_destinations(
    default_vllm_config, dependency
):
    """Unrelated compute overlaps; destination readers/zeroing/stores stay ordered."""
    page = 1024 * 1024
    tensor = torch.full((2, page), 17, dtype=torch.int8, device=DEVICES[0])
    worker = CPUOffloadingWorker(
        kv_caches=CanonicalKVCaches(
            tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page)],
            group_data_refs=[[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page)]],
        ),
        blocks_per_chunk=1,
        num_cpu_blocks=2,
        async_h2d_submission=True,
        async_h2d_chunk_bytes=8192,
        async_d2h_submission=True,
    )
    worker.configure_block_readiness(True)
    cpu = worker._load_handler.src_tensors[0]
    cpu[1].fill_(90)
    spec = GPULoadStoreSpec([1], group_sizes=(1,), block_indices=(0,))
    end = torch.Event()
    try:
        torch.accelerator.synchronize()
        # Populate the native worker before measuring completion order.
        worker.submit_load(10, CPULoadStoreSpec([1]), spec)
        worker.wait({10})
        worker.get_finished()
        tensor[1].fill_(17)
        torch.accelerator.synchronize()
        torch.cuda._sleep(300_000_000)
        if dependency == "zero":
            tensor[1].zero_()
        if dependency == "store":
            worker.submit_store(11, spec, CPULoadStoreSpec([0]))
        else:
            worker._block_readiness.record(
                [0] if dependency == "independent" else [1], dependency
            )
        end.record()
        worker.submit_load(12, CPULoadStoreSpec([1]), spec)
        worker.wait({12})
        if dependency == "independent":
            assert not end.query()
        else:
            assert end.query()
        end.synchronize()
        assert torch.equal(tensor[1].cpu(), cpu[1])
        if dependency == "store":
            worker.wait({11})
            assert torch.all(cpu[0] == 17)
    finally:
        worker.shutdown()


@pytest.mark.parametrize("resumed", [False, True])
def test_block_readiness_tracks_all_cached_blocks_and_releases_finished(resumed):
    """KV readers include old cached blocks; resumed allocations replace old IDs."""
    from types import SimpleNamespace

    from vllm.v1.kv_offload.h2d_readiness import BlockReadiness

    tracker = BlockReadiness.__new__(BlockReadiness)
    tracker.requests = {"done": ([9],), "active": ([1, 2],)}
    output = SimpleNamespace(
        finished_req_ids={"done"},
        preempted_req_ids=None,
        scheduled_new_reqs=[SimpleNamespace(req_id="new", block_ids=([4, 5],))],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["active"],
            new_block_ids=[([3],)],
            resumed_req_ids={"active"} if resumed else set(),
        ),
        num_scheduled_tokens={"new": 128, "active": 1},
    )
    tracker.begin(output)
    assert "done" not in tracker.requests
    seen = []
    tracker.record = lambda blocks, kind: seen.append((set(blocks), kind))
    tracker.record_compute(output)
    expected = {3, 4, 5} if resumed else {1, 2, 3, 4, 5}
    assert seen == [(expected, "compute")]
    output.preempted_req_ids = {"active"}
    output.scheduled_new_reqs = []
    output.scheduled_cached_reqs.req_ids = []
    tracker.begin(output)
    assert set(tracker.requests) == {"new"}


def test_block_readiness_retires_store_ownership_without_a_subsequent_load():
    """Completed stores release physical-block ownership in GPU-only episodes."""
    worker = CPUOffloadingWorker.__new__(CPUOffloadingWorker)
    worker._block_readiness = MagicMock()
    worker._store_blocks = {1: {17}, 2: {18}}
    stored = MagicMock(job_id=1)
    loaded = MagicMock(job_id=3)
    worker._store_handler = MagicMock()
    worker._store_handler.get_finished.return_value = [stored]
    worker._load_handler = MagicMock()
    worker._load_handler.get_finished.return_value = [loaded]

    assert worker.get_finished() == [stored, loaded]
    assert worker._store_blocks == {2: {18}}
