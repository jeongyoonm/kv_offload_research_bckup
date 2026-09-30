# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``ops.swap_blocks_batch`` C++ (cuMemcpyBatchAsync) path."""

import contextlib
import gc
import time

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform


def _addrs(buffers: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([b.data_ptr() for b in buffers], dtype=torch.int64)


def _run_batch(sizes: list[int]) -> None:
    src = [torch.randint(256, (s,), dtype=torch.uint8, device="cuda") for s in sizes]
    dst = [torch.zeros_like(s) for s in src]
    ops.swap_blocks_batch(
        _addrs(src), _addrs(dst), torch.tensor(sizes, dtype=torch.int64)
    )
    torch.accelerator.synchronize()
    for s, d in zip(src, dst):
        assert torch.equal(d, s)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="swap_blocks_batch requires CUDA"
)
def test_swap_blocks_batch_default_stream():
    # cuMemcpyBatchAsync rejects the legacy default stream; the op must fall
    # back to per-copy transfers instead of raising.
    _run_batch([8, 4096, 8192])


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="swap_blocks_batch requires CUDA"
)
def test_swap_blocks_batch_dedicated_stream():
    # A dedicated non-default stream exercises the cuMemcpyBatchAsync fast path.
    with torch.cuda.stream(torch.cuda.Stream()):
        _run_batch([8, 4096, 8192])


def _pinned_i64(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, pin_memory=True)


def _create_submitter_or_skip(
    chunk_bytes: int = 8192, max_descriptors: int = 4096
) -> int:
    try:
        return ops.create_async_batch_memcpy_submitter(
            torch.accelerator.current_device_index(), chunk_bytes, max_descriptors
        )
    except RuntimeError as error:
        if "CUDA 12.8" in str(error) or "cuMemcpyBatchAsync" in str(error):
            pytest.skip(str(error))
        raise


@pytest.mark.skipif(current_platform.is_cuda(), reason="non-CUDA failure path only")
def test_async_batch_memcpy_submitter_requires_cuda():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        ops.create_async_batch_memcpy_submitter(0, 8192)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="async batch memcpy requires CUDA"
)
@pytest.mark.parametrize(
    "chunk_bytes,max_descriptors", [(8192, 4096), (1 << 62, 0), (1 << 62, 2)]
)
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_async_batch_memcpy_fifo_chunking_lifetime_poll_and_wait(
    chunk_bytes, max_descriptors, depth
):
    handle = _create_submitter_or_skip(chunk_bytes, max_descriptors)
    ops.configure_async_batch_memcpy_pipeline(handle, depth)
    sources = [
        torch.full((4096 * 4,), value, dtype=torch.uint8, pin_memory=True)
        for value in (17, 93)
    ]
    destinations = [torch.zeros_like(source, device="cuda") for source in sources]
    try:
        for job_id, (source, destination) in enumerate(
            zip(sources, destinations), start=10
        ):
            offsets = [index * 4096 for index in range(4)]
            src_ptrs = _pinned_i64([source.data_ptr() + offset for offset in offsets])
            dst_ptrs = _pinned_i64(
                [destination.data_ptr() + offset for offset in offsets]
            )
            sizes = _pinned_i64([4096] * 4)
            before = time.perf_counter()
            ops.submit_async_batch_memcpy(
                handle,
                job_id,
                src_ptrs,
                dst_ptrs,
                sizes,
                src_access_any=True,
            )
            assert time.perf_counter() - before < 1
            del src_ptrs, dst_ptrs, sizes
        gc.collect()

        first_poll = ops.poll_async_batch_memcpy(handle)
        ops.wait_async_batch_memcpy(handle, [10, 11])
        second_poll = ops.poll_async_batch_memcpy(handle)
        assert first_poll[0] + second_poll[0] == [10, 11]
        durations_ms = first_poll[1] + second_poll[1]
        assert len(durations_ms) == 2
        assert all(duration >= 0 for duration in durations_ms)
        for source, destination in zip(sources, destinations):
            assert torch.equal(source, destination.cpu())
    finally:
        ops.destroy_async_batch_memcpy_submitter(handle)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="async batch memcpy requires CUDA"
)
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_async_completion_ring_is_visible_before_ordinary_poll(tmp_path, depth):
    path = str(tmp_path / "completion-ring")
    session = 1234
    ring = ops.create_async_batch_memcpy_completion_ring(path, session)
    handle = ops.create_async_batch_memcpy_submitter(
        torch.accelerator.current_device_index(), 8192, 4096, path, session
    )
    ops.configure_async_batch_memcpy_pipeline(handle, depth)
    source = torch.full((4096,), 29, dtype=torch.uint8, pin_memory=True)
    destination = torch.zeros_like(source, device="cuda")
    try:
        ops.submit_async_batch_memcpy(
            handle,
            17,
            _pinned_i64([source.data_ptr()]),
            _pinned_i64([destination.data_ptr()]),
            _pinned_i64([source.numel()]),
            src_access_any=True,
        )
        ops.wait_async_batch_memcpy(handle, [17])
        sessions, jobs, timestamps, statuses, overflow, failure = (
            ops.poll_async_batch_memcpy_completion_ring(ring)
        )
        assert (sessions, jobs, statuses) == ([session], [17], [1])
        assert timestamps[0] > 0
        assert not overflow
        assert not failure
        # The ordinary completion remains independently available afterward.
        assert ops.poll_async_batch_memcpy(handle)[0] == [17]
    finally:
        ops.destroy_async_batch_memcpy_submitter(handle)
        ops.destroy_async_batch_memcpy_completion_ring(ring)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="async batch memcpy requires CUDA"
)
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_async_completion_ring_reports_worker_failure(tmp_path, depth):
    path = str(tmp_path / "failed-completion-ring")
    session = 4321
    ring = ops.create_async_batch_memcpy_completion_ring(path, session)
    handle = ops.create_async_batch_memcpy_submitter(
        torch.accelerator.current_device_index(), 8192, 4096, path, session
    )
    ops.configure_async_batch_memcpy_pipeline(handle, depth)
    destination = torch.zeros(16, dtype=torch.uint8, device="cuda")
    try:
        ops.submit_async_batch_memcpy(
            handle,
            23,
            _pinned_i64([0]),
            _pinned_i64([destination.data_ptr()]),
            _pinned_i64([destination.numel()]),
            src_access_any=True,
        )
        with pytest.raises(RuntimeError, match="async batch memcpy worker failed"):
            ops.wait_async_batch_memcpy(handle, [23])
        sessions, jobs, _, statuses, overflow, failure = (
            ops.poll_async_batch_memcpy_completion_ring(ring)
        )
        assert (sessions, jobs, statuses) == ([session], [23], [0])
        assert not overflow
        assert failure
    finally:
        with contextlib.suppress(RuntimeError):
            ops.destroy_async_batch_memcpy_submitter(handle)
        ops.destroy_async_batch_memcpy_completion_ring(ring)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="async batch memcpy requires CUDA"
)
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_async_batch_memcpy_rejects_duplicate_ids_and_drains_shutdown(depth):
    handle = _create_submitter_or_skip()
    ops.configure_async_batch_memcpy_pipeline(handle, depth)
    source = torch.full((4096,), 37, dtype=torch.uint8, pin_memory=True)
    destination = torch.zeros_like(source, device="cuda")
    src_ptrs = _pinned_i64([source.data_ptr()])
    dst_ptrs = _pinned_i64([destination.data_ptr()])
    sizes = _pinned_i64([source.numel()])
    ops.submit_async_batch_memcpy(
        handle, 1, src_ptrs, dst_ptrs, sizes, src_access_any=True
    )
    with pytest.raises(RuntimeError, match="duplicate.*job ID 1"):
        ops.submit_async_batch_memcpy(
            handle, 1, src_ptrs, dst_ptrs, sizes, src_access_any=True
        )

    ops.destroy_async_batch_memcpy_submitter(handle)
    assert torch.equal(source, destination.cpu())
    with pytest.raises(RuntimeError, match="invalid.*handle"):
        ops.submit_async_batch_memcpy(
            handle, 2, src_ptrs, dst_ptrs, sizes, src_access_any=True
        )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="async batch memcpy requires CUDA"
)
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_async_batch_memcpy_surfaces_background_errors(depth):
    handle = _create_submitter_or_skip()
    ops.configure_async_batch_memcpy_pipeline(handle, depth)
    destination = torch.zeros(16, dtype=torch.uint8, device="cuda")
    ops.submit_async_batch_memcpy(
        handle,
        7,
        _pinned_i64([0]),
        _pinned_i64([destination.data_ptr()]),
        _pinned_i64([destination.numel()]),
        src_access_any=True,
    )
    with pytest.raises(RuntimeError, match="async batch memcpy worker failed"):
        ops.wait_async_batch_memcpy(handle, [7])
    with pytest.raises(RuntimeError, match="async batch memcpy worker failed"):
        ops.destroy_async_batch_memcpy_submitter(handle)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="native CUDA test")
def test_pipeline_drains_submitted_chunk_before_later_descriptor_failure():
    handle = _create_submitter_or_skip(4096, 1)
    ops.configure_async_batch_memcpy_pipeline(handle, 2)
    source = torch.full((4096,), 44, dtype=torch.uint8, pin_memory=True)
    destination = torch.zeros_like(source, device="cuda")
    ops.submit_async_batch_memcpy(
        handle,
        1,
        _pinned_i64([source.data_ptr()] * 2),
        _pinned_i64([destination.data_ptr()] * 2),
        _pinned_i64([4096, -1]),
        src_access_any=True,
    )
    with pytest.raises(RuntimeError, match="nonnegative"):
        ops.wait_async_batch_memcpy(handle, [1])
    assert torch.equal(source, destination.cpu())
    with pytest.raises(RuntimeError, match="nonnegative"):
        ops.destroy_async_batch_memcpy_submitter(handle)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="native CUDA test")
def test_pipeline_setting_requires_ordinary_descriptor_reclamation():
    handle = _create_submitter_or_skip()
    source = torch.full((4096,), 44, dtype=torch.uint8, pin_memory=True)
    destination = torch.zeros_like(source, device="cuda")
    try:
        ops.submit_async_batch_memcpy(
            handle,
            1,
            _pinned_i64([source.data_ptr()]),
            _pinned_i64([destination.data_ptr()]),
            _pinned_i64([4096]),
            src_access_any=True,
        )
        ops.wait_async_batch_memcpy(handle, [1])
        with pytest.raises(RuntimeError, match="drained jobs"):
            ops.configure_async_batch_memcpy_pipeline(handle, 2)
        assert ops.poll_async_batch_memcpy(handle)[0] == [1]
        ops.configure_async_batch_memcpy_pipeline(handle, 2)
    finally:
        ops.destroy_async_batch_memcpy_submitter(handle)
