# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of launch ordering and the unarmed forwarding path."""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.h2d_interference_20261009.worker import Calibration


def test_chunk_comparison_preserves_predictor_weights_and_other_identity():
    from experiments.h2d_interference_20261009.serving import state_for_chunk

    original = dict(
        context=dict(h2d_chunk_bytes=134217728, tensor_parallel=8),
        latency_surface=dict(weights=[[1, 2]]),
        samples=[[1, 2]],
    )
    result = state_for_chunk(original, 8)
    assert result == dict(
        original, context=dict(h2d_chunk_bytes=8388608, tensor_parallel=8)
    )
    result["latency_surface"]["weights"][0][0] = 3
    assert original["latency_surface"]["weights"] == [[1, 2]]
    assert original["context"]["h2d_chunk_bytes"] == 134217728


@pytest.mark.parametrize("mode", ["concurrent", "after_enqueued"])
def test_post_enqueue_copy_fence_uses_independent_stream(monkeypatch, mode):
    calibration = Calibration.__new__(Calibration)
    calibration.cell = dict(mode=mode, profile="4k", jobs=2)
    calibration.descriptors = {"4k": dict(src=None, dst=None, sizes=None)}
    calibration.submission_stream = object()
    calibration.device = None
    calibration.current = {}
    calibration.handle = 1
    calibration.job = 0
    active, recorded = ["compute"], []

    @contextlib.contextmanager
    def stream_context(stream):
        assert stream is calibration.submission_stream
        active[0] = "independent"
        try:
            yield
        finally:
            active[0] = "compute"

    monkeypatch.setattr(
        "experiments.h2d_interference_20261009.worker.torch.cuda.stream",
        stream_context,
    )
    monkeypatch.setattr(
        "experiments.h2d_interference_20261009.worker.ops.submit_async_batch_memcpy",
        lambda *args: recorded.append(active[0]),
    )
    monkeypatch.setattr(
        "experiments.h2d_interference_20261009.worker.torch.cuda.current_stream",
        lambda _: SimpleNamespace(cuda_stream=0 if active[0] == "compute" else 1),
    )
    calibration.submit()
    expected = "independent" if mode == "after_enqueued" else "compute"
    assert recorded == [expected, expected]
    assert calibration.ids == [1, 2]
    assert active == ["compute"]


def test_all_cohort_requests_are_enqueued_before_resuming_compute():
    from experiments.h2d_interference_20261009.run import queued_generate

    events = []
    core = SimpleNamespace(call_utility=lambda *args: events.append(args))
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(engine_core=core),
        enqueue=lambda *args, **kwargs: events.append("enqueue"),
        wait_for_completion=lambda **kwargs: events.append("wait"),
    )
    queued_generate(llm, [], None)
    assert events == [
        ("pause_scheduler", "keep", False),
        "enqueue",
        ("resume_scheduler",),
        "wait",
    ]


def test_joint_finish_preserves_native_compute_completion_uncertainty():
    from experiments.h2d_interference_20261009.analyze import aggregate_cell

    cell = dict(
        id=1,
        mode="concurrent",
        profile="4k",
        jobs=4,
        shape="short",
        repeat=0,
        chunk_mib=128,
        depth=2,
    )
    ranks = [
        dict(
            rank=i,
            **cell,
            host_begin_ns=1_000_000,
            compute_submit_begin_ns=1_200_000,
            compute_enqueued_ns=2_000_000,
            execution=[
                1,
                1_000_001,
                1_000_002,
                1_000_003,
                8_000_000,
                1,
                6_000_000,
                7_000_000,
                7_500_000,
            ],
            io_begin_ns=1_100_000,
            io_enqueued_ns=1_150_000,
            bytes_per_job=195379200,
            completion_ns=[2_000_000, 3_000_000, 4_000_000, 5_000_000],
            validated_bytes=99 * 3 * 64,
        )
        for i in range(8)
    ]
    result = aggregate_cell(dict(cell=cell, ranks=ranks))
    assert result["joint_finish_ms"] == 6.5
    assert result["joint_finish_lower_ms"] == 6.0
    assert result["first_ready_ms"] == 1.0
    assert result["max_end_uncertainty_ms"] == 0.5
    ranks[0]["completion_ns"] = [2_000_000, 3_000_000, 4_000_000, 10_000_000]
    result = aggregate_cell(dict(cell=cell, ranks=ranks))
    assert result["joint_finish_ms"] == result["joint_finish_lower_ms"] == 9.0


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("compute_only", ["start", "forward", "finish"]),
        ("concurrent", ["start", "copy", "forward", "finish"]),
        ("after_enqueued", ["start", "forward", "finish", "copy"]),
    ],
)
def test_copy_launch_respects_requested_compute_boundary(monkeypatch, mode, expected):
    events = []
    calibration = Calibration.__new__(Calibration)
    calibration.cell = dict(
        mode=mode, id=1, chunk_mib=128, depth=2, profile="4k", shape="short"
    )
    calibration.current = None
    calibration.forward_count = 0
    calibration.device = None
    calibration.original_pre = lambda _: None
    calibration.observer = SimpleNamespace(
        start=lambda _: events.append("start"),
        finish=lambda *args, **kwargs: events.append("finish"),
    )
    calibration.submit = lambda: events.append("copy")
    calibration.runner = SimpleNamespace(
        execute_model_state=SimpleNamespace(
            input_batch=SimpleNamespace(
                num_scheduled_tokens=np.array([64]),
                num_computed_tokens_np=np.array([4096]),
                num_tokens_after_padding=64,
            )
        )
    )
    monkeypatch.setattr(
        "experiments.h2d_interference_20261009.worker.torch.cuda.current_stream",
        lambda _: SimpleNamespace(cuda_stream=0),
    )
    monkeypatch.setattr(
        "experiments.h2d_interference_20261009.worker.torch.cuda.nvtx.range_push",
        lambda _: None,
    )
    monkeypatch.setattr(
        "experiments.h2d_interference_20261009.worker.torch.cuda.nvtx.range_pop",
        lambda: None,
    )
    schedule = SimpleNamespace(num_scheduled_tokens={"request": 64})

    def original_execute():
        calibration.pre_forward(schedule)
        events.append("forward")

    calibration.original_execute = original_execute
    calibration.execute()
    assert events == expected
    assert calibration.current["context"] == [4096]
    with pytest.raises(AssertionError, match="multiple forwards"):
        calibration.execute()


def test_unarmed_forward_preserves_output_and_does_not_start_measurement():
    calibration = Calibration.__new__(Calibration)
    calibration.cell = None
    result = object()
    calibration.original_pre = lambda _: result
    calibration.original_execute = lambda: result
    assert calibration.pre_forward(None) is result
    assert calibration.execute() is result
