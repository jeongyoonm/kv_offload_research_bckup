# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of launch ordering and the unarmed forwarding path."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.h2d_interference_20261009.worker import Calibration


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
    calibration.cell = dict(mode=mode, id=1)
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
        lambda _: None,
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
