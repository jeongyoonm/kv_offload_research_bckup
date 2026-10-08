# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of launch ordering and the unarmed forwarding path."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.h2d_interference_20261009.worker import Calibration


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
