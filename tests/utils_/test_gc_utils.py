# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

from vllm.utils.gc_utils import (
    GCDebugConfig,
    _compute_detailed_type,
    _compute_top_gc_collected_objects,
)


@dataclass
class Normal:
    v: int


@dataclass
class ListWrapper:
    vs: list[int]

    def __len__(self) -> int:
        return len(self.vs)


def test_compute_detailed_type():
    assert (
        _compute_detailed_type(Normal(v=8))
        == "<class 'tests.utils_.test_gc_utils.Normal'>"
    )

    assert _compute_detailed_type([1, 2, 3]) == "<class 'list'>(size:3)"
    assert _compute_detailed_type({4, 5}) == "<class 'set'>(size:2)"
    assert _compute_detailed_type({6: 7}) == "<class 'dict'>(size:1)"
    assert (
        _compute_detailed_type(ListWrapper(vs=[]))
        == "<class 'tests.utils_.test_gc_utils.ListWrapper'>(size:0)"
    )


def test_compute_top_gc_collected_objects():
    objects: list[Any] = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        {13, 14},
        {15: 16, 17: 18},
        Normal(v=19),
        Normal(v=20),
        Normal(v=21),
    ]
    assert _compute_top_gc_collected_objects(objects, top=-1) == ""
    assert _compute_top_gc_collected_objects(objects, top=0) == ""
    assert (
        _compute_top_gc_collected_objects(objects, top=1)
        == "    4:<class 'list'>(size:3)"
    )
    assert _compute_top_gc_collected_objects(objects, top=2) == "\n".join(
        [
            "    4:<class 'list'>(size:3)",
            "    3:<class 'tests.utils_.test_gc_utils.Normal'>",
        ]
    )
    assert _compute_top_gc_collected_objects(objects, top=3) == "\n".join(
        [
            "    4:<class 'list'>(size:3)",
            "    3:<class 'tests.utils_.test_gc_utils.Normal'>",
            "    1:<class 'set'>(size:2)",
        ]
    )


def test_gc_debug_config():
    assert not GCDebugConfig(None).enabled
    assert not GCDebugConfig("").enabled
    assert not GCDebugConfig("0").enabled

    config = GCDebugConfig("1")
    assert config.enabled
    assert config.top_objects == -1

    config = GCDebugConfig('{"top_objects":5}')
    assert config.enabled
    assert config.top_objects == 5


def test_timing_only_gc_probe_avoids_heap_walk_and_records_comparable_clock(
    monkeypatch,
):
    """Diagnosing pauses must not scan the heap before the collection being timed."""
    from unittest.mock import Mock

    from vllm.utils import gc_utils

    clock = iter([0, 1_000_000_000, 1_250_000_000])
    monkeypatch.setattr(gc_utils.time, "monotonic_ns", lambda: next(clock))
    walk = Mock(side_effect=AssertionError("timing-only probe walked the heap"))
    monkeypatch.setattr(gc_utils.gc, "get_objects", walk)
    log = Mock()
    monkeypatch.setattr(gc_utils.logger, "info", log)
    debugger = gc_utils.GCDebugger(
        GCDebugConfig('{"timing_only":true,"min_duration_ms":10}')
    )
    debugger.handle("start", {"generation": 2})
    debugger.handle("stop", {"generation": 2, "collected": 3})
    walk.assert_not_called()
    assert log.call_count == 1
    assert log.call_args.args[1:5] == (1_000_000_000, 1_250_000_000, 250, 2)


def test_timing_only_gc_probe_suppresses_short_collections(monkeypatch):
    from unittest.mock import Mock

    from vllm.utils import gc_utils

    clock = iter([0, 1_000_000_000, 1_000_050_000])
    monkeypatch.setattr(gc_utils.time, "monotonic_ns", lambda: next(clock))
    log = Mock()
    monkeypatch.setattr(gc_utils.logger, "info", log)
    debugger = gc_utils.GCDebugger(
        GCDebugConfig('{"timing_only":true,"min_duration_ms":10}')
    )
    debugger.handle("start", {"generation": 0})
    debugger.handle("stop", {"generation": 0, "collected": 3})
    log.assert_not_called()
