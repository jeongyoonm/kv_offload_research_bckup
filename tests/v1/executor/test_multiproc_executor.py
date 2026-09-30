# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.executor.multiproc_executor import WorkerProc


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output, diagnostic_rpc=None: None

    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[Any] = []
    worker_proc.handle_output = lambda output, diagnostic_rpc=None: outputs.append(
        output
    )

    worker_proc._execute_worker_rpc(("fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)
    assert str(outputs[0]) == "test error"


@pytest.mark.parametrize("enabled", [False, True])
def test_deferral_availability_precedes_dequeue_and_follows_response(
    monkeypatch, enabled
):
    from unittest.mock import Mock

    timing = SimpleNamespace(rpc_ready_ns=0)
    clock = Mock(side_effect=[100, 300])
    monkeypatch.setattr("vllm.v1.executor.multiproc_executor.time.monotonic_ns", clock)
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(
        worker=SimpleNamespace(_deferral_rpc_timing=enabled, _deferral_timing=timing),
        consume=lambda: "output",
    )
    responses: list[Any] = []
    worker_proc.handle_output = lambda output, diagnostic_rpc=None: responses.append(
        output
    )

    class Queue:
        calls = 0

        def dequeue(self, *, indefinite):
            self.calls += 1
            if self.calls == 1:
                assert timing.rpc_ready_ns == (100 if enabled else 0)
                return "consume", (), {}, None
            assert responses == ["output"]
            assert timing.rpc_ready_ns == (300 if enabled else 0)
            raise _ExitWorkerLoop

    worker_proc.rpc_broadcast_mq = Queue()
    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()
    assert clock.call_count == (2 if enabled else 0)


def test_cooperative_rpc_retains_partial_ranks_and_aggregates_once():
    """Earlier consumed ranks survive a miss; later RPCs never overtake them."""
    from collections import deque
    from unittest.mock import MagicMock

    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.rpc_broadcast_mq = MagicMock()
    executor.futures_queue = deque()
    first, second = MagicMock(), MagicMock()
    first.try_dequeue.side_effect = [(True, (WorkerProc.ResponseStatus.SUCCESS, "a"))]
    second.try_dequeue.side_effect = [
        (False, None),
        (True, (WorkerProc.ResponseStatus.SUCCESS, "b")),
    ]
    executor.response_mqs = [first, second]
    future = executor.collective_rpc("test", non_block=True)
    aggregate = MagicMock(side_effect=lambda x: tuple(x))
    future.aggregate = aggregate
    assert not future.poll()
    assert not future.done()
    assert future.poll()
    assert future.result() == ("a", "b")
    assert first.try_dequeue.call_count == 1
    aggregate.assert_called_once_with(["a", "b"])
    assert not executor.futures_queue


def test_cooperative_future_order_and_failure():
    from collections import deque
    from unittest.mock import MagicMock

    from vllm.v1.executor.multiproc_executor import FutureWrapper

    queue: deque[FutureWrapper] = deque()
    early_poll = MagicMock(side_effect=[(False, None), (True, "early")])
    late_poll = MagicMock(side_effect=RuntimeError("worker failed"))
    early = FutureWrapper(queue, lambda: None, poll_response=early_poll)
    late = FutureWrapper(queue, lambda: None, poll_response=late_poll)
    assert not late.poll()
    late_poll.assert_not_called()
    assert late.poll()
    assert early.result() == "early"
    with pytest.raises(RuntimeError, match="worker failed"):
        late.result()
    assert not queue


@pytest.mark.parametrize("failure", ["worker", "executor", "deadline"])
def test_cooperative_progress_propagates_failure(failure):
    from collections import deque
    from unittest.mock import MagicMock

    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.rpc_broadcast_mq = MagicMock()
    executor.futures_queue = deque()
    mq = MagicMock()
    mq.try_dequeue.return_value = (True, (WorkerProc.ResponseStatus.FAILURE, "failed"))
    executor.response_mqs = [mq]
    future = executor.collective_rpc(
        "test",
        non_block=True,
        unique_reply_rank=0,
        timeout=0 if failure == "deadline" else None,
    )
    executor.is_failed = failure == "executor"
    assert future.poll()
    with pytest.raises(TimeoutError if failure == "deadline" else RuntimeError):
        future.result()
    assert not executor.futures_queue


def test_cooperative_later_future_consumes_cancelled_predecessor_response():
    from collections import deque
    from concurrent.futures import CancelledError
    from unittest.mock import MagicMock

    from vllm.v1.executor.multiproc_executor import FutureWrapper

    queue: deque[FutureWrapper] = deque()
    cancelled_poll = MagicMock(return_value=(True, "discarded"))
    cancelled = FutureWrapper(queue, lambda: None, poll_response=cancelled_poll)
    later = FutureWrapper(queue, lambda: None, poll_response=lambda: (True, "later"))
    assert cancelled.cancel()
    assert later.poll()
    cancelled_poll.assert_called_once()
    with pytest.raises(CancelledError):
        cancelled.result()
    assert later.result() == "later"
    assert not queue
