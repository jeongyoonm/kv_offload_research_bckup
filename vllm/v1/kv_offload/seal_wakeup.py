# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only progress bridge for an interruptible predicted initial wait."""

import mmap
import os
import struct
import threading
import time


class SealWakeup:
    """Observe transport changes; only the engine may interpret their contents."""

    def __init__(self, probe, interval=0.001):
        self.probe = probe
        self.interval = interval
        self.condition = threading.Condition()
        self.generation = 0
        self.active = False
        self.closed = False
        self.error = None
        self.scans = self.notifications = self.cpu_ns = 0
        self.previous = probe()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def ticket(self):
        with self.condition:
            return self.generation

    def pause(self):
        with self.condition:
            self.active = False

    def wait(self, ticket, seconds):
        with self.condition:
            self.active = True
            self.condition.notify_all()
            changed = self.condition.wait_for(
                lambda: self.generation != ticket or self.error or self.closed,
                timeout=seconds,
            )
            if self.error:
                raise RuntimeError("seal progress bridge failed") from self.error
            return "progress" if changed else "timer"

    def _run(self):
        started = time.thread_time_ns()
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.active or self.closed)
                    if self.closed:
                        return
                begin = time.thread_time_ns()
                value = self.probe()
                self.scans += 1
                with self.condition:
                    if value != self.previous or getattr(self, "repeat", False):
                        self.previous = value
                        self.generation += 1
                        self.notifications += 1
                        self.condition.notify_all()
                    self.cpu_ns += time.thread_time_ns() - begin
                    self.condition.wait(timeout=self.interval)
        except BaseException as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            self.cpu_ns = time.thread_time_ns() - started

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=1)
        if self.thread.is_alive():
            raise RuntimeError("seal progress bridge did not stop")
        if self.error:
            raise RuntimeError("seal progress bridge failed") from self.error


class TransportWakeup(SealWakeup):
    """Read advisory ring heads, completion words, ACKs, and abort notices."""

    def __init__(self, core, ack_fd, queued):
        self.rings = []
        try:
            for path in core.early_load_readiness.paths:
                fd = os.open(path, os.O_RDONLY)
                try:
                    self.rings.append(mmap.mmap(fd, 64, access=mmap.ACCESS_READ))
                finally:
                    os.close(fd)
            world = core.vllm_config.parallel_config.world_size
            execution = core.deferral_timing.execution_ranks
            if len(execution) != world or len(self.rings) != world:
                raise RuntimeError("interruptible wait requires all-rank transports")
            steps = tuple(
                output.readiness_timeline_step
                for _, output, _ in queued
                if output.total_num_scheduled_tokens > 0
            )
            from vllm.v1.kv_offload.step_timing import CAPACITY, RECORD

            def probe():
                heads = tuple(
                    (
                        struct.unpack_from("Q", ring, 24)[0],
                        struct.unpack_from("2I", ring, 40),
                    )
                    for ring in self.rings
                )
                # Advisory reads need no lock: torn reads can only cause a wake.
                # The engine revalidates every authoritative completion snapshot.
                ends = tuple(
                    (
                        struct.unpack_from(
                            "Q", rank.mem, step % CAPACITY * RECORD.size
                        )[0],
                        struct.unpack_from(
                            "Q", rank.mem, step % CAPACITY * RECORD.size + 64
                        )[0],
                    )
                    for step in steps
                    for rank in execution
                )
                self.repeat = bool(steps) and all(
                    ends[index * world + rank][0] == step
                    and ends[index * world + rank][1] > 0
                    for index, step in enumerate(steps)
                    for rank in range(world)
                )
                self.repeat |= any(any(flags) for _, flags in heads)
                with core.aborts_queue.mutex:
                    aborts = tuple(
                        tuple(ids) if not isinstance(ids, str) else ids
                        for ids in core.aborts_queue.queue
                    )
                return heads, ends, os.pread(ack_fd, 1024, 16), aborts

            super().__init__(probe)
        except BaseException:
            for ring in self.rings:
                ring.close()
            raise

    def close(self):
        try:
            super().close()
        finally:
            for ring in self.rings:
                ring.close()
            self.rings.clear()
