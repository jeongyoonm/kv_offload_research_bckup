# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-driven, bounded seal notifications; readiness remains advisory."""

import fcntl
import mmap
import os
import queue
import select
import struct
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path

MAGIC = 0x564C4C4D534E3031
SLOTS = 16
JOBS = 256
COMPUTES = 16
HEADER_WORDS = 10
SLOT_OFFSET = HEADER_WORDS
JOB_OFFSET = SLOT_OFFSET + SLOTS * 4
COMPUTE_OFFSET = JOB_OFFSET + JOBS * 2
SIZE = (COMPUTE_OFFSET + COMPUTES * 3) * 8
PATH_KEY = "_coalesced_seal_notification_path"
WORD = struct.Struct("<Q")


class SealNotifications:
    """Serialize producers with flock; block the engine on a FIFO, never scan."""

    def __init__(self, path, session=None, world=None):
        self.path = str(path)
        self.owner = session is not None
        self.lock = threading.RLock()
        self.closed = False
        self.cpu_ns = self.scans = self.notifications = 0
        self.links = []
        self.fd = os.open(
            self.path, os.O_RDWR | (os.O_CREAT | os.O_EXCL if self.owner else 0), 0o600
        )
        try:
            if self.owner:
                if not 0 < session < 2**63 or not 1 <= world <= 63:
                    raise ValueError("invalid seal notification identity")
                os.ftruncate(self.fd, SIZE)
                os.mkfifo(self.path + ".fifo", 0o600)
            if os.fstat(self.fd).st_size != SIZE:
                raise ValueError("invalid seal notification size")
            self.mem = mmap.mmap(self.fd, SIZE)
            self.pipe = os.open(self.path + ".fifo", os.O_RDWR | os.O_NONBLOCK)
            if self.owner:
                self.mem[:] = bytes(SIZE)
                self._put(0, MAGIC)
                self._put(1, session)
                self._put(2, world)
            elif self._get(0) != MAGIC:
                raise ValueError("invalid seal notification magic")
        except BaseException:
            os.close(self.fd)
            if hasattr(self, "mem"):
                self.mem.close()
            if hasattr(self, "pipe"):
                os.close(self.pipe)
            if self.owner:
                for name in (self.path, self.path + ".fifo"):
                    if os.path.exists(name):
                        os.unlink(name)
            raise

    def _get(self, word):
        return WORD.unpack_from(self.mem, word * 8)[0]

    def _put(self, word, value):
        WORD.pack_into(self.mem, word * 8, value)

    @contextmanager
    def _locked(self):
        with self.lock:
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self.fd, fcntl.LOCK_UN)

    def descriptor(self, transport_path, rank, generation=0, track=True):
        path = str(transport_path) + ".seal"
        Path(path).write_text(f"{self.path}\n{rank}\n{self._get(1)}\n{generation}\n")
        if track:
            self.links.append(path)
        return path

    def arm(self, step, generation, candidates, computes):
        """Install job identities before publication; no accounting changes."""
        if not 0 < step < 2**64 or not 0 < generation < 2**64:
            raise ValueError("invalid seal window identity")
        jobs = [job for candidate in candidates for job in candidate]
        if len(candidates) > SLOTS or len(jobs) > JOBS or len(computes) > COMPUTES:
            raise ValueError("seal notification capacity exceeded")
        if any(type(job) is not int or not 0 <= job < 2**63 for job in jobs):
            raise ValueError("invalid seal notification job")
        if any(not 0 < compute < 2**64 for compute in computes):
            raise ValueError("invalid seal notification compute step")
        with self._locked():
            if self._get(3):
                raise RuntimeError("seal notification window already open")
            self.notifications = 0
            self.mem[SLOT_OFFSET * 8 :] = bytes(SIZE - SLOT_OFFSET * 8)
            self._put(4, generation)
            self._put(7, 0)
            self._put(8, len(candidates))
            self._put(9, len(computes))
            index = 0
            for slot, candidate in enumerate(candidates):
                offset = SLOT_OFFSET + slot * 4
                self._put(offset + 1, index)
                self._put(offset + 2, len(candidate))
                for job in candidate:
                    self._put(JOB_OFFSET + index * 2, job)
                    index += 1
            for index, compute in enumerate(computes):
                self._put(COMPUTE_OFFSET + index * 3, compute)
            self._put(3, step)

    def disarm(self):
        with self._locked():
            self._put(3, 0)
            self._notify()

    def _notify(self):
        self._put(5, self._get(5) + 1)
        # A full FIFO is already readable; sequence retains progress.
        with suppress(BlockingIOError):
            os.write(self.pipe, b"\1")

    def notify(self):
        """Wake after cancellation has been added to the authoritative queue."""
        with self.lock:
            if self.closed:
                return
            with self._locked():
                if self._get(3):
                    self._notify()

    def _check_candidates(self):
        mask = (1 << self._get(2)) - 1
        for slot in range(self._get(8)):
            offset = SLOT_OFFSET + slot * 4
            first, count = self._get(offset + 1), self._get(offset + 2)
            if (
                not self._get(offset + 3)
                and self._get(offset) == mask
                and all(
                    self._get(JOB_OFFSET + job * 2 + 1) == mask
                    for job in range(first, first + count)
                )
            ):
                self._put(offset + 3, 1)
                self._notify()

    def prepared(self, step, rank, status, slot, generation=None):
        with self._locked():
            if step != self._get(3) or (
                generation is not None and generation != self._get(4)
            ):
                return
            if not 0 <= rank < self._get(2) or not -1 <= slot < self._get(8):
                raise ValueError("invalid seal preparation identity")
            if status == 2:
                return  # Preserve the ordinary non-ready preparation outcome.
            if status != 1:
                self._put(6, 1)
                self._notify()
                return
            word = 7 if slot == -1 else SLOT_OFFSET + slot * 4
            self._put(word, self._get(word) | (1 << rank))
            self._check_candidates()

    def seed(self, readiness, ready_slots=()):
        """Include completions drained before arming; producers may race safely."""
        with self._locked():
            if not self._get(3):
                return
            mask = (1 << self._get(2)) - 1
            ready_slots = set(ready_slots)
            for slot in range(self._get(8)):
                offset = SLOT_OFFSET + slot * 4
                first, count = self._get(offset + 1), self._get(offset + 2)
                for index in range(first, first + count):
                    word = JOB_OFFSET + index * 2
                    job = self._get(word)
                    bits = (
                        mask
                        if slot in ready_slots or job in readiness._released_jobs
                        else sum(
                            1 << rank for rank in readiness._rank_reports.get(job, ())
                        )
                    )
                    self._put(word + 1, self._get(word + 1) | bits)
            self._check_candidates()

    def seed_compute(self, step, records):
        for rank, record in enumerate(records):
            if record and record[0] == step:
                if record[5] == 0:
                    self.compute(step, rank, False)
                elif record[8]:
                    self.compute(step, rank)

    def compute(self, step, rank, success=True):
        with self._locked():
            if not self._get(3):
                return
            for index in range(self._get(9)):
                offset = COMPUTE_OFFSET + index * 3
                if self._get(offset) != step:
                    continue
                if not 0 <= rank < self._get(2):
                    raise ValueError("invalid compute rank")
                if not success:
                    self._put(6, 1)
                    self._notify()
                    return
                bits = self._get(offset + 1) | (1 << rank)
                self._put(offset + 1, bits)
                if bits == (1 << self._get(2)) - 1 and not self._get(offset + 2):
                    self._put(offset + 2, 1)
                    self._notify()
                return

    def _drain(self):
        while True:
            try:
                if not os.read(self.pipe, 4096):
                    break
            except BlockingIOError:
                break

    def ticket(self):
        # Drain and snapshot under the same producer lock: a later completion
        # necessarily leaves either a new sequence or a readable FIFO.
        with self._locked():
            self._drain()
            return self._get(5)

    def wait(self, ticket, seconds):
        deadline = time.monotonic() + seconds
        while True:
            with self._locked():
                if self._get(6):
                    raise RuntimeError("seal notification producer failed")
                terminal_compute = self._get(9) > 0 and all(
                    self._get(COMPUTE_OFFSET + index * 3 + 2)
                    for index in range(self._get(9))
                )
                terminal_candidates = self._get(8) > 0 and all(
                    self._get(SLOT_OFFSET + slot * 4 + 3)
                    for slot in range(self._get(8))
                )
                if self._get(5) != ticket or terminal_compute or terminal_candidates:
                    self.notifications += 1
                    return "progress"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timer"
            select.select([self.pipe], [], [], remaining)
            with self._locked():
                self._drain()

    def pause(self):
        pass

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.mem.close()
            os.close(self.fd)
            os.close(self.pipe)
            if self.owner:
                for path in self.links + [self.path + ".fifo", self.path]:
                    with suppress(FileNotFoundError):
                        os.unlink(path)


def signal_preparation(control, step, rank, status, slot):
    try:
        lines = Path(control + ".seal").read_text().splitlines()
        path = lines[0]
    except FileNotFoundError:
        return
    channel = SealNotifications(path)
    try:
        if channel._get(1) != int(lines[2]):
            raise RuntimeError("stale preparation notification session")
        channel.prepared(step, rank, status, slot, int(lines[3]))
    finally:
        channel.close()


class NotificationQueue(queue.Queue):
    """Publish cancellation before signaling the sleeping engine."""

    def __init__(self, channel):
        super().__init__()
        self.channel = channel

    def put(self, item, block=True, timeout=None):
        super().put(item, block, timeout)
        if self.channel is not None:
            self.channel.notify()
