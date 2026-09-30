# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Passive timing-read diagnostics. Never delays or invokes the scheduler."""

import multiprocessing
import os
import queue
import time

from vllm.v1.kv_offload.execution_timing import start_snapshot_state, valid_snapshot
from vllm.v1.kv_offload.step_timing import RankFile, primary_record_reason


def sample(step, dispatched, primary, execution):
    def read_many(files):
        records: list[tuple[int, ...] | None] = []
        statuses: list[str] = []
        for rank in files:
            try:
                record, status = rank.read_with_status(step)
            except (OSError, ValueError):
                record, status = None, "read_error"
            records.append(record)
            statuses.append(status)
        return records, statuses

    records, reads = read_many(primary)
    starts, start_reads = read_many(execution)
    now = time.monotonic_ns()
    reasons = [
        status if status != "ok" else primary_record_reason(r, dispatched, now)
        for r, status in zip(records, reads)
    ]
    start_reasons = []
    for r, status in zip(starts, start_reads):
        if status != "ok":
            reason = status
        elif start_snapshot_state([r], step, now, 1) == "invalid":
            reason = (
                "start_uncertainty"
                if r and r[3] > r[2] + 2_000_000
                else "start_inconsistent_or_failed"
            )
        else:
            reason = "ready" if valid_snapshot([r], step, now) else "pending"
        start_reasons.append(reason)
    valid = bool(records) and all(r == "ok" for r in reasons)
    return dict(
        time_ns=now,
        primary=records,
        execution=starts,
        primary_reasons=reasons,
        execution_reasons=start_reasons,
        valid=valid,
        complete=valid and all(r[5] for r in records),
        start_state=start_snapshot_state(starts, step, now, len(primary)),
    )


def _collect(paths, commands, output, failures, ready, stopped, sample_limit):
    primary, execution = [], []
    batch: list[dict] = []
    batch_started = 0

    def send_batch():
        nonlocal batch, batch_started
        if batch:
            exported = time.monotonic_ns()
            for row in batch:
                row["export_ns"] = exported
            output.put_nowait(batch)
            batch = []
            batch_started = 0

    try:
        for path in paths:
            primary.append(RankFile(path))
            execution.append(RankFile(path + ".execution"))
        jobs = {}
        barriers = []
        samples = 0
        ready.set()
        while not stopped.wait(0.0005):
            while True:
                try:
                    opportunity, job = commands.get_nowait()
                except queue.Empty:
                    break
                if opportunity is None:
                    barriers.append(job)
                else:
                    job["activation_ns"] = time.monotonic_ns()
                    jobs[opportunity] = job
            for opportunity, job in list(jobs.items()):
                row = sample(job["step"], job["dispatch"], primary, execution)
                if not batch:
                    batch_started = time.monotonic_ns()
                batch.append(
                    dict(
                        event="timing_audit_sample",
                        opportunity_id=opportunity,
                        enqueue_ns=job["enqueue_ns"],
                        activation_ns=job["activation_ns"],
                        **row,
                    )
                )
                if len(batch) == 32:
                    send_batch()
                samples += 1
                if row["complete"] or row["time_ns"] >= job["end"]:
                    del jobs[opportunity]
                if samples >= sample_limit:
                    raise RuntimeError("shadow timing audit sample overflow")
            if batch and time.monotonic_ns() - batch_started >= 5_000_000:
                send_batch()
            if barriers and not jobs:
                send_batch()
                for token in barriers:
                    output.put_nowait(
                        [
                            dict(
                                event="timing_audit_flushed",
                                time_ns=time.monotonic_ns(),
                                token=token,
                            )
                        ]
                    )
                barriers.clear()
        send_batch()
    except Exception as error:
        failures.put(
            dict(
                event="timing_audit_failure",
                time_ns=time.monotonic_ns(),
                error=repr(error),
            )
        )
        stopped.set()
        ready.set()
    finally:
        for rank in primary + execution:
            rank.close()


class ShadowTimingAudit:
    """Sample in a spawned process with an independent Python interpreter/GIL."""

    def __init__(self, paths):
        self.horizon_ms = int(os.environ.get("VLLM_DEFERRAL_AUDIT_HORIZON_MS", "120"))
        sample_limit = int(os.environ.get("VLLM_DEFERRAL_AUDIT_SAMPLE_LIMIT", "100000"))
        if not 1 <= self.horizon_ms <= 1000 or not 1 <= sample_limit <= 1_000_000:
            raise ValueError("invalid bounded shadow audit settings")
        ctx = multiprocessing.get_context("spawn")
        self.commands = ctx.Queue(maxsize=4096)
        self.output = ctx.Queue(maxsize=4096)
        self.failures = ctx.Queue(maxsize=1)
        self.stopped = ctx.Event()
        ready = ctx.Event()
        self.events = []
        self.flush_token = 0
        self.process = ctx.Process(
            target=_collect,
            args=(
                paths,
                self.commands,
                self.output,
                self.failures,
                ready,
                self.stopped,
                sample_limit,
            ),
            daemon=True,
        )
        self.process.start()
        if not ready.wait(30) or self.stopped.is_set():
            self.close()
            raise RuntimeError("shadow timing audit failed to initialize")

    def begin(self, opportunity, now, data, snapshot):
        if self.stopped.is_set() or not self.process.is_alive():
            self.events.append(
                dict(
                    event="timing_audit_unavailable",
                    time_ns=now,
                    opportunity_id=opportunity,
                )
            )
            return
        try:
            self.commands.put_nowait(
                (
                    opportunity,
                    dict(
                        enqueue_ns=time.monotonic_ns(),
                        step=data["step"],
                        dispatch=data["dispatch_ns"],
                        end=now + self.horizon_ms * 1_000_000,
                    ),
                )
            )
        except queue.Full:
            self.stopped.set()
            self.events.append(
                dict(
                    event="timing_audit_failure",
                    time_ns=now,
                    error="audit command overflow",
                )
            )
            return
        self.events.append(
            dict(
                event="timing_audit_begin",
                time_ns=now,
                opportunity_id=opportunity,
                candidates=dict(snapshot),
                audit_horizon_ms=self.horizon_ms,
                data=dict(data),
            )
        )

    def drain(self):
        rows, self.events = self.events, []
        if not self.process.is_alive() and not self.stopped.is_set():
            self.stopped.set()
            rows.append(
                dict(
                    event="timing_audit_failure",
                    time_ns=time.monotonic_ns(),
                    error=f"collector exited: {self.process.exitcode}",
                )
            )
        for channel in (self.output, self.failures):
            while True:
                try:
                    packet = channel.get_nowait()
                    rows.extend(packet if isinstance(packet, list) else [packet])
                except queue.Empty:
                    break
        return rows

    def flush(self):
        """At a cell boundary, acknowledge all previously submitted audit jobs."""
        rows = self.drain()
        if self.stopped.is_set():
            return rows
        self.flush_token += 1
        token = self.flush_token
        try:
            self.commands.put_nowait((None, token))
        except queue.Full:
            self.stopped.set()
            rows.append(
                dict(
                    event="timing_audit_failure",
                    time_ns=time.monotonic_ns(),
                    error="audit flush command overflow",
                )
            )
            return rows
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            latest = self.drain()
            rows.extend(latest)
            if (
                any(
                    r["event"] == "timing_audit_flushed" and r["token"] == token
                    for r in latest
                )
                or self.stopped.is_set()
            ):
                return rows
            time.sleep(0.001)
        self.stopped.set()
        rows.append(
            dict(
                event="timing_audit_failure",
                time_ns=time.monotonic_ns(),
                error="audit flush acknowledgement timeout",
            )
        )
        return rows

    def close(self):
        self.stopped.set()
        deadline = time.monotonic() + 5
        rows = []
        while self.process.is_alive() and time.monotonic() < deadline:
            rows.extend(self.drain())
            self.process.join(timeout=0.01)
        rows.extend(self.drain())
        self.events.extend(rows)
        alive = self.process.is_alive()
        if alive:
            self.process.terminate()
            self.process.join(timeout=5)
        for channel in (self.commands, self.output, self.failures):
            channel.cancel_join_thread()
            channel.close()
        if alive:
            raise RuntimeError("shadow timing audit did not stop")
