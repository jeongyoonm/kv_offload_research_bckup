# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure native agent timing with independent model-call admission slots."""

import hashlib
import json
import os
import signal
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import regex as re

from experiments.tracelab_runner_20261013 import campaign as old

ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_request_slots_20261013")
SOURCE = Path(__file__).resolve().parents[2]
RUNNER = Path("/mnt/localnvme/jeongyoon/tracelab_request_slots_src_20261013")
BINARY = ROOT / "build/release/session_runner"
WARMUP = 600
MEASURE = 1200
LIMIT = 32


def setup():
    old.ROOT = ROOT
    old.SOURCE = SOURCE
    old.UPSTREAM = RUNNER
    old.BINARY = BINARY
    old.BASE = "http://127.0.0.1:18484"
    budget = old.setup()
    old.server.PORT = 18484
    return budget


def launch(arm):
    root = Path("/mnt/localnvme/jeongyoon/rs13" + ("b" if arm == "baseline" else "f"))
    assert not root.exists(), root
    (root / "server").mkdir(parents=True)
    process = old.server.launch(root, arm)
    old.save(
        root / "owner.json",
        dict(controller=os.getpid(), server_pid=process.pid, arm=arm),
    )
    old.log("server_started", arm=arm, pid=process.pid, root=str(root))
    run = old.server.old.base.collector
    run.BASE = old.BASE
    try:
        end = time.monotonic() + 1800
        while time.monotonic() < end:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited: {process.returncode}")
            try:
                run.call("/health", timeout=1)
                registrations = re.findall(
                    r"CPU cache registration rank=(\d+) bytes=(\d+) "
                    r"seconds=[\d.]+ result=(\d+)",
                    (root / "server/server.log").read_text(),
                )
                assert {int(r) for r, _, c in registrations if int(c) == 0} == set(
                    range(8)
                )
                old.save(root / "registration_audit.json", registrations)
                old.log(
                    "server_ready",
                    arm=arm,
                    registered_bytes=sum(int(n) for _, n, _ in registrations),
                )
                return root, process
            except (OSError, TimeoutError):
                pass
            time.sleep(2)
        raise RuntimeError("Server startup timed out")
    except BaseException:
        run.stop_server(process)
        raise


def command(directory):
    return [
        str(BINARY),
        "--trace",
        str(ROOT / "workload.csv"),
        "--text-file",
        "/mnt/localnvme/jeongyoon/tracelab_runner_20261013/corpus/enwik9",
        "--tokenizer",
        str(old.MODEL),
        "--model",
        "glm53-compat",
        "--base-url",
        old.BASE + "/v1",
        "--max-model-len",
        "1048576",
        "--max-in-flight-requests",
        str(LIMIT),
        "--run-seconds",
        str(WARMUP + MEASURE),
        "--log-path",
        str(directory / "rounds.jsonl"),
        "--summary-path",
        str(directory / "runner_summary.json"),
    ]


def span_union(intervals):
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    start, end = intervals[0]
    for a, b in intervals[1:]:
        if a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    return total + end - start


def summarize(directory, start, events):
    rows = [
        json.loads(line)
        for line in (directory / "rounds.jsonl").read_text().splitlines()
    ]
    begin, end = start + WARMUP, start + WARMUP + MEASURE
    completed = [r for r in rows if begin <= r["complete_timestamp"] < end]
    posted = [r for r in rows if begin <= r["post_timestamp"] < end]
    phases = {}
    groups = {"warmup": (start, begin), "measurement": (begin, end)}
    for name, (a, b) in groups.items():
        phases[name] = sum(a <= r["complete_timestamp"] < b for r in rows)
    occupied = span_union(
        [
            (max(begin, r["post_timestamp"]), min(end, r["complete_timestamp"]))
            for r in rows
            if r["post_timestamp"] < end and r["complete_timestamp"] > begin
        ]
    )
    in_flight_seconds = sum(
        max(0, min(end, r["complete_timestamp"]) - max(begin, r["post_timestamp"]))
        for r in rows
    )
    jobs, stores = defaultdict(dict), defaultdict(dict)
    for e in events:
        target = jobs if e["event"] == "job_observed_done" else stores
        target[e["job_id"]][e["pid"]] = e
    load_requests = {
        e.get("request", "") for ranks in jobs.values() for e in ranks.values()
    }
    outcomes = Counter()
    for r in posted:
        pattern = re.compile(r"(?:^|-)" + re.escape(r["request_id"]) + r"(?:$|-)")
        loaded = any(pattern.search(name) for name in load_requests)
        cached = r["server_cached_prompt_tokens"] or 0
        outcomes["cpu_hit" if loaded else "gpu_hit" if cached else "miss"] += 1
    status = (directory / "runner.log").read_text()
    telemetry = re.findall(
        r"elapsed=([\d.]+)s \| slots_in_flight=(\d+) ready_queue=(\d+)", status
    )
    measured = [
        (float(t), int(n), int(q))
        for t, n, q in telemetry
        if WARMUP <= float(t) < WARMUP + MEASURE
    ]
    queue_ttft = [
        (r["post_timestamp"] - r["ready_timestamp"]) * 1000 + r["first_token_ms"]
        for r in posted
        if r["first_token_ms"] is not None
    ]
    return dict(
        rows=rows,
        jobs=jobs,
        stores=stores,
        result=dict(
            requests_submitted=len(rows),
            measured_completions=len(completed),
            latency_cohort_submissions=len(posted),
            warmup_seconds=WARMUP,
            measurement_seconds=MEASURE,
            replay_start_timestamp=start,
            throughput_rps=len(completed) / MEASURE,
            measured_output_tokens=sum(
                r["server_completion_tokens"] for r in completed
            ),
            ttft_ms=old.prior.distribution(
                [r["first_token_ms"] for r in posted if r["first_token_ms"] is not None]
            ),
            ready_ttft_ms=old.prior.distribution(queue_ttft),
            slot_queue_ms=old.prior.distribution(
                [r["request_slot_wait_ms"] for r in posted]
            ),
            cached_token_fraction=sum(
                r["server_cached_prompt_tokens"] or 0 for r in posted
            )
            / sum(r["prompt_len"] for r in posted),
            request_outcomes=dict(outcomes),
            measured_no_http_in_flight_seconds=MEASURE - occupied,
            mean_http_in_flight=in_flight_seconds / MEASURE,
            status_samples=len(measured),
            sampled_fraction_slots_full=sum(n == LIMIT for _, n, _ in measured)
            / len(measured),
            sampled_fraction_ready_queue_nonempty=sum(q > 0 for _, _, q in measured)
            / len(measured),
            max_slots_in_flight=max(n for _, n, _ in measured),
            max_ready_queue=max(q for _, _, q in measured),
            measurement_request_slots=LIMIT,
            active_session_cap=None,
            planned_sessions=8058,
            planned_calls=665453,
            phase_completions=phases,
            h2d_bytes_all_ranks_whole_episode=sum(
                e.get("transfer_bytes") or 0
                for ranks in jobs.values()
                for e in ranks.values()
            ),
            d2h_bytes_all_ranks_whole_episode=sum(
                e.get("transfer_bytes") or 0
                for ranks in stores.values()
                for e in ranks.values()
            ),
        ),
    )


def cell(root, server_process, arm, deadline_epoch=None):
    directory = ROOT / "cells" / arm
    directory.mkdir(parents=True, exist_ok=False)
    run, state, policy = old.prior.configure(root, arm)
    old.cache_client.drain()
    run.reset(True)
    old.prior.flush_timeline(root, completion_only=True)
    before = old.cache_client.metrics()
    sources_before = old.prior.token_sources(run)
    trace_before = set((root / "server/timing/late_admission").glob("*.json"))
    args = command(directory)
    old.save(
        directory / "spec.json",
        dict(
            arm=arm,
            command=args,
            policy=policy,
            cpu_cache_bytes=3028287488000,
            gpu_cache_bytes_per_rank=96 * 1024**3,
            request_slots=32,
            session_cap=None,
            warmup_seconds=WARMUP,
            measurement_seconds=MEASURE,
            tool_wait_scaling=False,
            predictor_sha256=hashlib.sha256(old.STATE.read_bytes()).hexdigest(),
        ),
    )
    old.log("cell_started", arm=arm, request_slots=32, session_cap=None)
    clock = None
    with (directory / "runner.log").open("w") as output:
        process = subprocess.Popen(
            args, cwd=RUNNER, stdout=output, stderr=output, start_new_session=True
        )
        old.save(
            directory / "owner.json", dict(pid=process.pid, controller=os.getpid())
        )
        try:
            remaining = (
                4 * 3600
                if deadline_epoch is None
                else max(0, deadline_epoch - time.time())
            )
            cutoff = time.monotonic() + remaining
            with (directory / "runner.log").open() as reader:
                progress = None
                while process.poll() is None:
                    if server_process.poll() is not None:
                        raise RuntimeError("Server exited during replay")
                    if time.monotonic() > cutoff:
                        raise RuntimeError("Replay watchdog expired")
                    for line in reader:
                        if line.startswith("replay clock |"):
                            clock = float(re.search(r"unix_seconds=([\d.]+)", line)[1])
                        if line.startswith("sessions "):
                            progress = line.strip()
                    phase = (
                        "token_pool_preparation"
                        if clock is None
                        else "warmup"
                        if time.time() < clock + WARMUP
                        else "measurement"
                        if time.time() < clock + WARMUP + MEASURE
                        else "drain"
                    )
                    old.save(
                        ROOT / "progress.json",
                        dict(
                            complete=False,
                            arm=arm,
                            phase=phase,
                            runner_pid=process.pid,
                            replay_start=clock,
                            progress=progress,
                            time=time.time(),
                        ),
                    )
                    time.sleep(5)
            if process.returncode:
                raise RuntimeError(f"Runner failed with {process.returncode}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    if clock is None:
        clock = float(
            re.search(
                r"replay clock \| unix_seconds=([\d.]+)",
                (directory / "runner.log").read_text(),
            )[1]
        )
    old.cache_client.drain()
    frozen = True
    if arm == "full":
        exported = run.call(
            "/experimental/kv-load-scheduling",
            dict(
                early_load_completion=True,
                experiment_options=policy,
                seal_estimator=dict(action="export"),
            ),
        )["seal_estimator"]["state"]
        frozen = exported == state
    events = old.prior.flush_timeline(root, completion_only=True)
    after = old.cache_client.metrics()
    sources_after = old.prior.token_sources(run)
    packed = summarize(directory, clock, events)
    rows = packed["rows"]
    result = packed["result"]
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
    checks = dict(
        all_success=all(r["status"] == "SUCCESS" for r in rows),
        unique_calls=len({r["request_id"] for r in rows}) == len(rows),
        per_request_input=all(
            r["prompt_len"] == r["server_prompt_tokens"] for r in rows
        ),
        per_request_output=all(
            r["output_len_target"] == r["server_completion_tokens"] for r in rows
        ),
        input_accounting=delta.get("vllm:request_prompt_tokens_sum")
        == sum(r["prompt_len"] for r in rows) + 1024,
        output_accounting=delta.get("vllm:generation_tokens_total")
        == sum(r["output_len_actual"] for r in rows) + 2,
        exactly_once=delta.get("vllm:request_success_total") == len(rows) + 2,
        all_rank_loads=all(len(ranks) == 8 for ranks in packed["jobs"].values()),
        all_rank_stores=all(len(ranks) == 8 for ranks in packed["stores"].values()),
        frozen_predictor=frozen,
        bounded_request_slots=result["max_slots_in_flight"] <= LIMIT,
        peak_request_slots=int(
            re.search(
                r"request slots \| limit=Some\(32\) peak_in_flight=(\d+)",
                (directory / "runner.log").read_text(),
            )[1]
        )
        <= LIMIT,
        full_measurement_window=bool(rows)
        and max(
            float(value)
            for value in re.findall(
                r"elapsed=([\d.]+)s", (directory / "runner.log").read_text()
            )
        )
        >= WARMUP + MEASURE - 1,
    )
    result.update(
        arm=arm,
        accepted=all(checks.values()),
        checks=checks,
        metric_deltas=delta,
        token_source_deltas={
            k: sources_after.get(k, 0) - sources_before.get(k, 0)
            for k in set(sources_before) | set(sources_after)
        },
    )
    protocol = Counter()
    for path in (
        set((root / "server/timing/late_admission").glob("*.json")) - trace_before
    ):
        item = json.loads(path.read_text())
        assert not item.get("overflow")
        if item.get("role") == "engine":
            protocol.update(item.get("counts", {}))
    result["protocol_counts"] = dict(protocol)
    with (directory / "transfer_events.jsonl").open("w") as output:
        for event in events:
            output.write(json.dumps(event) + "\n")
    old.save(directory / "result.json", result)
    old.log(
        "cell_finished",
        arm=arm,
        accepted=result["accepted"],
        rps=result["throughput_rps"],
        checks=checks,
    )
    if not result["accepted"]:
        raise RuntimeError(f"Cell rejected: {directory}")
    return result


def main():
    budget = setup()
    assert json.loads((ROOT / "cpu_checks/result.json").read_text())["passed"]
    old.save(
        ROOT / "owner.json", dict(pid=os.getpid(), time=time.time(), source=str(SOURCE))
    )
    old.save(
        ROOT / "plan.json",
        dict(
            arms=["baseline", "full"],
            request_slots=32,
            active_session_cap=None,
            arrival_policy="Unmodified exporter Poisson 1/s",
            tool_wait_scaling=False,
            warmup_seconds=WARMUP,
            measurement_seconds=MEASURE,
            cpu_bytes=budget["cpu_bytes_to_use"],
            gpu_cache_bytes_per_rank=96 * 1024**3,
            no_predictor_retraining=True,
            original_model_selector=True,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
            ).strip(),
            runner_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=RUNNER, text=True
            ).strip(),
        ),
    )
    results = []
    try:
        for arm in ("baseline", "full"):
            root, process = launch(arm)
            try:
                results.append(cell(root, process, arm))
            finally:
                old.server.old.base.collector.stop_server(process)
                old.save(
                    root / "stopped.json",
                    dict(time=time.time(), returncode=process.poll()),
                )
                old.log("server_stopped", arm=arm)
        b, f = results
        gains = dict(
            throughput_gain_pct=(f["throughput_rps"] / b["throughput_rps"] - 1) * 100,
            mean_ttft_reduction_pct=(1 - f["ttft_ms"]["mean"] / b["ttft_ms"]["mean"])
            * 100,
            ready_ttft_reduction_pct=(
                1 - f["ready_ttft_ms"]["mean"] / b["ready_ttft_ms"]["mean"]
            )
            * 100,
        )
        old.save(
            ROOT / "completed.json",
            dict(complete=True, results=results, gains=gains, time=time.time()),
        )
        old.save(
            ROOT / "progress.json", dict(complete=True, time=time.time(), gains=gains)
        )
    except BaseException as error:
        old.log("campaign_failed", error=repr(error))
        old.save(ROOT / "failure.json", dict(error=repr(error), time=time.time()))
        raise


if __name__ == "__main__":
    main()
