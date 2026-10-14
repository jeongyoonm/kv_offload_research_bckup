# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Matched serving cells around the unmodified upstream TraceLab runner."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import regex as re

from experiments.agentic_1m_20261012 import common, server
from experiments.load_only_overlap_20261011 import cache_client
from experiments.real_traces_20261012 import replay as prior

ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_runner_20261013")
SOURCE = Path(__file__).resolve().parents[2]
UPSTREAM = Path("/mnt/localnvme/jeongyoon/tracelab_replay_review_20261013")
BINARY = ROOT / "build/release/session_runner"
MODEL = Path(
    "/mnt/localnvme/huggingface/hub/models--Inferact--GLM-5.3-NVFP4/snapshots/ce67b36f3669192b5bb233819f0fda6c8a9837f8"
)
STATE = Path("/mnt/localnvme/jeongyoon/agentic_1m_20261012/frozen_predictor.json")
BASE = "http://127.0.0.1:18483"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def log(event, **fields):
    row = dict(time=time.time(), event=event, **fields)
    with (ROOT / "process_log.jsonl").open("a") as output:
        output.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def setup():
    budget = json.loads(
        Path(
            "/mnt/localnvme/jeongyoon/agentic_1m_20261012/cache_budget.json"
        ).read_text()
    )
    common.ROOT = ROOT
    server.SOURCE = SOURCE
    server.PORT = 18483
    server.BASE = BASE
    server.cache_settings = lambda: budget
    server.log = log
    prior.BASE = BASE
    prior.CAMPAIGN = ROOT
    prior.STATE_PATH = STATE
    prior.STATE_SHA256 = (
        "f159f32d226894457fb56b3195c47d4ff6e2a5d982d8d600b1970ea000294f96"
    )
    return budget


def launch(arm):
    root = Path("/mnt/localnvme/jeongyoon/tr13" + ("b" if arm == "baseline" else "f"))
    assert not root.exists(), root
    (root / "server").mkdir(parents=True)
    process = server.launch(root, arm)
    save(
        root / "owner.json",
        dict(controller=os.getpid(), server_pid=process.pid, arm=arm),
    )
    log("server_started", arm=arm, root=str(root), pid=process.pid)
    run = server.old.base.collector
    run.BASE = BASE
    try:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited: {process.returncode}")
            try:
                run.call("/health", timeout=1)
                registrations = re.findall(
                    r"CPU cache registration rank=(\d+) bytes=(\d+) "
                    r"seconds=[\d.]+ result=(\d+)",
                    (root / "server/server.log").read_text(),
                )
                assert {
                    int(rank) for rank, _, code in registrations if int(code) == 0
                } == set(range(8))
                save(root / "registration_audit.json", registrations)
                log(
                    "server_ready",
                    arm=arm,
                    registered_bytes=sum(int(size) for _, size, _ in registrations),
                )
                return root, process
            except (OSError, TimeoutError):
                pass
            time.sleep(2)
        raise RuntimeError("Startup timed out")
    except BaseException:
        run.stop_server(process)
        raise


def runner_command(csv_path, directory, concurrency, dry=False):
    command = [
        str(BINARY),
        "--trace",
        str(csv_path),
        "--text-file",
        str(ROOT / "corpus/enwik9"),
        "--tokenizer",
        str(MODEL),
        "--model",
        "glm53-compat",
        "--base-url",
        BASE + "/v1",
        "--max-model-len",
        "1048576",
        "--max-active-sessions",
        str(concurrency),
        "--summary-path",
        str(directory / "upstream_summary.json"),
        "--log-path",
        str(directory / "rounds.jsonl"),
    ]
    if dry:
        command.append("--dry-run")
    return command


def cell(root, server_process, arm, concurrency):
    directory = ROOT / "cells" / f"{arm}_c{concurrency}"
    directory.mkdir(parents=True, exist_ok=False)
    run, state, policy = prior.configure(root, arm)
    cache_client.drain()
    run.reset(True)
    prior.flush_timeline(root, completion_only=True)
    before = cache_client.metrics()
    sources_before = prior.token_sources(run)
    trace_before = set((root / "server/timing/late_admission").glob("*.json"))
    command = runner_command(ROOT / "workload.csv", directory, concurrency)
    save(
        directory / "spec.json",
        dict(
            arm=arm,
            max_active_sessions=concurrency,
            command=command,
            policy=policy,
            predictor_sha256=hashlib.sha256(STATE.read_bytes()).hexdigest(),
            runner_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(),
            dataset_sha256=hashlib.sha256(
                (ROOT / "workload.csv").read_bytes()
            ).hexdigest(),
            replay_source_changes=False,
            timing_scaling=False,
            concurrency_semantics_changes=False,
        ),
    )
    log("cell_started", arm=arm, concurrency=concurrency)
    with (directory / "runner.log").open("w") as output:
        process = subprocess.Popen(
            command, cwd=UPSTREAM, stdout=output, stderr=output, start_new_session=True
        )
        save(directory / "owner.json", dict(pid=process.pid, controller=os.getpid()))
        try:
            deadline = time.monotonic() + 24 * 3600
            while process.poll() is None:
                if server_process.poll() is not None:
                    raise RuntimeError("Server exited during replay")
                if time.monotonic() >= deadline:
                    raise RuntimeError("24-hour replay watchdog expired; cell rejected")
                text = (directory / "runner.log").read_text()
                progress = re.findall(
                    r"sessions (\d+)/(\d+) \| steps (\d+)/(\d+) "
                    r"completed=(\d+) submitted=(\d+) active=(\d+) "
                    r"failed=(\d+) \| elapsed=([\d.]+)s",
                    text,
                )
                save(
                    ROOT / "progress.json",
                    dict(
                        complete=False,
                        arm=arm,
                        concurrency=concurrency,
                        runner_pid=process.pid,
                        progress=progress[-1] if progress else None,
                        time=time.time(),
                    ),
                )
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(f"Runner failed: {process.returncode}; {directory}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    rows = [
        json.loads(line)
        for line in (directory / "rounds.jsonl").read_text().splitlines()
    ]
    cache_client.drain()
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
    events = prior.flush_timeline(root, completion_only=True)
    after = cache_client.metrics()
    sources_after = prior.token_sources(run)
    delta = {
        key: after.get(key, 0) - before.get(key, 0) for key in set(before) | set(after)
    }
    jobs, stores = defaultdict(dict), defaultdict(dict)
    for event in events:
        target = jobs if event["event"] == "job_observed_done" else stores
        target[event["job_id"]][event["pid"]] = event
    protocol_counts = defaultdict(int)
    for path in (
        set((root / "server/timing/late_admission").glob("*.json")) - trace_before
    ):
        item = json.loads(path.read_text())
        assert not item.get("overflow"), path
        if item.get("role") == "engine":
            for key, count in item.get("counts", {}).items():
                protocol_counts[key] += count
    input_tokens = sum(row["prompt_len"] for row in rows)
    output_tokens = sum(row["output_len_actual"] for row in rows)
    start = min(row["submit_timestamp"] for row in rows)
    end = max(row["complete_timestamp"] for row in rows)
    checks = dict(
        all_rounds=len(rows) == 1880,
        unique_rounds=len({(r["session_id"], r["round_idx"]) for r in rows}) == 1880,
        all_success=all(r["status"] == "SUCCESS" for r in rows),
        per_round_input=all(r["server_prompt_tokens"] == r["prompt_len"] for r in rows),
        per_round_output=all(
            r["server_completion_tokens"]
            == r["output_len_target"]
            == r["output_len_actual"]
            for r in rows
        ),
        input_accounting=delta.get("vllm:request_prompt_tokens_sum")
        == input_tokens + 1024,
        output_accounting=delta.get("vllm:generation_tokens_total")
        == output_tokens + 2,
        exactly_once=delta.get("vllm:request_success_total") == 1882,
        all_rank_loads=all(len(ranks) == 8 for ranks in jobs.values()),
        all_rank_stores=all(len(ranks) == 8 for ranks in stores.values()),
        frozen_predictor=frozen,
    )
    result = dict(
        arm=arm,
        concurrency=concurrency,
        accepted=all(checks.values()),
        checks=checks,
        requests=len(rows),
        seconds=end - start,
        throughput_rps=len(rows) / (end - start),
        throughput_window=(
            "First round submit to last round completion; includes native arrivals, "
            "session queueing and tool waits; excludes token-pool build "
            "and upstream preflight"
        ),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttft_ms=prior.distribution(
            [r["first_token_ms"] for r in rows if r["first_token_ms"] is not None]
        ),
        cached_token_fraction=sum(r["server_cached_prompt_tokens"] or 0 for r in rows)
        / input_tokens,
        token_source_deltas={
            k: sources_after.get(k, 0) - sources_before.get(k, 0)
            for k in set(sources_before) | set(sources_after)
        },
        metric_deltas=delta,
        load_jobs=len(jobs),
        store_jobs=len(stores),
        h2d_bytes_all_ranks=sum(
            e.get("transfer_bytes", 0) for rs in jobs.values() for e in rs.values()
        ),
        d2h_bytes_all_ranks=sum(
            e.get("transfer_bytes", 0) for rs in stores.values() for e in rs.values()
        ),
        protocol_counts=dict(protocol_counts),
        preflight_accounting=dict(requests=2, input_tokens=1024, output_tokens=2),
        upstream_summary=json.loads((directory / "upstream_summary.json").read_text()),
    )
    save(directory / "result.json", result)
    with (directory / "transfer_events.jsonl").open("w") as output:
        for event in events:
            output.write(json.dumps(event) + "\n")
    log(
        "cell_finished",
        arm=arm,
        concurrency=concurrency,
        accepted=result["accepted"],
        rps=result["throughput_rps"],
        ttft_ms=result["ttft_ms"],
        checks=checks,
    )
    if not result["accepted"]:
        raise RuntimeError(f"Cell rejected: {directory}")
    return result


def report(results):
    pairs = []
    for concurrency in (8, 16, 32):
        b = next(
            (
                r
                for r in results
                if r["arm"] == "baseline" and r["concurrency"] == concurrency
            ),
            None,
        )
        f = next(
            (
                r
                for r in results
                if r["arm"] == "full" and r["concurrency"] == concurrency
            ),
            None,
        )
        if b and f:
            pairs.append(
                dict(
                    concurrency=concurrency,
                    baseline_rps=b["throughput_rps"],
                    full_rps=f["throughput_rps"],
                    throughput_gain_pct=(f["throughput_rps"] / b["throughput_rps"] - 1)
                    * 100,
                    baseline_ttft_ms=b["ttft_ms"],
                    full_ttft_ms=f["ttft_ms"],
                    mean_ttft_reduction_pct=(
                        1 - f["ttft_ms"]["mean"] / b["ttft_ms"]["mean"]
                    )
                    * 100,
                )
            )
    save(ROOT / "comparisons.json", pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[32, 16, 8])
    parser.add_argument(
        "--arms", nargs="+", choices=["baseline", "full"], default=["baseline", "full"]
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    budget = setup()
    save(
        ROOT / "owner.json", dict(pid=os.getpid(), source=str(SOURCE), time=time.time())
    )
    save(
        ROOT / "plan.json",
        dict(
            cells=[
                dict(arm=a, concurrency=c)
                for a in ("baseline", "full")
                for c in args.concurrencies
            ],
            cpu_cache_budget=budget["cpu_bytes_to_use"],
            gpu_cache_bytes_per_rank=96 * 1024**3,
            runner_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
            ).strip(),
            serving_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
            ).strip(),
            runner_rust_source_unmodified=True,
            compatibility=json.loads((ROOT / "compatibility.json").read_text()),
            exporter_defaults_unmodified=True,
            sampling="Previous frozen 64-session windows",
            replicated=False,
        ),
    )
    results = []
    if args.resume:
        for path in sorted((ROOT / "cells").glob("*/result.json")):
            result = json.loads(path.read_text())
            if result["accepted"] and result["concurrency"] in args.concurrencies:
                results.append(result)
    try:
        for arm in args.arms:
            pending = [
                c
                for c in args.concurrencies
                if not any(r["arm"] == arm and r["concurrency"] == c for r in results)
            ]
            if not pending:
                continue
            root, process = launch(arm)
            try:
                for concurrency in pending:
                    results.append(cell(root, process, arm, concurrency))
                    report(results)
            finally:
                server.old.base.collector.stop_server(process)
                save(
                    root / "stopped.json",
                    dict(time=time.time(), returncode=process.poll()),
                )
                log("server_stopped", arm=arm)
        assert {(r["arm"], r["concurrency"]) for r in results} == {
            (a, c) for a in ("baseline", "full") for c in args.concurrencies
        }, "A matched baseline/full pair is required for each concurrency"
        save(
            ROOT / "completed.json",
            dict(complete=True, results=results, time=time.time()),
        )
        save(
            ROOT / "progress.json",
            dict(complete=True, cells=len(results), time=time.time()),
        )
    except BaseException as error:
        log("campaign_failed", error=repr(error))
        save(ROOT / "failure.json", dict(error=repr(error), time=time.time()))
        raise


if __name__ == "__main__":
    main()
