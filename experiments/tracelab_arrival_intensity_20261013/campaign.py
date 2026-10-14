# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Choose arrival intensity on baseline, then freeze it for matched comparisons."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import regex as re

from experiments.tracelab_arrival_intensity_20261013.prepare import ROOT
from experiments.tracelab_request_slots_20261013 import campaign as replay

SOURCE = Path(__file__).resolve().parents[2]
OLD_ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_request_slots_20261013")
DEADLINE = json.loads((ROOT / "owner.json").read_text())["deadline"]


def save(path, data):
    replay.old.save(path, data)


def log(event, **fields):
    replay.old.log(event, **fields)


def timeout_handler(_signum, _frame):
    raise TimeoutError("Three-hour experiment allocation reached")


def setup():
    replay.ROOT = ROOT
    replay.SOURCE = SOURCE
    replay.BINARY = OLD_ROOT / "build/release/session_runner"
    budget = replay.setup()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max(1, int(DEADLINE - time.time())))
    save(
        ROOT / "provenance.json",
        dict(
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
            ).strip(),
            runner_binary=str(replay.BINARY),
            runner_sha256=hashlib.sha256(replay.BINARY.read_bytes()).hexdigest(),
            predictor_sha256=hashlib.sha256(replay.old.STATE.read_bytes()).hexdigest(),
            deadline=DEADLINE,
            cpu_cache_budget=budget["cpu_bytes_to_use"],
            gpu_cache_per_rank=96 * 1024**3,
            no_serving_model_or_predictor_changes=True,
        ),
    )


def launch(arm, label, root=None):
    if root is None:
        root = Path("/mnt/localnvme/jeongyoon/ai13" + label)
    (root / "server").mkdir(parents=True, exist_ok=False)
    save(ROOT / "active.json", dict(arm=arm, phase="server_startup", label=label))
    process = replay.old.server.launch(root, arm)
    save(root / "owner.json", dict(controller=os.getpid(), server_pid=process.pid))
    log("server_started", arm=arm, root=str(root), pid=process.pid)
    run = replay.old.server.old.base.collector
    run.BASE = replay.old.BASE
    try:
        startup_deadline = min(DEADLINE, time.time() + 1800)
        while time.time() < startup_deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Server exited during startup: {process.returncode}"
                )
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
                log("server_ready", arm=arm, root=str(root))
                return root, process
            except (OSError, TimeoutError):
                pass
            time.sleep(2)
        raise RuntimeError("Server startup timed out")
    except BaseException:
        replay.old.server.old.base.collector.stop_server(process)
        raise


def stop(root, process):
    replay.old.server.old.base.collector.stop_server(process)
    save(root / "stopped.json", dict(time=time.time(), returncode=process.poll()))
    log("server_stopped", root=str(root))


def cache_profile(directory, result):
    rows = [json.loads(line) for line in (directory / "rounds.jsonl").open()]
    events = [json.loads(line) for line in (directory / "transfer_events.jsonl").open()]
    names = {
        event["request"] for event in events if event["event"] == "job_observed_done"
    }
    loaded = set()
    for name in names:
        match = re.fullmatch(r"cmpl-(\d+_round_\d+)-0-[0-9a-f]{8}", name)
        if match:
            loaded.add(match[1])
    start = result["replay_start_timestamp"]
    begin = start + result["warmup_seconds"]
    end = begin + result["measurement_seconds"]
    windows = {}
    for name, lower, upper in (
        ("measurement", begin, end),
        ("first_half", begin, (begin + end) / 2),
        ("last_half", (begin + end) / 2, end),
    ):
        posted = [row for row in rows if lower <= row["post_timestamp"] < upper]
        windows[name] = dict(
            requests=len(posted),
            cpu_hit_requests=sum(row["request_id"] in loaded for row in posted),
            cpu_hit_request_percent=100
            * sum(row["request_id"] in loaded for row in posted)
            / len(posted)
            if posted
            else 0,
            cached_prompt_token_percent=100
            * sum(row["server_cached_prompt_tokens"] or 0 for row in posted)
            / sum(row["prompt_len"] for row in posted)
            if posted
            else 0,
            queue_mean_ms=sum(row["request_slot_wait_ms"] for row in posted)
            / len(posted)
            if posted
            else 0,
            first_calls=sum(row["round_idx"] == 0 for row in posted),
        )
    samples = []
    gpu_path = ROOT / "gpu_samples.jsonl"
    if gpu_path.exists():
        for line in gpu_path.open():
            sample = json.loads(line)
            if (
                sample.get("episode") == result["episode"]
                and begin <= sample["time"] < end
            ):
                samples.extend(gpu[1] for gpu in sample["gpus"])
    windows["gpu_utilization_mean_percent"] = (
        sum(samples) / len(samples) if samples else None
    )
    windows["request_source_counts_whole_episode"] = dict(
        cpu_hit_requests=len(loaded),
        cache_hit_requests=sum(
            (row["server_cached_prompt_tokens"] or 0) > 0 for row in rows
        ),
    )
    return windows


def cell(root, process, arm, rate, episode, warmup=600, measure=1200):
    episode_root = ROOT / "episodes" / episode
    episode_root.mkdir(parents=True, exist_ok=False)
    (episode_root / "workload.csv").symlink_to(ROOT / "workloads" / f"rate_{rate}.csv")
    replay.ROOT = episode_root
    replay.WARMUP, replay.MEASURE = warmup, measure
    save(
        ROOT / "active.json",
        dict(
            arm=arm,
            episode=episode,
            rate=rate,
            phase="replay",
            progress_path=str(episode_root / "progress.json"),
        ),
    )
    log(
        "episode_started",
        episode=episode,
        arm=arm,
        rate=rate,
        warmup=warmup,
        measure=measure,
    )
    result = replay.cell(root, process, arm, deadline_epoch=DEADLINE - 60)
    result.update(episode=episode, arrival_rate_sessions_per_second=float(rate))
    directory = episode_root / "cells" / arm
    result["cache_profile"] = cache_profile(directory, result)
    result["selection_basis"] = (
        "Baseline queue, GPU utilization, and CPU reuse; no prototype result used."
    )
    save(directory / "result.json", result)
    save(ROOT / "results" / f"{episode}.json", result)
    log(
        "episode_finished",
        episode=episode,
        throughput=result["throughput_rps"],
        cache=result["cache_profile"],
    )
    return result


def usable(result):
    profile = result["cache_profile"]
    growth = (
        profile["last_half"]["queue_mean_ms"] - profile["first_half"]["queue_mean_ms"]
    ) / 1000
    return (
        profile["last_half"]["cpu_hit_request_percent"] >= 10
        and result["slot_queue_ms"]["p95"] <= 90000
        and growth <= 30
        and (profile["gpu_utilization_mean_percent"] or 0) >= 85
    )


def main():
    setup()
    results = []
    root = process = None
    try:
        root, process = launch("baseline", "b")
        pilots = []
        for rate in ("0.1", "0.2"):
            pilots.append(
                cell(
                    root, process, "baseline", rate, f"pilot_baseline_r{rate}", 300, 600
                )
            )
            results.append(pilots[-1])
        viable = [result for result in pilots if usable(result)]
        if not viable and DEADLINE - time.time() > 5300:
            # Lower rate if queues grow; higher rate only if both pilots are idle.
            rate = (
                "0.25"
                if all(
                    (p["cache_profile"]["gpu_utilization_mean_percent"] or 0) < 85
                    and p["slot_queue_ms"]["p95"] < 30000
                    for p in pilots
                )
                else "0.05"
            )
            pilots.append(
                cell(
                    root, process, "baseline", rate, f"pilot_baseline_r{rate}", 300, 600
                )
            )
            results.append(pilots[-1])
            viable = [result for result in pilots if usable(result)]
        candidates = viable or [
            result
            for result in pilots
            if result["cache_profile"]["last_half"]["cpu_hit_request_percent"] >= 10
        ]
        if not candidates:
            raise RuntimeError(
                "No pilot retained CPU hits in the final measurement half"
            )
        selected = max(
            candidates,
            key=lambda p: (
                (p["cache_profile"]["gpu_utilization_mean_percent"] or 0)
                - min(30, p["slot_queue_ms"]["p95"] / 10000)
            ),
        )
        rate = str(selected["arrival_rate_sessions_per_second"])
        save(
            ROOT / "selection.json",
            dict(
                rate=rate,
                selected_pilot=selected["episode"],
                gates_satisfied=usable(selected),
                criterion=(
                    "Last-half CPU-hit requests >=10%, GPU mean >=85%, "
                    "queue p95 <=90s, last-half queue growth <=30s. "
                    "Rank by GPU mean minus queue penalty; baseline only."
                ),
                pilots=[p["episode"] for p in pilots],
            ),
        )
        log(
            "arrival_rate_frozen",
            rate=rate,
            pilot=selected["episode"],
            gates_satisfied=usable(selected),
        )
        results.append(
            cell(root, process, "baseline", rate, f"primary_baseline_r{rate}")
        )
        stop(root, process)
        root = process = None
        root, process = launch("full", "f")
        results.append(cell(root, process, "full", rate, f"primary_full_r{rate}"))
        if DEADLINE - time.time() > 3500:
            results.append(
                cell(root, process, "full", rate, f"reverse_full_r{rate}", 300, 900)
            )
            stop(root, process)
            root = process = None
            root, process = launch("baseline", "br")
            results.append(
                cell(
                    root,
                    process,
                    "baseline",
                    rate,
                    f"reverse_baseline_r{rate}",
                    300,
                    900,
                )
            )
        save(
            ROOT / "completed.json",
            dict(time=time.time(), rate=rate, results=[r["episode"] for r in results]),
        )
        log("completed", results=len(results), rate=rate)
    except BaseException as error:
        save(
            ROOT / "failure.json",
            dict(
                time=time.time(),
                error=repr(error),
                completed_episodes=[r["episode"] for r in results],
            ),
        )
        log("failed", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        if process is not None:
            stop(root, process)


if __name__ == "__main__":
    main()
