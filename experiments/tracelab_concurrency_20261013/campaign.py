# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched 8/16-call cells using the frozen arrival-retimed TraceLab catalogue."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from experiments.tracelab_arrival_intensity_20261013 import campaign as base

ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_concurrency_20261013")
PREVIOUS = Path("/mnt/localnvme/jeongyoon/tracelab_arrival_intensity_20261013")
RUNNER = Path("/mnt/localnvme/jeongyoon/tracelab_arrival_intensity_src_20261013")
BINARY = PREVIOUS / "build_keepalive/release/session_runner"
ORDER = [(8, "baseline"), (8, "full"), (16, "full"), (16, "baseline")]


def timeout_handler(_signum, _frame):
    raise TimeoutError("Five-hour concurrency campaign watchdog expired")


def report():
    results = [
        json.loads(path.read_text())
        for path in sorted((ROOT / "results").glob("*.json"))
    ]
    for arm in ("baseline", "full"):
        result = json.loads((PREVIOUS / f"results/repair_{arm}_r0.2.json").read_text())
        result["historical"] = True
        results.append(result)
    pairs = []
    for limit in (8, 16, 32):
        pair = {
            result["arm"]: result
            for result in results
            if result["measurement_request_slots"] == limit
        }
        if set(pair) != {"baseline", "full"}:
            continue
        b, f = pair["baseline"], pair["full"]
        pairs.append(
            dict(
                request_slots=limit,
                historical=limit == 32,
                baseline=b,
                full=f,
                throughput_gain_pct=(f["throughput_rps"] / b["throughput_rps"] - 1)
                * 100,
                mean_ttft_reduction_pct=(
                    1 - f["ttft_ms"]["mean"] / b["ttft_ms"]["mean"]
                )
                * 100,
                p95_ttft_reduction_pct=(1 - f["ttft_ms"]["p95"] / b["ttft_ms"]["p95"])
                * 100,
            )
        )
    base.save(ROOT / "comparisons.json", pairs)
    status = (
        "complete"
        if (ROOT / "completed.json").exists()
        else "failed"
        if (ROOT / "failure.json").exists()
        else "running"
    )
    lines = [
        "TraceLab request-call concurrency comparison",
        f"Status: {status}",
        "",
        (
            "Calls | arm | req/s | HTTP mean TTFT ms | HTTP p95 TTFT ms | "
            "CPU-load requests % | last-half CPU-load % | cached tokens % | "
            "queue mean s | GPU utilization % | mean in-flight calls | accepted"
        ),
    ]
    for result in sorted(
        results, key=lambda r: (r["measurement_request_slots"], r["arm"])
    ):
        profile = result["cache_profile"]
        gpu = profile["gpu_utilization_mean_percent"]
        gpu_text = "N/A" if gpu is None else f"{gpu:.2f}"
        lines.append(
            f"{result['measurement_request_slots']} | {result['arm']} | "
            f"{result['throughput_rps']:.5f} | {result['ttft_ms']['mean']:.2f} | "
            f"{result['ttft_ms']['p95']:.2f} | "
            f"{profile['measurement']['cpu_hit_request_percent']:.2f} | "
            f"{profile['last_half']['cpu_hit_request_percent']:.2f} | "
            f"{profile['measurement']['cached_prompt_token_percent']:.2f} | "
            f"{result['slot_queue_ms']['mean'] / 1000:.2f} | {gpu_text} | "
            f"{result['mean_http_in_flight']:.2f} | {result['accepted']}"
        )
    lines += [
        "",
        "Calls | throughput gain % | mean TTFT reduction % | p95 TTFT reduction %",
    ]
    for pair in pairs:
        lines.append(
            f"{pair['request_slots']} | {pair['throughput_gain_pct']:+.2f} | "
            f"{pair['mean_ttft_reduction_pct']:+.2f} | "
            f"{pair['p95_ttft_reduction_pct']:+.2f}"
        )
    lines += [
        "",
        "Fresh 8/16 cells; 32-call results are the earlier repaired-client pair.",
        "Same full seed0 catalogue, 0.2 sessions/s, prefixes, appended inputs,",
        "recorded output lengths, and tool waits. Only the request-call limit changes.",
        "No active-session cap. Tool waits occupy no model-call slot.",
        "Each cell: 600s warmup + 900s measurement, fresh server and cache reset.",
        "Same 1s HTTP idle timeout client without retries, frozen predictor, model,",
        "original selector, worker/scheduler policies, maximum validated CPU cache,",
        "and 96GiB/rank GPU KV. No priming, pinning, or eviction-policy changes.",
        "Throughput counts fixed-window completions. Latency/cache use POST cohort",
        "in that window, including drain responses. All calls drain and audit.",
        "CPU-load request means any observed load, including preemption reloads.",
        "One pair per concurrency, no confidence intervals. Lower limits can raise",
        "queueing and evictions. Closed-loop progress changes the measured mix.",
        "Growing queues do not establish indefinite open-loop steady-state stability.",
    ]
    (ROOT / "REPORT.txt").write_text("\n".join(lines) + "\n")


def main():
    assert not (ROOT / "owner.json").exists(), "Use a fresh artifact root"
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "workloads").mkdir()
    workload = PREVIOUS / "workloads/rate_0.2.csv"
    (ROOT / "workloads/rate_0.2.csv").symlink_to(workload)
    base.ROOT = ROOT
    base.DEADLINE = time.time() + 5 * 3600
    replay = base.replay
    replay.ROOT, replay.SOURCE = ROOT, base.SOURCE
    replay.BINARY, replay.RUNNER = BINARY, RUNNER
    budget = replay.setup()
    base.save(
        ROOT / "owner.json",
        dict(pid=os.getpid(), start=time.time(), deadline=base.DEADLINE),
    )
    assert hashlib.sha256(BINARY.read_bytes()).hexdigest() == (
        "a1d83917d7cf74dabccdc14ba4d1396c8a034ef9fc7e3ac60ef66b41f1b73636"
    )
    assert json.loads((PREVIOUS / "keepalive_checks/result.json").read_text())["passed"]
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip(), "GPUs must be idle before starting this campaign"
    base.save(
        ROOT / "plan.json",
        dict(
            order=ORDER,
            arrival_rate_sessions_per_second=0.2,
            warmup_seconds=600,
            measurement_seconds=900,
            active_session_cap=None,
            http_idle_timeout_seconds=1,
            retries=False,
            workload=str(workload),
            workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
            runner_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(),
            predictor_sha256=hashlib.sha256(replay.old.STATE.read_bytes()).hexdigest(),
            cpu_cache_budget=budget["cpu_bytes_to_use"],
            gpu_cache_bytes_per_rank=96 * 1024**3,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=base.SOURCE, text=True
            ).strip(),
            serving_changes=False,
            fresh_server_and_cache_each_cell=True,
            no_sampling_or_content_changes=True,
            rationale="Change only in-flight model-call limit; alternate pair order.",
        ),
    )
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5 * 3600)
    monitor = subprocess.Popen(
        [
            str(base.SOURCE / ".venv/bin/python"),
            "-m",
            "experiments.tracelab_concurrency_20261013.monitor",
        ],
        cwd=base.SOURCE,
    )
    root = process = None
    results = []
    try:
        report()
        for limit, arm in ORDER:
            replay.LIMIT = limit
            episode = f"{arm}_c{limit}"
            target = ROOT / "servers" / episode
            target.mkdir(parents=True)
            alias = Path(f"/mnt/localnvme/jeongyoon/tc13{arm[0]}{limit}")
            assert len(str(alias / "tmp")) <= 45
            alias.symlink_to(target, target_is_directory=True)
            base.save(
                target / "path_alias.json",
                dict(alias=str(alias), target=str(target), reason="Unix socket length"),
            )
            root, process = base.launch(arm, episode, alias)
            results.append(base.cell(root, process, arm, "0.2", episode, 600, 900))
            base.stop(root, process)
            root = process = None
            report()
        base.save(
            ROOT / "completed.json",
            dict(time=time.time(), results=[r["episode"] for r in results]),
        )
        base.save(ROOT / "progress.json", dict(complete=True, time=time.time()))
        base.log("concurrency_comparison_complete", cells=len(results))
    except BaseException as error:
        base.save(ROOT / "failure.json", dict(time=time.time(), error=repr(error)))
        base.log("concurrency_comparison_failed", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        if process is not None:
            base.stop(root, process)
        monitor.wait(timeout=20)
        report()


if __name__ == "__main__":
    main()
