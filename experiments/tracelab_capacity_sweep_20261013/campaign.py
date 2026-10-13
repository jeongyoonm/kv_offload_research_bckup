# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive the ten-hour matched TraceLab concurrency and arrival-intensity study."""

import hashlib
import json
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path

from experiments.tracelab_arrival_intensity_20261013 import campaign as base
from experiments.tracelab_capacity_sweep_20261013 import analyze
from experiments.tracelab_concurrency_20261013.campaign import BINARY, RUNNER

ROOT = analyze.ROOT
PREVIOUS = analyze.PREVIOUS
ALLOCATION = json.loads((ROOT / "allocation.json").read_text())
DEADLINE = ALLOCATION["deadline"]
RESERVE = 900


def decision(event, **fields):
    base.log(event, **fields)
    with (ROOT / "DECISIONS.txt").open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(), event=event, **fields)) + "\n")


def timeout_handler(_signum, _frame):
    raise TimeoutError("Ten-hour TraceLab experiment allocation reached")


def run_group(label, arm, cases, repeat=0):
    replay = base.replay
    target = ROOT / "servers" / label
    target.mkdir(parents=True)
    alias = Path(f"/mnt/localnvme/jeongyoon/ts13{label}")
    assert len(str(alias / "tmp")) <= 45
    alias.symlink_to(target, target_is_directory=True)
    base.save(target / "path_alias.json", dict(alias=str(alias), target=str(target)))
    root, process = base.launch(arm, label, alias)
    durations = []
    try:
        for rate, limit in cases:
            replay.LIMIT = limit
            episode = f"r{rate}_c{limit}_rep{repeat}_{arm}"
            timing = root / "server/timing"
            paths = list(timing.glob("engine-*.jsonl"))
            offsets = {str(path): path.stat().st_size for path in paths}
            clock_before = time.time() - time.monotonic()
            begin = time.monotonic()
            result = base.cell(root, process, arm, rate, episode, 600, 900)
            result.update(
                repeat=repeat,
                stage=label,
                selection_basis="Predeclared capacity sweep; no new arrival selection.",
                reused_server=True,
                logical_caches_reset=True,
                accepted=False,
                capacity_sweep_finalized=False,
            )
            directory = ROOT / "episodes" / episode / "cells" / arm
            base.save(directory / "result.json", result)
            base.save(ROOT / "results" / f"{episode}.json", result)
            result["audit"] = analyze.audit(
                directory,
                result,
                list(timing.glob("engine-*.jsonl")),
                offsets,
                [clock_before, time.time() - time.monotonic()],
            )
            result.update(accepted=True, capacity_sweep_finalized=True)
            result["cell_wall_seconds"] = time.monotonic() - begin
            durations.append(result["cell_wall_seconds"])
            base.save(directory / "result.json", result)
            base.save(ROOT / "results" / f"{episode}.json", result)
            base.log(
                "capacity_cell_audited",
                episode=episode,
                audit_checks=result["audit"]["accounting_checks"],
                cell_wall_seconds=result["cell_wall_seconds"],
            )
            analyze.write_report()
    finally:
        base.stop(root, process)
    return durations


def setup():
    assert not (ROOT / "owner.json").exists(), "Use a fresh output root"
    base.ROOT, base.DEADLINE = ROOT, DEADLINE
    replay = base.replay
    replay.ROOT, replay.SOURCE = ROOT, base.SOURCE
    replay.BINARY, replay.RUNNER = BINARY, RUNNER
    budget = replay.setup()
    (ROOT / "workloads").mkdir()
    manifest = json.loads((PREVIOUS / "dataset_manifest.json").read_text())
    for rate in ("0.15", "0.2", "0.25"):
        workload = PREVIOUS / "workloads" / f"rate_{rate}.csv"
        digest = hashlib.sha256(workload.read_bytes()).hexdigest()
        assert digest == manifest["variants"][rate]["csv_sha256"]
        (ROOT / "workloads" / f"rate_{rate}.csv").symlink_to(workload)
    assert hashlib.sha256(BINARY.read_bytes()).hexdigest() == (
        "a1d83917d7cf74dabccdc14ba4d1396c8a034ef9fc7e3ac60ef66b41f1b73636"
    )
    assert hashlib.sha256(replay.old.STATE.read_bytes()).hexdigest() == (
        "f159f32d226894457fb56b3195c47d4ff6e2a5d982d8d600b1970ea000294f96"
    )
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    assert json.loads((ROOT / "cpu_checks.json").read_text())["passed"]
    base.save(
        ROOT / "owner.json",
        dict(
            pid=os.getpid(),
            time=time.time(),
            deadline=DEADLINE,
            source=str(base.SOURCE),
        ),
    )
    base.save(
        ROOT / "plan.json",
        dict(
            deadline=DEADLINE,
            primary_rate=0.2,
            primary_concurrencies=[32, 40, 48],
            secondary_rates=[0.15, 0.25],
            secondary_concurrencies=[32, 40, 48],
            warmup_seconds=600,
            measurement_seconds=900,
            analysis_reserve_seconds=RESERVE,
            logical_cache_reset_each_cell=True,
            server_reuse_within_arm_group=True,
            active_session_cap=None,
            client_idle_expiry_seconds=1,
            request_retries=False,
            cpu_cache_bytes=budget["cpu_bytes_to_use"],
            gpu_cache_bytes_per_rank=96 * 1024**3,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=base.SOURCE, text=True
            ).strip(),
            runner_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(),
            predictor_sha256=hashlib.sha256(replay.old.STATE.read_bytes()).hexdigest(),
            trace_nonarrival_sha256=manifest["original_content"]["nonarrival_sha256"],
            model="GLM5.3 NVFP4",
            parallelism="TP8",
            original_selector=True,
            serving_changes=False,
            optional_replication=(
                "If time permits, reverse-order repeat the largest measured gain "
                "among 40/48 at 0.2 with >=50% CPU-load requests in both arms. "
                "Keep that confirmation separate and preserve all other results."
            ),
        ),
    )
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max(1, int(DEADLINE - time.time())))


def main():
    setup()
    monitor = subprocess.Popen(
        [
            str(base.SOURCE / ".venv/bin/python"),
            "-m",
            "experiments.tracelab_capacity_sweep_20261013.monitor",
        ],
        cwd=base.SOURCE,
    )
    durations = []
    try:
        decision(
            "frozen_design",
            rationale=(
                "Repeat32 and test40/48 at frozen0.2 first, then arrival sweep. "
                "Amortize unchanged server initialization with cache resets; "
                "600s warmup precedes every900s measurement. Alternate group "
                "order and reverse case order. Preserve all results and content."
            ),
        )
        analyze.write_report()
        main_cases = [("0.2", limit) for limit in (32, 40, 48)]
        durations += run_group("b0", "baseline", main_cases)
        durations += run_group("f0", "full", list(reversed(main_cases)))
        estimate = max(1650, statistics.mean(durations) * 1.08)
        secondary = [("0.15", limit) for limit in (32, 40, 48)] + [
            ("0.25", limit) for limit in (48, 40, 32)
        ]
        remaining = DEADLINE - time.time()
        affordable = min(6, max(0, int((remaining - RESERVE - 1000) / (2 * estimate))))
        if affordable < 6:
            priority = [
                (rate, limit) for rate in ("0.15", "0.25") for limit in (40, 48, 32)
            ]
            secondary = priority[:affordable]
        decision(
            "arrival_sweep_budget",
            cell_wall_estimate_seconds=estimate,
            remaining_seconds=remaining,
            cases=secondary,
            rationale="Time-based subset; preserve paired 600+900 windows.",
        )
        if secondary:
            durations += run_group("f1", "full", secondary)
            durations += run_group("b1", "baseline", list(reversed(secondary)))
        pairs = analyze.write_report()
        candidates = [
            pair
            for pair in pairs
            if pair["rate"] == 0.2
            and pair["concurrency"] in (40, 48)
            and all(
                pair[arm]["cache_profile"]["measurement"]["cpu_hit_request_percent"]
                >= 50
                for arm in ("baseline", "full")
            )
        ]
        estimate = max(1650, statistics.mean(durations) * 1.08)
        if candidates and DEADLINE - time.time() > 2 * estimate + 1000 + RESERVE:
            selected = max(candidates, key=lambda pair: pair["throughput_gain_pct"])
            cases = [("0.2", selected["concurrency"])]
            decision(
                "confirmation_selected",
                concurrency=selected["concurrency"],
                rationale="Reverse-order confirmation; selection bias disclosed.",
            )
            durations += run_group("f2", "full", cases, repeat=1)
            durations += run_group("b2", "baseline", cases, repeat=1)
        else:
            decision(
                "confirmation_skipped",
                remaining_seconds=DEADLINE - time.time(),
                rationale="Preserve windows, draining, shutdown and analysis reserve.",
            )
        base.save(
            ROOT / "completed.json",
            dict(time=time.time(), cells=len(analyze.results()), groups_completed=True),
        )
        base.save(ROOT / "progress.json", dict(complete=True, time=time.time()))
        decision("campaign_complete", cells=len(analyze.results()))
    except BaseException as error:
        base.save(ROOT / "failure.json", dict(time=time.time(), error=repr(error)))
        decision("campaign_failed", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        monitor.wait(timeout=20)
        analyze.write_report()


if __name__ == "__main__":
    main()
