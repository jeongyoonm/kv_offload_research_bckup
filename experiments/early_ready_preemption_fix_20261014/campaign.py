# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate formerly failing cells and compare original/fixed serving performance."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from experiments.tracelab_capacity_sweep_20261013 import campaign as study

ROOT = Path("/mnt/localnvme/jeongyoon/early_ready_preemption_fix_20261014")
FIXED = Path(__file__).resolve().parents[2]
ORIGINAL = Path("/mnt/localnvme/jeongyoon/vllm_tracelab_arrival_intensity_20261013")


def configure(source, deadline):
    study.ROOT = study.analyze.ROOT = study.base.ROOT = ROOT
    study.DEADLINE = study.base.DEADLINE = deadline
    study.base.SOURCE = source
    replay = study.base.replay
    replay.ROOT, replay.SOURCE = ROOT, source
    replay.BINARY, replay.RUNNER = study.BINARY, study.RUNNER
    return replay.setup()


def comparisons():
    rows = study.analyze.results()
    pairs = []
    for concurrency in (32, 48):
        arms = {
            r["repeat"]: r
            for r in rows
            if r["arrival_rate_sessions_per_second"] == 0.2
            and r["measurement_request_slots"] == concurrency
        }
        if not {0, 1} <= arms.keys():
            continue
        original, fixed = arms[0], arms[1]
        pairs.append(
            dict(
                concurrency=concurrency,
                original=original,
                fixed=fixed,
                throughput_change_pct=(
                    fixed["throughput_rps"] / original["throughput_rps"] - 1
                )
                * 100,
                mean_ttft_change_pct=(
                    fixed["ttft_ms"]["mean"] / original["ttft_ms"]["mean"] - 1
                )
                * 100,
                p95_ttft_change_pct=(
                    fixed["ttft_ms"]["p95"] / original["ttft_ms"]["p95"] - 1
                )
                * 100,
            )
        )
    study.base.save(ROOT / "performance_comparisons.json", pairs)
    return pairs


def proof_events():
    events = []
    prefix = '{"event": "preempt_copy_complete_load_pending_retirement",'
    for path in (ROOT / "servers").glob("*/server/load_timeline/*.jsonl"):
        with path.open() as stream:
            for line in stream:
                if line.startswith(prefix):
                    events.append(dict(server=path.parts[-4], **json.loads(line)))
    study.base.save(ROOT / "live_preemption_proofs.json", events)
    return events


def main():
    assert not (ROOT / "owner.json").exists()
    deadline = time.time() + 8 * 3600
    budget = configure(FIXED, deadline)
    (ROOT / "workloads").mkdir()
    for rate in ("0.15", "0.2", "0.25"):
        (ROOT / "workloads" / f"rate_{rate}.csv").symlink_to(
            study.PREVIOUS / "workloads" / f"rate_{rate}.csv"
        )
    study.base.save(
        ROOT / "owner.json", dict(pid=os.getpid(), time=time.time(), deadline=deadline)
    )
    study.base.save(
        ROOT / "plan.json",
        dict(
            cpu_cache_bytes=budget["cpu_bytes_to_use"],
            gpu_cache_bytes_per_rank=96 * 1024**3,
            frozen_predictor=str(study.base.replay.old.STATE),
            original_source=str(ORIGINAL),
            fixed_source=str(FIXED),
            correctness_cases=[["0.25", 48], ["0.15", 32], ["0.15", 48]],
            performance_cases=[["0.2", 32], ["0.2", 48]],
            warmup=600,
            measurement=900,
            request_retries=False,
            original_repeat=0,
            fixed_repeat=1,
            clarification="repeat tags distinguish source versions, not replications",
            watchdog_deadline=deadline,
        ),
    )
    signal.signal(signal.SIGALRM, study.timeout_handler)
    signal.alarm(int(deadline - time.time()))
    monitor_log = (ROOT / "monitor.log").open("a")
    monitor = subprocess.Popen(
        [
            str(FIXED / ".venv/bin/python"),
            "-m",
            "experiments.early_ready_preemption_fix_20261014.monitor",
        ],
        cwd=FIXED,
        stdout=monitor_log,
        stderr=subprocess.STDOUT,
    )
    try:
        study.decision(
            "fixed_correctness_started",
            rationale="Run complete windows at the three previously failing points",
        )
        study.run_group(
            "ef0r", "full", [("0.25", 48), ("0.15", 32), ("0.15", 48)], repeat=1
        )
        proofs = proof_events()
        study.decision("fixed_correctness_passed", live_preemption_proofs=len(proofs))
        configure(ORIGINAL, deadline)
        study.run_group("eo0", "full", [("0.2", 32), ("0.2", 48)], repeat=0)
        configure(FIXED, deadline)
        study.run_group("ef1", "full", [("0.2", 48), ("0.2", 32)], repeat=1)
        pairs = comparisons()
        proof_events()
        study.base.save(
            ROOT / "completed.json",
            dict(
                time=time.time(),
                accepted_cells=len(study.analyze.results()),
                performance_pairs=len(pairs),
            ),
        )
        study.decision(
            "fix_campaign_complete", accepted_cells=len(study.analyze.results())
        )
    except BaseException as error:
        study.base.save(
            ROOT / "failure.json", dict(time=time.time(), error=repr(error))
        )
        study.decision("fix_campaign_failed", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        monitor.wait(timeout=20)
        comparisons()
        proof_events()


if __name__ == "__main__":
    main()
