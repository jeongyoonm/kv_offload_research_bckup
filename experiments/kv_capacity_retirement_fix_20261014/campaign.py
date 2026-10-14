# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen original/fixed TP8 cells within the user's three-hour allocation."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from experiments.early_ready_preemption_fix_20261014 import campaign as previous

ROOT = Path("/mnt/localnvme/jeongyoon/kv_capacity_retirement_fix_20261014")
FIXED = Path(__file__).resolve().parents[2]
ORIGINAL = Path("/mnt/localnvme/jeongyoon/vllm_early_ready_preemption_fix_20261014")
DEADLINE = 1791937184 + 10800
study = previous.study


def configure(source, capacity=False, retirement=False):
    previous.ROOT = ROOT
    budget = previous.configure(source, DEADLINE)
    base = study.base.replay.old.server.old.base
    original = getattr(configure, "command_factory", None)
    if original is None:
        original = configure.command_factory = base.server_command

    def command():
        args, config = original()
        if source == FIXED:
            config["kv_connector_extra_config"].update(
                capacity_boundary_accounting=capacity,
                retirement_aware_preemption=retirement,
            )
        return args, config

    base.server_command = command
    return budget


def summarize():
    rows = study.analyze.results()
    pairs = []
    for concurrency in (32, 48):
        cells = {
            r["repeat"]: r
            for r in rows
            if r["measurement_request_slots"] == concurrency
        }
        if not {0, 1} <= cells.keys():
            continue
        old, fixed = cells[0], cells[1]
        pairs.append(
            dict(
                concurrency=concurrency,
                original=old,
                fixed=fixed,
                throughput_change_pct=100
                * (fixed["throughput_rps"] / old["throughput_rps"] - 1),
                mean_ttft_change_pct=100
                * (fixed["ttft_ms"]["mean"] / old["ttft_ms"]["mean"] - 1),
                p95_ttft_change_pct=100
                * (fixed["ttft_ms"]["p95"] / old["ttft_ms"]["p95"] - 1),
                preemptions_original=old["metric_deltas"].get(
                    "vllm:num_preemptions_total", 0
                ),
                preemptions_fixed=fixed["metric_deltas"].get(
                    "vllm:num_preemptions_total", 0
                ),
            )
        )
    study.base.save(ROOT / "performance_comparisons.json", pairs)
    return rows, pairs


def main():
    assert not (ROOT / "owner.json").exists()
    budget = configure(ORIGINAL)
    (ROOT / "workloads").mkdir()
    for rate in ("0.15", "0.2", "0.25"):
        (ROOT / "workloads" / f"rate_{rate}.csv").symlink_to(
            study.PREVIOUS / "workloads" / f"rate_{rate}.csv"
        )
    study.base.save(
        ROOT / "owner.json", dict(pid=os.getpid(), time=time.time(), deadline=DEADLINE)
    )
    study.base.save(
        ROOT / "plan.json",
        dict(
            original_source=str(ORIGINAL),
            fixed_source=str(FIXED),
            cpu_cache_bytes=budget["cpu_bytes_to_use"],
            gpu_cache_bytes_per_rank=96 * 1024**3,
            predictor_sha256=hashlib.sha256(
                study.base.replay.old.STATE.read_bytes()
            ).hexdigest(),
            comparisons=[["0.2", 32], ["0.2", 48]],
            warmup_seconds=600,
            measurement_seconds=900,
            optional_accounting_only=True,
            request_retries=False,
            deadline=DEADLINE,
            clarification=(
                "Repeat tags identify variants: 0 original, 1 full fix, "
                "2 accounting only; not statistical replicates"
            ),
        ),
    )
    signal.signal(signal.SIGALRM, study.timeout_handler)
    signal.alarm(max(1, int(DEADLINE - time.time())))
    monitor_log = (ROOT / "monitor.log").open("a")
    monitor = subprocess.Popen(
        [
            str(FIXED / ".venv/bin/python"),
            "-m",
            "experiments.kv_capacity_retirement_fix_20261014.monitor",
        ],
        cwd=FIXED,
        stdout=monitor_log,
        stderr=subprocess.STDOUT,
    )
    try:
        study.decision(
            "original_control_started",
            rationale=(
                "Freeze model, predictor, cache budgets, workload and 600+900s windows"
            ),
        )
        study.run_group("kc0", "full", [("0.2", 32), ("0.2", 48)], repeat=0)
        assert json.loads((ROOT / "cpu_checks.json").read_text())["passed"]
        configure(FIXED, capacity=True, retirement=True)
        study.run_group("kc1", "full", [("0.2", 48), ("0.2", 32)], repeat=1)
        rows, pairs = summarize()
        if DEADLINE - time.time() >= 2600:
            configure(FIXED, capacity=True, retirement=False)
            study.decision(
                "accounting_only_started",
                rationale=(
                    "Isolate reservation protection from retirement-aware retry "
                    "while time remains"
                ),
            )
            study.run_group("kc2", "full", [("0.2", 32)], repeat=2)
        rows, pairs = summarize()
        study.base.save(
            ROOT / "completed.json",
            dict(time=time.time(), accepted_cells=len(rows), matched_pairs=len(pairs)),
        )
    except BaseException as error:
        study.base.save(
            ROOT / "failure.json", dict(time=time.time(), error=repr(error))
        )
        raise
    finally:
        signal.alarm(0)
        if monitor.poll() is None:
            monitor.terminate()
        monitor.wait(timeout=15)
        summarize()


if __name__ == "__main__":
    main()
