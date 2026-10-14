# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Match uncached and cached dispatch-horizon serving cells."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from experiments.early_ready_preemption_fix_20261014 import campaign as previous
from experiments.kv_capacity_retirement_fix_20261014.campaign import summarize

ROOT = Path("/mnt/localnvme/jeongyoon/cached_dispatch_horizon_20261014")
FIXED = Path(__file__).resolve().parents[2]
ORIGINAL = Path("/mnt/localnvme/jeongyoon/vllm_kv_capacity_retirement_fix_20261014")
study = previous.study


def configure(source, deadline):
    previous.ROOT = ROOT
    budget = previous.configure(source, deadline)
    base = study.base.replay.old.server.old.base
    factory = getattr(configure, "factory", None)
    if factory is None:
        factory = configure.factory = base.server_command

    def command():
        args, config = factory()
        config["kv_connector_extra_config"].update(
            capacity_boundary_accounting=True, retirement_aware_preemption=True
        )
        return args, config

    base.server_command = command
    return budget


def report():
    from experiments.kv_capacity_retirement_fix_20261014 import campaign as prior

    prior.ROOT = ROOT
    return summarize()


def main():
    assert not (ROOT / "owner.json").exists()
    assert json.loads((ROOT / "cpu_checks.json").read_text())["passed"]
    deadline = time.time() + 5 * 3600
    budget = configure(ORIGINAL, deadline)
    assert budget["cpu_bytes_to_use"] == 3028287488000
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
            original_source=str(ORIGINAL),
            fixed_source=str(FIXED),
            original_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ORIGINAL, text=True
            ).strip(),
            fixed_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=FIXED, text=True
            ).strip(),
            cpu_cache_bytes=budget["cpu_bytes_to_use"],
            gpu_cache_bytes_per_rank=96 * 1024**3,
            predictor_sha256=hashlib.sha256(
                study.base.replay.old.STATE.read_bytes()
            ).hexdigest(),
            comparisons=[["0.2", 32], ["0.2", 48]],
            warmup_seconds=600,
            measurement_seconds=900,
            request_retries=False,
            deadline=deadline,
            clarification=(
                "Tags: repeat0 uncached capacity fix, repeat1 cached capacity fix. "
                "Both capacity and retirement policies enabled in both sources. "
                "Original pre-capacity prototype is a historical reference only."
            ),
        ),
    )
    signal.signal(signal.SIGALRM, study.timeout_handler)
    signal.alarm(int(deadline - time.time()))
    monitor_log = (ROOT / "monitor.log").open("a")
    monitor = subprocess.Popen(
        [
            str(FIXED / ".venv/bin/python"),
            "-m",
            "experiments.cached_dispatch_horizon_20261014.monitor",
        ],
        cwd=FIXED,
        stdout=monitor_log,
        stderr=subprocess.STDOUT,
    )
    try:
        study.decision("uncached_started", rationale="Fresh matched regression control")
        study.run_group("dh0", "full", [("0.2", 32), ("0.2", 48)], repeat=0)
        configure(FIXED, deadline)
        study.decision(
            "cached_started", rationale="Only dispatch-depth snapshot changed"
        )
        study.run_group("dh1", "full", [("0.2", 48), ("0.2", 32)], repeat=1)
        rows, pairs = report()
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
        report()


if __name__ == "__main__":
    main()
