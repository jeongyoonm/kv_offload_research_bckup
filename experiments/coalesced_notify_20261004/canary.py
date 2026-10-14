# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short matched TP8 A/B cells with total server-process CPU accounting."""

import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import psutil

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_coalesced_notify_20261004/ab")
HELPERS = SOURCE / "experiments/adaptive_seal_20261002"
ROOT.mkdir(parents=True, exist_ok=True)
shutil.copytree(
    "/mnt/localnvme/glm53_adaptive_seal_20261002/code/bin",
    ROOT / "code/bin",
    dirs_exist_ok=True,
)
if not (ROOT / "aiperf_venv").exists():
    (ROOT / "aiperf_venv").symlink_to(
        "/mnt/localnvme/glm53_adaptive_seal_20261002/aiperf_venv"
    )
sys.path[:0] = [str(HELPERS), str(ROOT / "code/bin")]
spec = importlib.util.spec_from_file_location("adaptive_run", HELPERS / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.SOURCE, runner.ROOT = SOURCE, ROOT
policy = json.loads(
    Path("/mnt/localnvme/glm53_full_predicted_wait_20261004/policies.json").read_text()
)["full_margin50_fine2"]
runner.POLICIES = {
    "bridge": dict(policy, late_seal_coalesced_notifications=False),
    "coalesced": dict(policy, late_seal_coalesced_notifications=True),
}
original_config = runner.early.config


def config(*args, **kwargs):
    cfg = original_config(*args, **kwargs)
    cfg["kv_connector_extra_config"]["coalesced_seal_notifications"] = True
    return cfg


runner.early.config = config
original_profile = runner.profile


def profile(directory, dataset, concurrency, count, phase):
    if phase != "replay":
        return original_profile(directory, dataset, concurrency, count, phase)
    api_pid = json.loads((ROOT / "gpu/server/process.json").read_text())["pid"]
    processes = [psutil.Process(api_pid)] + psutil.Process(api_pid).children(
        recursive=True
    )
    identities = {p.pid: dict(name=p.name(), cmdline=p.cmdline()) for p in processes}
    snapshots, errors = [], []
    stopped = threading.Event()

    def sample():
        values = {}
        for process in processes:
            try:
                times = process.cpu_times()
                values[str(process.pid)] = times.user + times.system
            except psutil.Error as error:
                errors.append(dict(pid=process.pid, error=str(error)))
        snapshots.append(dict(time_ns=time.monotonic_ns(), cpu_seconds=values))

    def monitor():
        while not stopped.wait(0.025):
            sample()

    sample()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        return original_profile(directory, dataset, concurrency, count, phase)
    finally:
        stopped.set()
        thread.join(timeout=2)
        sample()
        (directory / "server_cpu.json").write_text(
            json.dumps(
                dict(
                    identities=identities,
                    samples=snapshots,
                    errors=errors,
                    interval_ms=25,
                    request_count=count,
                ),
                indent=2,
            )
            + "\n"
        )


runner.profile = profile
original_cells = runner.run_cells


def cells(run, server_dir, workloads, counts, policies, repeats, seed=55):
    # Train the common predictor on both workloads, then freeze compute and
    # preparation histories before any measured A/B cells or their priming.
    original_cells(
        ROOT / "training", server_dir, workloads, counts, ["bridge"], 1, seed=61
    )
    runner.PRIME = dict(runner.PRIME, late_seal_freeze_estimator=True)
    runner.POLICIES = {
        key: dict(value, late_seal_freeze_estimator=True)
        for key, value in runner.POLICIES.items()
    }
    runner.configure(run, runner.PRIME, "freeze_setting")
    (ROOT / "policies.json").write_text(json.dumps(runner.POLICIES, indent=2) + "\n")
    return original_cells(
        run, server_dir, workloads, counts, policies, repeats, seed=62
    )


runner.run_cells = cells
if __name__ == "__main__":
    os.environ["VLLM_DEFERRAL_NATIVE_OBSERVER"] = str(
        ROOT.parent / "build/libseal_execution_observer.so"
    )
    with (ROOT / "process_log.jsonl").open("a") as log:
        log.write(
            json.dumps(dict(event="started", time=time.time(), argv=sys.argv)) + "\n"
        )
    try:
        # Resuming saved cells would skip common training on a fresh server.
        if any((ROOT / "gpu").glob("p*/*/result.json")):
            raise RuntimeError("Archive the existing A/B directory before rerunning")
        from environment_preflight import check

        check(SOURCE, ROOT, runner.glm.UUIDS)
        runner.main()
    except BaseException as error:
        with (ROOT / "process_log.jsonl").open("a") as log:
            log.write(
                json.dumps(dict(event="failed", time=time.time(), error=repr(error)))
                + "\n"
            )
        raise
    else:
        with (ROOT / "process_log.jsonl").open("a") as log:
            log.write(json.dumps(dict(event="completed", time=time.time())) + "\n")
