# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short GLM TP8 cached-reference and pinned-packing integration cells."""

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
ROOT = Path("/mnt/localnvme/glm53_cached_pinned_20261005/canary")
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
    "current": dict(
        policy,
        late_seal_coalesced_notifications=True,
        late_cached_requests=False,
        late_pinned_packing=False,
    ),
    "references": dict(
        policy,
        late_seal_coalesced_notifications=True,
        late_cached_requests=True,
        late_pinned_packing=False,
    ),
    "cached_pinned": dict(
        policy,
        late_seal_coalesced_notifications=True,
        late_cached_requests=True,
        late_pinned_packing=True,
    ),
}
original_config = runner.early.config


def config(*args, **kwargs):
    cfg = original_config(*args, **kwargs)
    cfg["kv_connector_extra_config"]["coalesced_seal_notifications"] = True
    return cfg


runner.early.config = config
runner.glm.BASE = "http://127.0.0.1:18255"
original_launch = runner.early.original_popen


def launch(command, *args, **kwargs):
    if isinstance(command, list) and "serve" in command:
        command = list(command)
        command[command.index("--port") + 1] = "18255"
        env = dict(kwargs.get("env", os.environ))
        env["VLLM_CACHE_ROOT"] = str(ROOT.parent / "vllm-cache")
        env["TRITON_CACHE_DIR"] = str(ROOT.parent / "triton-cache")
        env["CUDA_CACHE_PATH"] = str(ROOT.parent / "cuda-cache")
        kwargs["env"] = env
    return original_launch(command, *args, **kwargs)


runner.early.original_popen = launch
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
        ROOT / "training", server_dir, workloads, counts, ["current"], 1, seed=61
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
        Path(
            "/mnt/localnvme/glm53_coalesced_notify_20261004/build/libseal_execution_observer.so"
        )
    )
    with (ROOT / "process_log.jsonl").open("a") as log:
        log.write(
            json.dumps(dict(event="started", time=time.time(), argv=sys.argv)) + "\n"
        )
    try:
        # Resuming saved cells would skip common training on a fresh server.
        if any((ROOT / "gpu").glob("p*/*/result.json")):
            raise RuntimeError("Archive the existing A/B directory before rerunning")
        sys.path.insert(0, str(SOURCE / "experiments/coalesced_notify_20261004"))
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
