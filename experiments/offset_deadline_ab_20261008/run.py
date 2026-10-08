# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched active/shadow deadline evaluation using one frozen offset predictor."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_offset_deadline_ab_20261008")
MODEL_ROOT = Path("/mnt/localnvme/glm53_offset_predictor_20261008")
COVERAGE = SOURCE / "experiments/dataset_coverage_20261007"
sys.path.insert(0, str(COVERAGE))
spec = importlib.util.spec_from_file_location("run", COVERAGE / "run.py")
collector = importlib.util.module_from_spec(spec)
sys.modules["run"] = collector
spec.loader.exec_module(collector)
import live  # noqa: E402

collector.ROOT = ROOT
collector.BASE = "http://127.0.0.1:18411"
original_command = collector.server_command
original_options = collector.options
original_snapshots = collector.snapshots
original_traces = collector.traces
CURRENT_POLICY = "deadline_off"
LOG_OFFSET = 0
POLICIES = ["deadline_off", "deadline_on"]
REPEATS = {"familiar60k": 8, "short16k": 4, "heterogeneous": 4}


def server_command():
    command, cfg = original_command()
    command[command.index("--port") + 1] = "18411"
    return command, cfg


def options(active=False):
    result = original_options(active)
    if active:
        result.update(
            late_seal_compute_deadline=CURRENT_POLICY == "deadline_on",
            late_seal_compute_deadline_shadow=CURRENT_POLICY == "deadline_off",
            late_seal_compute_deadline_lead_us=5000,
        )
    return result


def snapshots():
    global LOG_OFFSET
    LOG_OFFSET = (ROOT / "server/server.log").stat().st_size
    return original_snapshots()


def traces(offsets, directory):
    with (ROOT / "server/server.log").open() as stream:
        stream.seek(LOG_OFFSET)
        (directory / "measurement_server.log").write_text(stream.read())
    return original_traces(offsets, directory)


collector.server_command = server_command
collector.options = options
collector.snapshots = snapshots
collector.traces = traces


def cell(policy, state, workload, seed, repeat, requests=75):
    global CURRENT_POLICY
    CURRENT_POLICY = policy
    return live.cell(policy, state, workload, seed, repeat, requests_count=requests)


def main():
    if (ROOT / "allocation.json").exists():
        raise RuntimeError("Use a fresh artifact root for another allocation")
    shutil.copy2(
        "/mnt/localnvme/glm53_dataset_coverage_20261007/libseal_execution_observer.so",
        ROOT / "libseal_execution_observer.so",
    )
    checkpoint_path = MODEL_ROOT / "models/matched_offset.json"
    frozen = json.loads((MODEL_ROOT / "freeze.json").read_text())
    assert (
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        == frozen["models"]["matched_offset"]
    )
    state = json.loads(checkpoint_path.read_text())
    assert state["latency_surface"]["architecture"] == "guarded_sparse_mlp_offset"
    for policy in POLICIES:
        collector.save(ROOT / "models" / f"{policy}.json", state)
    policies = {}
    for policy in POLICIES:
        global CURRENT_POLICY
        CURRENT_POLICY = policy
        policies[policy] = options(True)
    changed = {
        key
        for key in policies[POLICIES[0]]
        if policies[POLICIES[0]][key] != policies[POLICIES[1]][key]
    }
    assert changed == {
        "late_seal_compute_deadline",
        "late_seal_compute_deadline_shadow",
    }
    expected_cells = 2 * sum(REPEATS.values())
    collector.save(
        ROOT / "plan.json",
        dict(
            model="GLM 5.3 NVFP4",
            parallelism="TP8",
            concurrency=32,
            requests_per_episode=75,
            repeats=REPEATS,
            measured_cells=expected_cells,
            measured_requests=75 * expected_cells,
            policies=policies,
            checkpoint=str(checkpoint_path),
            checkpoint_file_sha256=hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest(),
            seeds=[4401 + i for i in range(8)],
            order="off/on on even repeats; on/off on odd repeats",
            scope=(
                "Separate reset/primed 75-request episodes; "
                "not continuous 300-request cells."
            ),
            output_tokens=1,
            excluded_warmups=6,
            maximum_allocation_seconds=10800,
        ),
    )
    proc, timer = None, None
    started = time.monotonic()
    results = []
    try:
        collector.log("comparison_started", expected_cells=expected_cells)
        spec = importlib.util.spec_from_file_location(
            "preflight",
            SOURCE / "experiments/coalesced_notify_20261004/environment_preflight.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        uuids = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).splitlines()
        module.check(SOURCE, ROOT, uuids)
        proc = collector.launch()
        collector.save(
            ROOT / "ownership.json", dict(server_pid=proc.pid, session_id=proc.pid)
        )
        timer = threading.Timer(10740, collector.stop_server, (proc,))
        timer.daemon = True
        timer.start()
        ready_deadline = time.monotonic() + 1800
        while time.monotonic() < ready_deadline:
            if proc.poll() is not None:
                raise RuntimeError("Server exited during startup")
            try:
                collector.call("/health", timeout=2)
                break
            except (OSError, TimeoutError):
                time.sleep(2)
        else:
            raise RuntimeError("Startup timeout")
        collector.log("server_ready")
        for workload in REPEATS:
            for policy in POLICIES:
                cell(policy, state, workload, 4390, "warmup", requests=32)
        for repeat in range(max(REPEATS.values())):
            order = POLICIES if repeat % 2 == 0 else list(reversed(POLICIES))
            for workload, repeats in REPEATS.items():
                if repeat >= repeats:
                    continue
                for policy in order:
                    if time.monotonic() - started > 10200:
                        raise RuntimeError("Allocation budget exhausted")
                    results.append(cell(policy, state, workload, 4401 + repeat, repeat))
                    collector.save(
                        ROOT / "progress.json",
                        dict(
                            cells=len(results),
                            expected_cells=expected_cells,
                            latest=dict(
                                policy=policy, workload=workload, repeat=repeat
                            ),
                        ),
                    )
        collector.log("comparison_complete", cells=len(results))
    except BaseException as error:
        collector.log("comparison_failed", error=repr(error))
        raise
    finally:
        if timer:
            timer.cancel()
        if proc:
            collector.stop_server(proc)
        collector.save(
            ROOT / "allocation.json",
            dict(
                seconds=time.monotonic() - started,
                server_stopped=proc is None or proc.poll() is not None,
            ),
        )
        collector.save(
            ROOT / "summary.json",
            dict(complete=len(results) == expected_cells, cells=results),
        )
        collector.log("comparison_stopped", cells=len(results))


if __name__ == "__main__":
    main()
