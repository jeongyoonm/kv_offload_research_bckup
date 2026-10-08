# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Counterbalanced waiting ablation with unchanged authoritative seal conditions."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_efficient_wait_ab_20261009")
spec = importlib.util.spec_from_file_location(
    "deadline_driver", SOURCE / "experiments/offset_deadline_ab_20261008/run.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
collector = driver.collector
driver.ROOT = collector.ROOT = ROOT
collector.BASE = "http://127.0.0.1:18415"
POLICIES = ["full", "bridge", "poll"]
ORDERS = [
    ["full", "bridge", "poll"],
    ["bridge", "poll", "full"],
    ["poll", "full", "bridge"],
    ["full", "poll", "bridge"],
    ["poll", "bridge", "full"],
    ["bridge", "full", "poll"],
]
REPEATS = {"familiar60k": 6, "short16k": 6, "heterogeneous": 6}
CURRENT_POLICY = "full"
CPU_START = None
CPU_CELL = None
original_snapshots = collector.snapshots
original_traces = collector.traces
spec = importlib.util.spec_from_file_location(
    "wait_cpu", SOURCE / "experiments/efficient_wait_ab_20261009/cpu.py"
)
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)


def options(active=False):
    result = driver.original_options(active)
    if active:
        result.update(
            late_seal_load_only_ready_percent=50,
            late_seal_compute_deadline=True,
            late_seal_compute_deadline_shadow=False,
            late_seal_compute_deadline_lead_us=5000,
            late_seal_wait_trace=True,
            late_candidate_slots=16,
            late_seal_adaptive_poll_us=2000,
            late_seal_full_prediction_wait=CURRENT_POLICY != "poll",
            late_seal_coalesced_notifications=CURRENT_POLICY == "full",
            late_seal_hybrid_wait_us=0 if CURRENT_POLICY == "poll" else 50000,
        )
    return result


def snapshots():
    global CPU_START
    CPU_START = cpu.snapshot(ROOT)
    return original_snapshots()


def traces(offsets, directory):
    global CPU_CELL
    end = cpu.snapshot(ROOT)
    CPU_CELL = cpu.difference(CPU_START, end)
    collector.save(directory / "cpu.json", CPU_CELL)
    return original_traces(offsets, directory)


def server_command():
    command, cfg = driver.original_command()
    command[command.index("--port") + 1] = "18415"
    return command, cfg


collector.options = options
collector.snapshots = snapshots
collector.traces = traces
collector.server_command = server_command


def cell(policy, state, workload, seed, repeat, requests=75):
    global CURRENT_POLICY
    CURRENT_POLICY = policy
    result = driver.live.cell(
        policy, state, workload, seed, repeat, requests_count=requests
    )
    if not result["accepted"]:
        raise RuntimeError(f"Rejected measurement: {workload}/{repeat}/{policy}")
    return result


def main():
    global CURRENT_POLICY
    if (ROOT / "allocation.json").exists():
        raise RuntimeError("Use a fresh artifact root for another allocation")
    shutil.copy2(
        "/mnt/localnvme/glm53_dataset_coverage_20261007/libseal_execution_observer.so",
        ROOT / "libseal_execution_observer.so",
    )
    checkpoint = driver.MODEL_ROOT / "models/matched_offset.json"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    freeze = json.loads((driver.MODEL_ROOT / "freeze.json").read_text())
    assert digest == freeze["models"]["matched_offset"]
    state = json.loads(checkpoint.read_text())
    policies = {}
    for policy in POLICIES:
        CURRENT_POLICY = policy
        policies[policy] = options(True)
        collector.save(ROOT / "models" / f"{policy}.json", state)
    reference = policies["full"]
    for policy, expected in [
        ("bridge", {"late_seal_coalesced_notifications"}),
        (
            "poll",
            {
                "late_seal_coalesced_notifications",
                "late_seal_full_prediction_wait",
                "late_seal_hybrid_wait_us",
            },
        ),
    ]:
        assert {k for k in reference if reference[k] != policies[policy][k]} == expected
    expected_cells = len(POLICIES) * sum(REPEATS.values())
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
            checkpoint=str(checkpoint),
            checkpoint_file_sha256=digest,
            seeds=[5501 + i for i in range(6)],
            orders=ORDERS,
            scope="Six reset/primed 75-request episodes per arm/workload; "
            "not continuous 300-request cells.",
            output_tokens=1,
            excluded_warmups=9,
            maximum_allocation_seconds=10800,
        ),
    )
    collector.save(
        ROOT / "source_provenance.json",
        dict(
            source=str(SOURCE),
            branch=subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=SOURCE, text=True
            ).strip(),
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
            ).strip(),
            serving_parent="673ddfa9fc3a2a02df2d202de01166b428624a05",
        ),
    )
    collector.save(
        ROOT / "SESSION_STATE.json",
        dict(
            status="starting",
            branch="experiment/glm53-efficient-wait-ablation-20261009",
            source=str(SOURCE),
            plan=str(ROOT / "plan.json"),
        ),
    )
    proc = timer = None
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
        processes = subprocess.check_output(
            ["ps", "-eo", "pid,sid,comm"], text=True
        ).splitlines()[1:]
        owned = [
            {"pid": int(p), "session_id": int(s), "command": c}
            for line in processes
            for p, s, c in [line.split(maxsplit=2)]
            if int(s) == proc.pid
        ]
        collector.save(
            ROOT / "ownership.json",
            dict(server_pid=proc.pid, session_id=proc.pid, processes=owned),
        )
        collector.log("server_ready")
        for workload in REPEATS:
            for policy in POLICIES:
                cell(policy, state, workload, 5490, "warmup", requests=32)
        for repeat, order in enumerate(ORDERS):
            for workload in REPEATS:
                for policy in order:
                    if time.monotonic() - started > 10200:
                        raise RuntimeError("Allocation budget exhausted")
                    results.append(cell(policy, state, workload, 5501 + repeat, repeat))
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
