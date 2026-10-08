# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Counterbalanced load-only readiness versus compute deadline evaluation."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_load_only_deadline_ab_20261008")
spec = importlib.util.spec_from_file_location(
    "deadline_driver", SOURCE / "experiments/offset_deadline_ab_20261008/run.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
collector = driver.collector
driver.ROOT = collector.ROOT = ROOT
collector.BASE = "http://127.0.0.1:18413"
POLICIES = ["all_off", "half_off", "all_on", "half_on"]
ORDERS = [
    ["all_off", "half_off", "half_on", "all_on"],
    ["half_off", "all_on", "all_off", "half_on"],
    ["all_on", "half_on", "half_off", "all_off"],
    ["half_on", "all_off", "all_on", "half_off"],
]
REPEATS = {"familiar60k": 4, "short16k": 4, "heterogeneous": 4}
CURRENT_POLICY = "all_off"


def options(active=False):
    result = driver.original_options(active)
    if active:
        enabled = CURRENT_POLICY.endswith("_on")
        result.update(
            late_seal_load_only_ready_percent=50
            if CURRENT_POLICY.startswith("half_")
            else 0,
            late_seal_compute_deadline=enabled,
            late_seal_compute_deadline_shadow=not enabled,
            late_seal_compute_deadline_lead_us=5000,
        )
    return result


def server_command():
    command, cfg = driver.original_command()
    command[command.index("--port") + 1] = "18413"
    return command, cfg


collector.options = options
collector.server_command = server_command


def cell(policy, state, workload, seed, repeat, requests=75):
    global CURRENT_POLICY
    CURRENT_POLICY = policy
    return driver.live.cell(
        policy, state, workload, seed, repeat, requests_count=requests
    )


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
    reference = policies["all_off"]
    for policy, expected in [
        ("half_off", {"late_seal_load_only_ready_percent"}),
        (
            "all_on",
            {"late_seal_compute_deadline", "late_seal_compute_deadline_shadow"},
        ),
        (
            "half_on",
            {
                "late_seal_load_only_ready_percent",
                "late_seal_compute_deadline",
                "late_seal_compute_deadline_shadow",
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
            seeds=[4401 + i for i in range(4)],
            orders=ORDERS,
            scope="Four reset/primed 75-request episodes per arm/workload; "
            "not continuous 300-request cells.",
            output_tokens=1,
            excluded_warmups=12,
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
                cell(policy, state, workload, 4390, "warmup", requests=32)
        for repeat, order in enumerate(ORDERS):
            for workload in REPEATS:
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
