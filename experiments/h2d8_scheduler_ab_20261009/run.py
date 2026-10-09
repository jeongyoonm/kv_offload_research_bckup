# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired scheduler ablation with the improved 8 MiB/depth 2 H2D worker."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

from experiments.h2d_interference_20261009.serving import state_for_chunk
from vllm.v1.kv_offload.early_readiness import (
    EXPERIMENT_DEFAULTS,
    validate_experiment_options,
)

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "scheduler_ab_wait", SOURCE / "experiments/efficient_wait_ab_20261009/run.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
collector = module.collector
CURRENT_POLICY = "D"
original_command = module.server_command


def options(active=False):
    result = module.options(active)
    result["copy_pipeline_depth"] = 2
    if active and CURRENT_POLICY == "C":
        for key, value in EXPERIMENT_DEFAULTS.items():
            if key.startswith("late_"):
                result[key] = value
        result.update(late_admission_markers=True, late_seal_freeze_estimator=True)
    return result


def server_command():
    command, config = original_command()
    command[0] = str(SOURCE / ".venv/bin/python")
    command[command.index("--port") + 1] = "18421"
    config["kv_connector_extra_config"]["async_h2d_chunk_bytes"] = 8 * 1024**2
    command[command.index("--kv-transfer-config") + 1] = json.dumps(config)
    return command, config


def main():
    global CURRENT_POLICY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--requests", type=int, default=75)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "plan.json").exists(), "Use a fresh output root"
    module.ROOT = module.driver.ROOT = collector.ROOT = root
    collector.BASE = "http://127.0.0.1:18421"
    collector.options = options
    collector.server_command = server_command
    checkpoint = module.driver.MODEL_ROOT / "models/matched_offset.json"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    freeze = json.loads((module.driver.MODEL_ROOT / "freeze.json").read_text())
    assert digest == freeze["models"]["matched_offset"]
    state = state_for_chunk(json.loads(checkpoint.read_text()), 8)
    collector.save(root / "loaded_predictor.json", state)
    policies = {}
    for policy in ("C", "D"):
        CURRENT_POLICY = policy
        policies[policy] = options(True)
        validate_experiment_options(policies[policy])
    changed = {k for k in policies["C"] if policies["C"][k] != policies["D"][k]}
    assert changed and all(k.startswith("late_") for k in changed)
    for policy in policies.values():
        assert policy["early_submission"] and policy["cooperative_response_progress"]
        assert policy["copy_pipeline_depth"] == 2
    assert policies["C"]["late_admission_mode"] == 0
    assert policies["D"]["late_admission_mode"] == 4
    assert policies["D"]["late_seal_compute_deadline"]
    command, config = server_command()
    assert config["kv_connector_extra_config"]["async_h2d_submission"]
    workloads = ("short16k", "familiar60k", "heterogeneous")
    expected = 2 * len(workloads) * args.repeats
    collector.save(
        root / "plan.json",
        dict(
            model="GLM 5.3 NVFP4",
            parallelism="TP8",
            chunk_mib=8,
            depth=2,
            policies=policies,
            changed_policy_fields=sorted(changed),
            workloads=workloads,
            repeats=args.repeats,
            requests=args.requests,
            concurrency=32,
            measured_cells=expected,
            checkpoint_file_sha256=digest,
            command=command,
            connector=config,
            order="C/D on even repeats; D/C on odd repeats",
            seeds=[8101 + i for i in range(args.repeats)],
            output_tokens=1,
            scope="Paired reset/primed episodes; not continuous 300-request cells. "
            "C disables provisional preparation and late admission; "
            "D retains optimized preparation and full hold/seal policy. "
            "All worker-side transfer settings and early submission are identical.",
        ),
    )
    collector.save(
        root / "source_provenance.json",
        dict(
            source=str(SOURCE),
            branch=subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=SOURCE, text=True
            ).strip(),
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
            ).strip(),
        ),
    )
    if args.preflight_only:
        print("C/D configuration validation passed", flush=True)
        return
    shutil.copy2(
        "/mnt/localnvme/glm53_dataset_coverage_20261007/libseal_execution_observer.so",
        root / "libseal_execution_observer.so",
    )
    results = []
    proc = timer = None
    started = time.monotonic()

    def cell(policy, workload, seed, repeat, count):
        global CURRENT_POLICY
        CURRENT_POLICY = policy
        result = module.driver.live.cell(
            policy,
            state,
            workload,
            seed,
            repeat,
            requests_count=count,
            require_preparation=policy == "D",
        )
        assert result["accepted"], result
        if policy == "C":
            assert result["late_admissions"] == 0
            assert not result["worker_preparation_coverage"]
        return result

    try:
        collector.log("comparison_started", expected_cells=expected)
        proc = collector.launch()
        collector.save(root / "ownership.json", dict(server_pid=proc.pid))
        timer = threading.Timer(10740, collector.stop_server, (proc,))
        timer.daemon = True
        timer.start()
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            assert proc.poll() is None, "Server exited during startup"
            try:
                collector.call("/health", timeout=2)
                break
            except (OSError, TimeoutError):
                time.sleep(2)
        else:
            raise TimeoutError("Serving startup timeout")
        collector.log("server_ready")
        for workload in workloads:
            for policy in ("C", "D"):
                cell(policy, workload, 8090, "warmup", 32)
        for repeat in range(args.repeats):
            order = ("C", "D") if repeat % 2 == 0 else ("D", "C")
            for workload in workloads:
                for policy in order:
                    assert time.monotonic() - started < 10200, "Budget exhausted"
                    results.append(
                        cell(policy, workload, 8101 + repeat, repeat, args.requests)
                    )
                    collector.save(
                        root / "progress.json",
                        dict(
                            cells=len(results),
                            expected_cells=expected,
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
            root / "summary.json",
            dict(
                complete=len(results) == expected,
                results=results,
                seconds=time.monotonic() - started,
                stopped=proc is None or proc.poll() is not None,
            ),
        )


if __name__ == "__main__":
    main()
