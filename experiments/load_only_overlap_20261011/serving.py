# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fresh paired serving tests with frozen predictors and identical seal policy."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from experiments.load_only_overlap_20261011 import live
from experiments.native_scheduler_ab_20261010 import run as baseline
from experiments.shape_calibration_20261010.run import PREFIX_WORKING_SETS
from vllm.v1.kv_offload.early_readiness import (
    EXPERIMENT_DEFAULTS,
    validate_experiment_options,
)
from vllm.v1.kv_offload.seal_predictor import ComputeEstimator

SOURCE = Path(__file__).resolve().parents[2]
module, collector = baseline.module, baseline.collector
CURRENT_ARM = ""


def main():
    global CURRENT_ARM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=STATE")
    parser.add_argument("--bias-us", action="append", default=[], metavar="NAME=US")
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["short16k", "mid32k", "familiar60k", "unseen48k"],
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--requests", type=int, choices=(32, 75, 300), default=300)
    parser.add_argument("--seed-start", type=int, default=12101)
    parser.add_argument(
        "--idle-arm", action="append", default=[], metavar="NAME=TARGET"
    )
    parser.add_argument("--initial-barrier", action="store_true")
    parser.add_argument("--block-arm", action="append", default=[])
    parser.add_argument("--worker-arm", action="append", default=[])
    parser.add_argument("--dma-audit", action="store_true")
    parser.add_argument("--lead-us", type=int, default=50000)
    parser.add_argument("--arm-lead-us", action="append", default=[], metavar="NAME=US")
    parser.add_argument("--allocation-seconds", type=int, default=10800)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert len(str(root / "tmp")) <= 45, "Use a short Unix IPC path"
    assert not (root / "plan.json").exists(), "Use a fresh output root"
    paths = dict(item.split("=", 1) for item in args.arm)
    assert len(paths) == len(args.arm)
    assert set(args.workloads) <= set(PREFIX_WORKING_SETS)
    idle_arms = {
        name: int(value)
        for name, value in (item.split("=", 1) for item in args.idle_arm)
    }
    assert set(idle_arms) <= set(paths)
    biases = {name: 0 for name in paths}
    leads = dict.fromkeys(paths, args.lead_us)
    for item in args.arm_lead_us:
        name, lead = item.split("=", 1)
        assert name in paths
        leads[name] = int(lead)
        assert leads[name] >= 0
    for item in args.bias_us:
        name, bias = item.split("=", 1)
        assert name in paths
        biases[name] = int(bias)
    baseline.CURRENT_POLICY = "D"
    active, inactive = baseline.options(True), baseline.options(False)
    command, config = baseline.server_command()
    if args.dma_audit:
        config.update(
            kv_connector="AuditedOffloadingConnector",
            kv_connector_module_path="experiments.load_only_overlap_20261011.dma_audit",
        )
        command[command.index("--kv-transfer-config") + 1] = json.dumps(config)
        os.environ["CACHE_INIT_DMA_AUDIT_DIR"] = str(root / "dma_audit")
    command[command.index("--port") + 1] = "18462"
    states, policies = {}, {}
    context = None
    for name, path in paths.items():
        state = json.loads(Path(path).read_text())
        surface = state["latency_surface"]
        assert surface["schema"] == 4
        assert "duration_calibration" not in surface["checkpoint"]
        assert surface["checkpoint"]["excluded_feature_indices"] == [21]
        if context is None:
            context = state["context"]
        assert state["context"] == context
        estimator = ComputeEstimator.from_state_dict(state, context)
        assert estimator.state_dict(context) == state
        states[name] = state
        policy = copy.deepcopy(active)
        policy.update(
            late_seal_compute_deadline_lead_us=leads[name],
            late_seal_prediction_bias_us=biases[name],
        )
        if name in idle_arms:
            policy.update(
                late_seal_load_only_consistent=True,
                late_seal_load_only_idle_target=idle_arms[name],
            )
        if name in args.worker_arm:
            for key, value in EXPERIMENT_DEFAULTS.items():
                if key.startswith("late_"):
                    policy[key] = value
            policy.update(late_admission_markers=True, late_seal_freeze_estimator=True)
        policy["h2d_block_readiness"] = name in args.block_arm
        policy = validate_experiment_options(policy)
        policies[name] = policy
    for policy in policies.values():
        assert policy["late_seal_freeze_estimator"]
        assert policy["copy_pipeline_depth"] == 2
        assert policy["store_reuse_gpu_fence"]
        assert policy["store_reuse_allocator_preference"]
        changed = {k for k in policy if policy[k] != policies[next(iter(paths))][k]}
        assert all(
            key.startswith("late_")
            for key in changed
            - {
                "h2d_block_readiness",
            }
        )
        assert changed - {key for key in changed if key.startswith("late_")} <= {
            "h2d_block_readiness",
            "late_seal_prediction_bias_us",
            "late_seal_compute_deadline_lead_us",
            "late_seal_load_only_consistent",
            "late_seal_load_only_idle_target",
        }
        if args.arm_lead_us:
            assert changed <= {"late_seal_compute_deadline_lead_us"}
    if args.arm_lead_us:
        reference = states[next(iter(paths))]
        assert all(state == reference for state in states.values())
    extra = config["kv_connector_extra_config"]
    assert extra["async_h2d_chunk_bytes"] == 8 * 1024**2
    assert extra["async_d2h_chunk_bytes"] == 1024**3
    assert extra["cpu_bytes_to_use"] == 2130300354560
    module.ROOT = module.driver.ROOT = collector.ROOT = root
    collector.BASE = "http://127.0.0.1:18462"
    collector.options = lambda enabled=False: copy.deepcopy(
        policies[CURRENT_ARM] if enabled else inactive
    )
    collector.server_command = lambda: copy.deepcopy((command, config))
    expected = len(paths) * len(args.workloads) * args.repeats
    collector.save(
        root / "plan.json",
        dict(
            model="GLM 5.3 NVFP4",
            parallelism="TP8",
            arms=list(paths),
            experiment_kind="load_only_policy",
            initial_barrier=args.initial_barrier,
            dma_audit=args.dma_audit,
            policies=policies,
            workloads=args.workloads,
            requests=args.requests,
            concurrency=32,
            output_tokens=1,
            repeats=args.repeats,
            seeds=[args.seed_start + i for i in range(args.repeats)],
            measured_cells=expected,
            command=command,
            connector=config,
            prefix_working_sets=PREFIX_WORKING_SETS,
            sources={
                name: dict(
                    path=path,
                    sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                )
                for name, path in paths.items()
            },
            scope=(
                "Identical frozen predictor and policies except deadline lead; "
                "fresh serving seeds and reversed arm order on odd repeats."
                if args.arm_lead_us
                else "Frozen models, fresh serving seeds, reversed arm order on odd "
                "repeats; identical hold, deadline lead, candidate and load-only "
                "policy. Legacy bias is an explicitly recorded target conversion; "
                "refits predict critical duration with zero external correction."
            ),
        ),
    )
    collector.save(
        root / "source_provenance.json",
        dict(
            source=str(SOURCE),
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
            ).strip(),
            branch=subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=SOURCE, text=True
            ).strip(),
        ),
    )
    if args.preflight_only:
        print("Frozen predictor and identical-policy preflight passed", flush=True)
        return
    shutil.copy2(
        "/mnt/localnvme/jeongyoon/glm53_dataset_coverage_20261007/"
        "libseal_execution_observer.so",
        root / "libseal_execution_observer.so",
    )
    proc = timer = None
    results = []
    started = time.monotonic()

    def cell(name, workload, seed, repeat, count):
        global CURRENT_ARM
        CURRENT_ARM = name
        result = live.cell(
            name,
            states[name],
            workload,
            seed,
            repeat,
            requests_count=count,
            initial_barrier=args.initial_barrier,
            require_preparation=policies[name]["late_admission_mode"] == 4,
            prefix_working_set=PREFIX_WORKING_SETS[workload] if count == 300 else None,
        )
        assert result["accepted"] and result["frozen_unchanged"], result
        return result

    try:
        proc = collector.launch()
        timer = threading.Timer(args.allocation_seconds, collector.stop_server, (proc,))
        timer.daemon = True
        timer.start()
        while time.monotonic() - started < 1800:
            assert proc.poll() is None, "Server exited during startup"
            try:
                collector.call("/health", timeout=2)
                break
            except (OSError, TimeoutError):
                time.sleep(2)
        else:
            raise TimeoutError("Serving startup timeout")
        collector.log("server_ready")
        for workload in args.workloads:
            for name in paths:
                cell(name, workload, args.seed_start - 10, "warmup", 32)
        for repeat in range(args.repeats):
            order = list(paths) if repeat % 2 == 0 else list(reversed(paths))
            for workload in args.workloads:
                for name in order:
                    assert time.monotonic() - started < args.allocation_seconds - 120
                    results.append(
                        cell(
                            name,
                            workload,
                            args.seed_start + repeat,
                            repeat,
                            args.requests,
                        )
                    )
                    collector.save(
                        root / "progress.json",
                        dict(
                            cells=len(results),
                            expected_cells=expected,
                            latest=dict(arm=name, workload=workload, repeat=repeat),
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
                stopped=proc is None or proc.poll() is not None,
                results=results,
                seconds=time.monotonic() - started,
            ),
        )


if __name__ == "__main__":
    main()
