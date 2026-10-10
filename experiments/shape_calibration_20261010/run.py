# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched frozen batch calibration with unchanged optimized worker settings."""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

from experiments.native_scheduler_ab_20261010 import run as baseline
from vllm.v1.kv_offload.early_readiness import validate_experiment_options
from vllm.v1.kv_offload.seal_predictor import ComputeEstimator

SOURCE = Path(__file__).resolve().parents[2]
module, collector = baseline.module, baseline.collector
CURRENT_ARM = "control"
LEADS = {
    "control": 100_000,
    "shape100": 100_000,
    "shape50": 50_000,
    "guard50": 50_000,
    "basis100": 100_000,
    "basis50": 50_000,
    "margin75": 50_000,
    "margin90": 50_000,
    "productive70": 50_000,
    "productive80": 50_000,
    "direct50": 50_000,
}
PREFIX_WORKING_SETS = {
    "short16k": 300,
    "mid32k": 150,
    "familiar60k": 75,
    "short_long_suffix": 250,
    "long_short_suffix": 75,
    "unseen48k": 90,
    "mixed_supported": 100,
}


def options(active=False):
    baseline.CURRENT_POLICY = "D"
    result = baseline.options(active)
    if active:
        result["late_seal_compute_deadline_lead_us"] = LEADS[CURRENT_ARM]
    return result


def server_command():
    command, config = baseline.server_command()
    command[command.index("--port") + 1] = "18451"
    return command, config


def main():
    global CURRENT_ARM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--arm-fit", action="append", default=[], metavar="ARM=PATH")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--requests", type=int, choices=(75, 300), default=75)
    parser.add_argument(
        "--workloads", nargs="+", default=["short16k", "mid32k", "familiar60k"]
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=tuple(LEADS),
        default=["control", "shape100", "shape50"],
    )
    parser.add_argument("--seed-start", type=int, default=9501)
    parser.add_argument("--allocation-timeout", type=int, default=7200)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.repeats <= 8 and 900 <= args.allocation_timeout <= 10800
    assert set(args.workloads) <= set(module.driver.live.WORKLOAD_SHAPES)
    assert args.requests != 300 or set(args.workloads) <= set(PREFIX_WORKING_SETS)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "plan.json").exists(), "Use a fresh output root"
    module.ROOT = module.driver.ROOT = collector.ROOT = root
    collector.BASE = "http://127.0.0.1:18451"
    collector.options, collector.server_command = options, server_command
    assert module.driver.live.run is collector
    module.driver.MODEL_ROOT = Path(
        "/mnt/localnvme/jeongyoon/glm53_offset_predictor_20261008"
    )
    checkpoint = module.driver.MODEL_ROOT / "models/matched_offset.json"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    freeze = json.loads((module.driver.MODEL_ROOT / "freeze.json").read_text())
    assert digest == freeze["models"]["matched_offset"]
    base = baseline.native.baseline.prior.state_for_chunk(
        json.loads(checkpoint.read_text()), 8
    )
    fit_paths = {arm: args.fit for arm in args.arms if arm != "control"}
    for value in args.arm_fit:
        arm, path = value.split("=", 1)
        assert arm in fit_paths, "Override only selected calibrated arms"
        fit_paths[arm] = Path(path)
    calibrations = {
        arm: json.loads(path.read_text())["calibration"]
        for arm, path in fit_paths.items()
    }
    states, policies = {}, {}
    for arm in args.arms:
        CURRENT_ARM = arm
        states[arm] = copy.deepcopy(base)
        if arm != "control":
            states[arm]["latency_surface"]["checkpoint"]["duration_calibration"] = (
                calibrations[arm]
            )
            restored = copy.deepcopy(states[arm])
            restored["latency_surface"]["checkpoint"].pop("duration_calibration")
            assert restored == base
        estimator = ComputeEstimator.from_state_dict(states[arm], base["context"])
        assert estimator.state_dict(base["context"]) == states[arm]
        policies[arm] = options(True)
        validate_experiment_options(policies[arm])
        assert policies[arm]["store_reuse_allocator_preference"]
        assert policies[arm]["store_reuse_gpu_fence"]
        assert policies[arm]["late_seal_prediction_bias_us"] == 90000
        assert policies[arm]["copy_pipeline_depth"] == 2
    for arm in args.arms:
        changed = {
            k
            for k in policies[args.arms[0]]
            if policies[args.arms[0]][k] != policies[arm][k]
        }
        assert changed <= {"late_seal_compute_deadline_lead_us"}
    collector.save(root / "loaded_predictor.json", base)
    collector.save(root / "calibrations.json", calibrations)
    command, config = server_command()
    expected = len(args.arms) * len(args.workloads) * args.repeats
    collector.save(
        root / "plan.json",
        dict(
            model="GLM 5.3 NVFP4",
            parallelism="TP8",
            arms=args.arms,
            workloads=args.workloads,
            policies=policies,
            repeats=args.repeats,
            requests=args.requests,
            concurrency=32,
            output_tokens=1,
            prefix_working_sets=PREFIX_WORKING_SETS if args.requests == 300 else None,
            seeds=[args.seed_start + i for i in range(args.repeats)],
            checkpoint_sha256=digest,
            fit_sources={
                arm: dict(
                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
                for arm, path in fit_paths.items()
            },
            calibrations=calibrations,
            measured_cells=expected,
            command=command,
            connector=config,
            scope="Frozen shape correction replaces fixed correction within support; "
            "original 90ms fallback and all weights/history retained. Native stores, "
            "allocator avoidance, H2D8/depth2, preparation, readiness and load-only "
            "policy unchanged. Optional frozen shape margins alter only deadline "
            "lead. Optional productive-work policy adds a load-only early exit "
            "after all-rank base and selected candidate readiness. Calibration "
            "fitted before these independent seeds. "
            "75-request episodes are reset/primed; 300-request episodes have one "
            "initial reset and a workload-specific prefix working set with "
            "distinct suffixes. Working sets exceed GPU token capacity while "
            "prefixes plus new suffixes fit the CPU token budget.",
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
        print("Configuration and frozen checkpoint validation passed", flush=True)
        return
    shutil.copy2(
        "/mnt/localnvme/jeongyoon/glm53_dataset_coverage_20261007/"
        "libseal_execution_observer.so",
        root / "libseal_execution_observer.so",
    )
    results = []
    proc = timer = None
    started = time.monotonic()

    def cell(arm, workload, seed, repeat, count):
        global CURRENT_ARM
        CURRENT_ARM = arm
        result = module.driver.live.cell(
            arm,
            states[arm],
            workload,
            seed,
            repeat,
            requests_count=count,
            require_preparation=True,
            prefix_working_set=PREFIX_WORKING_SETS[workload] if count == 300 else None,
        )
        assert result["accepted"], result
        return result

    try:
        collector.log("comparison_started", expected_cells=expected)
        proc = collector.launch()
        collector.save(root / "ownership.json", dict(server_pid=proc.pid))
        timer = threading.Timer(args.allocation_timeout, collector.stop_server, (proc,))
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
        for workload in args.workloads:
            for arm in args.arms:
                cell(arm, workload, args.seed_start - 10, "warmup", 32)
        for repeat in range(args.repeats):
            order = args.arms if repeat % 2 == 0 else list(reversed(args.arms))
            for workload in args.workloads:
                for arm in order:
                    assert time.monotonic() - started < args.allocation_timeout - 120
                    results.append(
                        cell(
                            arm,
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
                            latest=dict(arm=arm, workload=workload, repeat=repeat),
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
