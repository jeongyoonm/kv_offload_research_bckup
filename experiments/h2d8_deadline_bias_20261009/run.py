# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed prediction correction and B setup reserve, independently and together."""

import argparse
import hashlib
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

from experiments.h2d8_scheduler_ab_20261009 import run as prior
from vllm.v1.kv_offload.early_readiness import validate_experiment_options

SOURCE = Path(__file__).resolve().parents[2]
collector = prior.collector
ARMS = {
    "control": (0, 5000),
    "bias": (90000, 5000),
    "setup": (0, 100000),
    "both": (90000, 100000),
}
ORDERS = (
    ("control", "bias", "both", "setup"),
    ("bias", "setup", "control", "both"),
    ("setup", "both", "bias", "control"),
    ("both", "control", "setup", "bias"),
)
CURRENT_ARM = "control"


def options(active=False):
    result = prior.options(active)
    if active:
        bias, lead = ARMS[CURRENT_ARM]
        result.update(
            late_seal_prediction_bias_us=bias,
            late_seal_compute_deadline_lead_us=lead,
        )
    return result


def server_command():
    command, config = prior.server_command()
    command[command.index("--port") + 1] = "18423"
    return command, config


def main():
    global CURRENT_ARM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--requests", type=int, default=75)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "plan.json").exists(), "Use a fresh artifact root"
    prior.module.ROOT = prior.module.driver.ROOT = collector.ROOT = root
    collector.BASE = "http://127.0.0.1:18423"
    collector.options = options
    collector.server_command = server_command
    checkpoint = prior.module.driver.MODEL_ROOT / "models/matched_offset.json"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    freeze = json.loads((prior.module.driver.MODEL_ROOT / "freeze.json").read_text())
    assert digest == freeze["models"]["matched_offset"]
    state = prior.state_for_chunk(json.loads(checkpoint.read_text()), 8)
    collector.save(root / "loaded_predictor.json", state)
    policies = {}
    for arm in ARMS:
        CURRENT_ARM = arm
        policies[arm] = options(True)
        validate_experiment_options(policies[arm])
        assert policies[arm]["late_seal_compute_deadline"]
        assert policies[arm]["late_seal_freeze_estimator"]
        assert policies[arm]["copy_pipeline_depth"] == 2
        changed = {
            k for k in policies["control"] if policies["control"][k] != policies[arm][k]
        }
        assert changed <= {
            "late_seal_prediction_bias_us",
            "late_seal_compute_deadline_lead_us",
        }
    command, config = server_command()
    expected = len(ARMS) * args.repeats
    collector.save(
        root / "plan.json",
        dict(
            model="GLM 5.3 NVFP4",
            parallelism="TP8",
            chunk_mib=8,
            depth=2,
            policies=policies,
            workloads=["familiar60k"],
            repeats=args.repeats,
            requests=args.requests,
            concurrency=32,
            output_tokens=1,
            measured_cells=expected,
            orders=ORDERS,
            seeds=[9201 + i for i in range(args.repeats)],
            checkpoint_file_sha256=digest,
            command=command,
            connector=config,
            scope="Matched primed/reset 75-request episodes. Four episodes per arm "
            "total 300 measured requests, not one continuous 300-request run. "
            "Bias subtracts 90 ms from A's prediction; setup replaces the existing "
            "5 ms lead with 100 ms. No training, dynamic setup estimator, or fence "
            "changes. Constants chosen from prior allocation before this study.",
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
        print("Four-arm configuration validation passed", flush=True)
        return
    shutil.copy2(
        "/mnt/localnvme/glm53_dataset_coverage_20261007/libseal_execution_observer.so",
        root / "libseal_execution_observer.so",
    )
    results = []
    proc = timer = None
    started = time.monotonic()

    def cell(arm, seed, repeat, count):
        global CURRENT_ARM
        CURRENT_ARM = arm
        result = prior.module.driver.live.cell(
            arm,
            state,
            "familiar60k",
            seed,
            repeat,
            requests_count=count,
            require_preparation=True,
        )
        assert result["accepted"], result
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
        for arm in ARMS:
            cell(arm, 9190, "warmup", 32)
        for repeat in range(args.repeats):
            for arm in ORDERS[repeat % len(ORDERS)]:
                assert time.monotonic() - started < 10200, "Budget exhausted"
                results.append(cell(arm, 9201 + repeat, repeat, args.requests))
                collector.save(
                    root / "progress.json",
                    dict(
                        cells=len(results),
                        expected_cells=expected,
                        latest=dict(arm=arm, repeat=repeat),
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
