# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native D2H submission with fixed allocator avoidance and scheduler policy."""

import argparse
import hashlib
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

from experiments.h2d8_deadline_bias_20261009 import run as baseline
from vllm.v1.kv_offload.early_readiness import validate_experiment_options

SOURCE = Path(__file__).resolve().parents[2]
collector = baseline.collector
CURRENT_ARM = "control"
ARMS = ("control", "gpu")
NATIVE_STORE = True
D2H_CHUNK_MIB = 1024
LEAD_US = 25_000


def options(active=False):
    baseline.CURRENT_ARM = "both"
    result = baseline.options(active)
    result["store_reuse_gpu_fence"] = active
    result["store_reuse_allocator_preference"] = active
    if active:
        result["late_seal_compute_deadline_lead_us"] = (
            LEAD_US if CURRENT_ARM == "gpu" else 100_000
        )
    return result


def server_command():
    command, config = baseline.server_command()
    command[command.index("--port") + 1] = "18449"
    extra = config["kv_connector_extra_config"]
    extra["async_d2h_submission"] = NATIVE_STORE
    extra["async_d2h_chunk_bytes"] = D2H_CHUNK_MIB * 1024 * 1024
    command[command.index("--kv-transfer-config") + 1] = json.dumps(config)
    return command, config


def main():
    global CURRENT_ARM, LEAD_US
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--requests", type=int, choices=(75, 300), default=300)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--lead-ms", type=int, choices=(25, 50), default=25)
    args = parser.parse_args()
    LEAD_US = args.lead_ms * 1000
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "plan.json").exists(), "Use a fresh artifact root"
    prior = baseline.prior
    prior.module.ROOT = prior.module.driver.ROOT = collector.ROOT = root
    collector.BASE = "http://127.0.0.1:18449"
    collector.options, collector.server_command = options, server_command
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
    assert {
        key
        for key in policies["control"]
        if policies["control"][key] != policies["gpu"][key]
    } == {"late_seal_compute_deadline_lead_us"}
    command, config = server_command()
    expected = len(ARMS) * args.repeats
    collector.save(
        root / "plan.json",
        dict(
            model="GLM 5.3 NVFP4",
            parallelism="TP8",
            chunk_mib=8,
            depth=2,
            native_store=NATIVE_STORE,
            d2h_chunk_mib=D2H_CHUNK_MIB,
            d2h_depth=2,
            candidate_deadline_lead_ms=args.lead_ms,
            prefix_working_set=75,
            policies=policies,
            repeats=args.repeats,
            requests=args.requests,
            concurrency=32,
            output_tokens=1,
            measured_cells=expected,
            seeds=[9301 + i for i in range(args.repeats)],
            checkpoint_file_sha256=digest,
            command=command,
            connector=config,
            scope="Paired continuous 300-request episodes with 75 CPU prefixes "
            "and distinct suffixes, exact CPU restoration and no GPU hits. "
            "Both arms use native full D2H submission, allocator avoidance, "
            "90 ms prediction subtraction, cached input preparation and the "
            "same frozen predictor. Only deadline setup lead differs: 100 ms "
            "control versus the selected 25 or 50 ms candidate. Startup and "
            "priming are excluded. No intermediate reset occurs inside episodes.",
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
        print("Native D2H configuration validation passed", flush=True)
        return
    shutil.copy2(
        "/mnt/localnvme/jeongyoon/glm53_dataset_coverage_20261007/libseal_execution_observer.so",
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
            prefix_working_set=75 if count > 75 else None,
        )
        assert result["accepted"], result
        return result

    try:
        collector.log("comparison_started", expected_cells=expected)
        proc = collector.launch()
        collector.save(root / "ownership.json", dict(server_pid=proc.pid))
        timer = threading.Timer(7200, collector.stop_server, (proc,))
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
            cell(arm, 9290, "warmup", 32)
        for repeat in range(args.repeats):
            order = ARMS if repeat % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                assert time.monotonic() - started < 7000, "Budget exhausted"
                results.append(cell(arm, 9301 + repeat, repeat, args.requests))
                collector.save(
                    root / "progress.json",
                    dict(
                        cells=len(results), expected_cells=expected, latest=results[-1]
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
