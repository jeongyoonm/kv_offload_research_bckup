# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU physical layout and descriptor optimization with a frozen scheduler."""

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
ARMS = ("control",)
MEMORY_LAYOUT = "private"
PHYSICAL_OPTIMIZATION = False


def options(active=False):
    baseline.CURRENT_ARM = "both"
    result = baseline.options(active)
    result["store_reuse_gpu_fence"] = active
    result["store_reuse_allocator_preference"] = active
    for key in (
        "store_descriptor_coalescing",
        "load_descriptor_coalescing",
        "ascending_gpu_block_assignment",
        "load_descriptor_packet_alignment",
    ):
        result[key] = active and PHYSICAL_OPTIMIZATION
    return result


def server_command():
    command, config = baseline.server_command()
    command[command.index("--port") + 1] = "18440"
    extra = config["kv_connector_extra_config"]
    extra["cpu_cache_layout"] = MEMORY_LAYOUT
    command[command.index("--kv-transfer-config") + 1] = json.dumps(config)
    return command, config


def main():
    global CURRENT_ARM, MEMORY_LAYOUT, ARMS, PHYSICAL_OPTIMIZATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--requests", type=int, choices=(75, 300), default=300)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--layout", choices=("private", "private_tensor"), default="private"
    )
    parser.add_argument("--physical-optimization", action="store_true")
    args = parser.parse_args()
    MEMORY_LAYOUT = args.layout
    PHYSICAL_OPTIMIZATION = args.physical_optimization
    ARMS = ("gpu",) if PHYSICAL_OPTIMIZATION else ("control",)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "plan.json").exists(), "Use a fresh artifact root"
    prior = baseline.prior
    prior.module.ROOT = prior.module.driver.ROOT = collector.ROOT = root
    collector.BASE = "http://127.0.0.1:18440"
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
    command, config = server_command()
    expected = len(ARMS) * args.repeats
    collector.save(
        root / "plan.json",
        dict(
            model="GLM 5.3 NVFP4",
            parallelism="TP8",
            chunk_mib=8,
            depth=2,
            cpu_memory_layout=MEMORY_LAYOUT,
            physical_optimization=PHYSICAL_OPTIMIZATION,
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
            scope="Continuous 300-request episodes with 75 CPU prefixes and "
            "four distinct suffixes per prefix, one GPU reset before each "
            "episode and no intermediate reset or prime. The 75-request mode "
            "reproduces the earlier episode shape. Compare identical seeds "
            "across independent allocations. All modes use allocator avoidance "
            "and global GPU reuse fences. Physical optimization additionally "
            "enables ascending GPU assignment, H2D/D2H coalescing and packet "
            "alignment. Original model, selector, cache capacity, precision, "
            "frozen predictor and 8 MiB/depth-two H2D remain fixed. Post-response "
            "trace-drain time includes a draining RPC and file reads.",
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
        print("CPU tensor layout configuration validation passed", flush=True)
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
