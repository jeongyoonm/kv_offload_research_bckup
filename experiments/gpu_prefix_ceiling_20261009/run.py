# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capacity-fitting GPU-only prefix-cache ceiling and matched CPU controls."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from experiments.store_reuse_fence_20261009 import run as baseline
from vllm.v1.kv_offload.early_readiness import validate_experiment_options

SOURCE = Path(__file__).resolve().parents[2]
collector = baseline.collector
PORT = 18432
SHAPES = {"familiar60k": (58368, 3072), "short16k": (15552, 832)}


def options(active=False):
    baseline.CURRENT_ARM = "gpu"
    result = baseline.options(active)
    if active:
        result.update(POLICY_OVERRIDES)
    return result


POLICY_OVERRIDES = {}


def command_for(cache):
    command, config = baseline.server_command()
    command[command.index("--port") + 1] = str(PORT)
    if cache == "gpu":
        index = command.index("--kv-transfer-config")
        del command[index : index + 2]
        config = None
    return command, config


def launch_gpu(root):
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    assert not occupied, f"GPUs occupied: {occupied}"
    command, config = command_for("gpu")
    # The CPU experiment observer requires a connector. Ordinary GPU caching
    # uses the same model/worker settings without that experimental observer.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("VLLM_DEFERRAL_", "VLLM_PREDICTOR_", "VLLM_DATASET_"))
        and k != "VLLM_COOPERATIVE_RESPONSE_PROGRESS"
    }
    env.update(
        PYTHONPATH=str(SOURCE),
        HF_HOME="/mnt/localnvme/huggingface",
        HF_HUB_OFFLINE="1",
        VLLM_SERVER_DEV_MODE="1",
        PYTHONHASHSEED="0",
        CUDA_HOME="/usr/local/cuda-13.0",
        VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS="0",
        VLLM_CACHE_ROOT=(
            "/mnt/localnvme/cpu_cache_init_20261007/glm_canary_final/vllm_cache"
        ),
        FLASHINFER_WORKSPACE_BASE=(
            "/mnt/localnvme/cpu_cache_init_20261007/"
            "glm_canary_final/flashinfer_workspace"
        ),
        TMPDIR=str(root / "tmp"),
    )
    env["PATH"] = (
        str(SOURCE / ".venv/bin") + ":/usr/local/cuda-13.0/bin:" + env.get("PATH", "")
    )
    (root / "server").mkdir(exist_ok=True)
    (root / "tmp").mkdir(exist_ok=True)
    collector.save(root / "server/command.json", command)
    collector.save(root / "server/connector_config.json", config)
    collector.save(
        root / "server/environment.json",
        {
            k: v
            for k, v in env.items()
            if k.startswith(("VLLM_", "FLASHINFER_", "HF_", "CUDA_", "PYTHON"))
            and not k.endswith(("_TOKEN", "_KEY", "_SECRET", "_PASSWORD"))
        },
    )
    with (root / "server/server.log").open("w") as log:
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
            cwd=SOURCE,
        )
    collector.save(
        root / "server/process.json", dict(pid=proc.pid, started=time.time())
    )
    return proc


def gpu_cell(workload, seed, repeat, count, *, full_hits=True):
    label = "gpu_only" if full_hits else "gpu_capacity_limited"
    directory = collector.ROOT / "live" / workload / f"r{repeat}_{label}"
    directory.mkdir(parents=True, exist_ok=True)
    collector.reset(True)
    context, suffix = SHAPES[workload]
    requests, primes = [], []
    for i in range(count):
        prefix, tail = collector.prompt(context, suffix, seed * 100000 + i)
        requests.append(prefix + tail)
        primes.append(prefix + [31700] * 64)
    live = baseline.baseline.prior.module.driver.live
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for i in range(0, count, 32):
            futures = [
                pool.submit(collector.completion, p, f"gpu-prime-{seed}-{j}")
                for j, p in enumerate(primes[i : i + 32], start=i)
            ]
            for future in futures:
                future.result()
        before = collector.counters()
        start = time.monotonic_ns()
        futures = [
            pool.submit(live.stream, p, f"gpu-main-{seed}-{i}")
            for i, p in enumerate(requests)
        ]
        responses = [future.result() for future in futures]
        elapsed = (time.monotonic_ns() - start) / 1e9
    after = collector.counters()
    restored = [
        r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        for r in responses
    ]
    hits = after.get("vllm:prefix_cache_hits_total", 0) - before.get(
        "vllm:prefix_cache_hits_total", 0
    )
    expected_hits = (
        restored == [context] * count
        if full_hits
        else all(0 <= value <= context for value in restored)
    )
    accepted = (
        expected_hits
        and hits == sum(restored)
        and after.get("vllm:num_preemptions_total", 0)
        == before.get("vllm:num_preemptions_total", 0)
        and after.get("vllm:kv_offload_load_bytes_total", 0)
        == before.get("vllm:kv_offload_load_bytes_total", 0)
        and all(r["usage"]["completion_tokens"] == 1 for r in responses)
    )
    ttft = [r["ttft_ms"] for r in responses]
    result = dict(
        accepted=accepted,
        label=label,
        full_prefix_residency_required=full_hits,
        workload=workload,
        seed=seed,
        repeat=repeat,
        concurrency=32,
        requests=count,
        seconds=elapsed,
        throughput_rps=count / elapsed,
        mean_ttft_ms=statistics.mean(ttft),
        p50_ttft_ms=float(np.quantile(ttft, 0.5)),
        p95_ttft_ms=float(np.quantile(ttft, 0.95)),
        expected_cached=[context] * count if full_hits else None,
        observed_cached=restored,
        gpu_hit_tokens=hits,
        target_prefix_tokens=context * count,
        gpu_hit_target_fraction=hits / (context * count),
        gpu_hit_prompt_fraction=hits / ((context + suffix) * count),
        metrics_before=before,
        metrics_after=after,
        responses=responses,
    )
    collector.save(directory / "result.json", result)
    collector.log(
        "live_cell",
        label=label,
        workload=workload,
        repeat=repeat,
        accepted=accepted,
        throughput_rps=result["throughput_rps"],
    )
    assert accepted, f"GPU cache cell failed validation: {directory}"
    return result


def main():
    global POLICY_OVERRIDES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--policy-overrides", type=Path)
    parser.add_argument("--include-capacity-limited", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    assert not args.include_capacity_limited or args.cache == "gpu"
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "plan.json").exists(), "Use a fresh artifact root"
    if args.policy_overrides:
        POLICY_OVERRIDES = json.loads(args.policy_overrides.read_text())
    prior = baseline.baseline.prior
    prior.module.ROOT = prior.module.driver.ROOT = collector.ROOT = root
    collector.BASE = f"http://127.0.0.1:{PORT}"
    collector.options = options
    collector.server_command = lambda: command_for("cpu")
    validate_experiment_options(options(True))
    checkpoint = prior.module.driver.MODEL_ROOT / "models/matched_offset.json"
    freeze = json.loads((prior.module.driver.MODEL_ROOT / "freeze.json").read_text())
    assert (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        == freeze["models"]["matched_offset"]
    )
    state = prior.state_for_chunk(json.loads(checkpoint.read_text()), 8)
    collector.save(root / "loaded_predictor.json", state)
    cells = [("familiar60k", 32, 10), ("short16k", 75, 4)]
    expected_cells = 18 if args.include_capacity_limited else 14
    command, config = command_for(args.cache)
    collector.save(
        root / "plan.json",
        dict(
            model="GLM 5.3 NVFP4",
            parallelism="TP8",
            cache=args.cache,
            command=command,
            connector=config,
            policies=options(True),
            cells=cells,
            measured_cells=expected_cells,
            capacity_limited_60k_75_requests=args.include_capacity_limited,
            concurrency=32,
            output_tokens=1,
            scope="60K: ten separately primed 32-request capacity-fitting waves, "
            "320 measured requests. 16K: four separately primed 75-request episodes, "
            "300 measured requests. GPU-only retains primed GPU blocks; CPU controls "
            "reset GPU cache after priming. Every request must hit the expected "
            "prefix and generate one token; no preemption. Priming is excluded. "
            "GPU-only removes the connector, holds, and connector-dependent observers. "
            "Optional four 75-request 60K GPU-only episodes use seeds 9301–9304 "
            "and permit capacity-related prefix misses. They are a separate "
            "fixed-capacity reference, not an all-hits ceiling.",
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
        print("GPU-only ceiling configuration validation passed", flush=True)
        return
    if args.cache == "cpu":
        shutil.copy2(
            "/mnt/localnvme/jeongyoon/glm53_dataset_coverage_20261007/"
            "libseal_execution_observer.so",
            root / "libseal_execution_observer.so",
        )
    results = []
    proc = timer = None
    started = time.monotonic()

    def cell(workload, seed, repeat, count):
        if args.cache == "gpu":
            return gpu_cell(workload, seed, repeat, count)
        return prior.module.driver.live.cell(
            "cpu_only",
            state,
            workload,
            seed,
            repeat,
            requests_count=count,
            require_preparation=True,
        )

    try:
        collector.log(
            "comparison_started", cache=args.cache, expected_cells=expected_cells
        )
        proc = launch_gpu(root) if args.cache == "gpu" else collector.launch()
        collector.save(root / "ownership.json", dict(server_pid=proc.pid))
        timer = threading.Timer(7100, collector.stop_server, (proc,))
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
        for workload, _, _ in cells:
            cell(workload, 9390, "warmup", 32)
        for workload, count, repeats in cells:
            for repeat in range(repeats):
                assert time.monotonic() - started < 7000, "Budget exhausted"
                result = cell(workload, 9401 + repeat, repeat, count)
                assert result["accepted"], result
                results.append(result)
                collector.save(
                    root / "progress.json",
                    dict(
                        cells=len(results),
                        expected_cells=expected_cells,
                        latest=dict(workload=workload, repeat=repeat),
                    ),
                )
        if args.include_capacity_limited:
            for repeat in range(4):
                assert time.monotonic() - started < 7000, "Budget exhausted"
                result = gpu_cell(
                    "familiar60k", 9301 + repeat, repeat, 75, full_hits=False
                )
                results.append(result)
                collector.save(
                    root / "progress.json",
                    dict(
                        cells=len(results),
                        expected_cells=expected_cells,
                        latest=dict(
                            workload="familiar60k",
                            label="gpu_capacity_limited",
                            repeat=repeat,
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
                complete=len(results) == expected_cells,
                results=results,
                seconds=time.monotonic() - started,
                stopped=proc is None or proc.poll() is not None,
            ),
        )


if __name__ == "__main__":
    main()
