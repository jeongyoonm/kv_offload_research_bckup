# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Baseline/worker/full measurements with fixed budgets and audited cache outcomes."""

import argparse
import copy
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from experiments.load_only_overlap_20261011 import cache_client as client
from experiments.load_only_overlap_20261011 import workloads
from experiments.native_scheduler_ab_20261010 import run as base
from vllm.v1.kv_offload.early_readiness import (
    EXPERIMENT_DEFAULTS,
    validate_experiment_options,
)

SOURCE = Path(__file__).resolve().parents[2]
STATE_PATH = Path(
    "/mnt/localnvme/jeongyoon/setup_prediction_20261011/runtime/frozen_A.json"
)
DATA = Path("/mnt/localnvme/jeongyoon/load_only_overlap_20261011/datasets")
CURRENT_ARM = "worker"


def policy(arm, idle_target):
    base.CURRENT_POLICY = "D"
    result = validate_experiment_options(base.options(True))
    result.update(
        late_seal_prediction_bias_us=0,
        late_seal_compute_deadline_lead_us=50000,
        h2d_block_readiness=True,
        late_seal_load_only_consistent=True,
        late_seal_load_only_idle_target=idle_target,
    )
    if arm != "full":
        for key, value in EXPERIMENT_DEFAULTS.items():
            if key.startswith("late_"):
                result[key] = value
        result.update(late_admission_markers=True, late_seal_freeze_estimator=True)
    return validate_experiment_options(result)


def launch(root, kind, port):
    command, config = base.server_command()
    command[command.index("--port") + 1] = str(port)
    command[command.index("--max-model-len") + 1] = "81920"
    if kind == "baseline":
        extra = config["kv_connector_extra_config"]
        extra.update(async_h2d_submission=False, async_d2h_submission=False)
        for name in list(extra):
            if name.startswith(("early_load_", "_early_load_")) or name in (
                "schedule_delay_ms",
                "coalesced_seal_notifications",
            ):
                del extra[name]
    command[command.index("--kv-transfer-config") + 1] = json.dumps(config)
    inherited = json.loads(
        Path("/mnt/localnvme/jeongyoon/lo11c/server/environment.json").read_text()
    )
    env = dict(os.environ, **inherited)
    env.update(
        PYTHONPATH=str(SOURCE),
        TMPDIR=str(root / "tmp"),
        VLLM_DEFERRAL_TIMING_DIR=str(root / "server/timing"),
        VLLM_DATASET_TARGET_GATE=str(root / "target_hold"),
        VLLM_DEFERRAL_NATIVE_OBSERVER=str(root / "libseal_execution_observer.so"),
        VLLM_KV_READINESS_TIMELINE_DIR=str(root / "server/load_timeline"),
        VLLM_DEFERRAL_LOAD_AUDIT="1",
    )
    if kind == "baseline":
        for key in list(env):
            if key.startswith(("VLLM_DEFERRAL_", "VLLM_PREDICTOR_", "VLLM_DATASET_")):
                del env[key]
        env["VLLM_COOPERATIVE_RESPONSE_PROGRESS"] = "0"
    env["PATH"] = (
        str(SOURCE / ".venv/bin") + ":/usr/local/cuda-13.0/bin:" + env.get("PATH", "")
    )
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    assert not occupied, occupied
    (root / "server").mkdir(exist_ok=True)
    (root / "tmp").mkdir(exist_ok=True)
    (root / "server/load_timeline").mkdir(exist_ok=True)
    base.collector.save(root / "server/command.json", command)
    base.collector.save(root / "server/config.json", config)
    base.collector.save(
        root / "server/environment.json",
        {
            key: value
            for key, value in env.items()
            if key.startswith(("VLLM_", "HF_", "CUDA_", "PYTHON", "FLASHINFER_"))
        },
    )
    with (root / "server/server.log").open("w") as log:
        proc = subprocess.Popen(
            command, cwd=SOURCE, env=env, stdout=log, stderr=log, start_new_session=True
        )
    base.collector.save(root / "server/process.json", dict(pid=proc.pid))
    return proc


def cell(root, spec, arm, seed, repeat, concurrency, count, state, kind):
    name = f"r{repeat}_{arm}_n{concurrency}"
    directory = root / "cells" / spec["id"] / name
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "result.json").exists()
    data_path = DATA / f"{spec['id']}_s{seed}_r{count}.json.gz"
    evolving = spec["family"] == "evolving"
    data = (
        workloads.evolving(data_path, count, seed, prefix_pool=spec["prefix_pool"])
        if evolving
        else workloads.build(data_path, spec, count, seed)
    )
    run = base.collector
    run.reset(True)
    run.configure(False)
    all_responses, windows, timeline, engine = [], [], [], []
    previous = client.metrics()
    if evolving:
        client.issue(data["warmup"], f"warm-{seed}", concurrency)
        # Flush observers at an episode boundary without resetting either cache.
        run.configure(True, state)
        client.flush_timeline(root)
        offsets = run.snapshots() if kind != "baseline" else {}
        before = client.metrics()
        responses, start, end = client.issue(
            data["requests"], f"main-{seed}", concurrency
        )
        all_responses = [(r, start) for r in responses]
        windows = [(start, end)]
        timeline = client.flush_timeline(root)
        engine = run.traces(offsets, directory) if kind != "baseline" else []
        after = client.metrics()
        expected = None
    else:
        # Independent prefixes; bound each primed cohort by actual CPU capacity.
        for offset in range(0, count, 64):
            cohort = data["requests"][offset : offset + 64]
            run.reset(True)
            run.configure(False)
            client.prime(cohort, f"prime-{seed}")
            run.reset(False)
            run.configure(True, state)
            client.flush_timeline(root)
            offsets = run.snapshots() if kind != "baseline" else {}
            responses, start, end = client.issue(cohort, f"main-{seed}", concurrency)
            all_responses.extend((r, start) for r in responses)
            windows.append((start, end))
            timeline.extend(client.flush_timeline(root))
            part = directory / f"cohort{offset}"
            part.mkdir()
            if kind != "baseline":
                engine.extend(run.traces(offsets, part))
        before, after = previous, client.metrics()
        expected = [r["context"] for r in data["requests"]]
    result = client.summarize(
        all_responses, windows, timeline, engine, expected=expected
    )
    result.update(
        arm=arm,
        workload=spec,
        seed=seed,
        repeat=repeat,
        concurrency=concurrency,
        dataset_sha256=workloads.digest(data_path),
        metrics_before=before,
        metrics_after=after,
        windows=windows,
        cache_reset_between_measurement_cohorts=not evolving,
        cold_prime_time_excluded=not evolving,
        cache_budgets=dict(cpu_bytes=2130300354560, gpu_bytes_per_rank=96 * 1024**3),
    )
    run.save(directory / "result.json", result)
    (directory / "load_timeline.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in timeline)
    )
    (directory / "engine.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in engine)
    )
    run.log(
        "cache_evaluation_cell",
        path=str(directory),
        accepted=result["accepted"],
        throughput_rps=result["throughput_rps"],
        outcomes=result["realized_outcomes"],
    )
    assert result["accepted"], directory
    return result


def main():
    global CURRENT_ARM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--server-kind", choices=["baseline", "optimized"], required=True
    )
    parser.add_argument(
        "--suite",
        choices=["grid", "selected", "evolving", "heterogeneous"],
        required=True,
    )
    parser.add_argument("--classes", type=Path)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[32])
    parser.add_argument("--arms", nargs="+", default=["worker", "full"])
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--seed-start", type=int, default=12501)
    parser.add_argument("--idle-target", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=10800)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "plan.json").exists()
    assert len(str(root / "tmp")) <= 45
    assert set(args.concurrencies) <= set(workloads.CONCURRENCIES)
    assert set(args.arms) <= {"worker", "full"}
    arms = ["baseline"] if args.server_kind == "baseline" else args.arms
    specs = workloads.homogeneous_grid()
    if args.suite == "selected":
        selected = {"c4096_q8192", "c16384_q2048", "c32768_q512", "c65536_q128"}
        specs = [spec for spec in specs if spec["id"] in selected]
    elif args.suite == "heterogeneous":
        assert args.classes is not None
        specs = workloads.heterogeneous(json.loads(args.classes.read_text()))
    elif args.suite == "evolving":
        specs = [
            dict(id=f"evolving_pool{size}", family="evolving", prefix_pool=size)
            for size in (192, 384)
        ]
    if args.ids:
        specs = [spec for spec in specs if spec["id"] in args.ids]
    assert specs
    run = base.collector
    run.ROOT = root
    run.BASE = "http://127.0.0.1:18468"
    policies = {
        arm: validate_experiment_options({})
        if arm == "baseline"
        else policy(arm, args.idle_target)
        for arm in arms
    }
    state = json.loads(STATE_PATH.read_text())
    run.options = lambda active=False: copy.deepcopy(policies[CURRENT_ARM])
    if args.server_kind == "baseline":
        run.configure = lambda *a, **k: dict(baseline=True)
    expected = len(specs) * len(arms) * len(args.concurrencies) * args.repeats
    run.save(
        root / "plan.json",
        dict(
            specs=specs,
            arms=arms,
            kind=args.server_kind,
            requests=args.requests,
            concurrencies=args.concurrencies,
            policies=policies,
            expected_cells=expected,
            repeats=args.repeats,
            seeds=[args.seed_start + i for i in range(args.repeats)],
            source=str(SOURCE),
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
            ).strip(),
            scope="Shared private cache initialization; "
            "baseline ordinary host H2D/D2H, "
            "no chunked async submission, early submission or late admission. "
            "Worker adds optimized I/O and allocator; "
            "full additionally adds seal/preparation. "
            "Independent CPU-hit cohorts are reset and primed outside measurement. "
            "Evolving cells retain both caches throughout warmup and measured stream.",
        ),
    )
    if args.preflight_only:
        return
    shutil.copy2(
        "/mnt/localnvme/jeongyoon/glm53_dataset_coverage_20261007/libseal_execution_observer.so",
        root / "libseal_execution_observer.so",
    )
    proc = timer = None
    results = []
    started = time.monotonic()
    try:
        proc = launch(root, args.server_kind, 18468)
        timer = threading.Timer(args.seconds, run.stop_server, (proc,))
        timer.daemon = True
        timer.start()
        while time.monotonic() - started < 1800:
            assert proc.poll() is None, "Server exited during startup"
            try:
                run.call("/health", timeout=2)
                break
            except (OSError, TimeoutError):
                time.sleep(2)
        else:
            raise TimeoutError("Startup timed out")
        for repeat in range(args.repeats):
            for spec in specs:
                for concurrency in args.concurrencies:
                    for arm in arms if repeat % 2 == 0 else list(reversed(arms)):
                        assert time.monotonic() - started < args.seconds - 120
                        CURRENT_ARM = arm
                        results.append(
                            cell(
                                root,
                                spec,
                                arm,
                                args.seed_start + repeat,
                                repeat,
                                concurrency,
                                args.requests,
                                state,
                                args.server_kind,
                            )
                        )
                        run.save(
                            root / "progress.json",
                            dict(
                                cells=len(results),
                                expected_cells=expected,
                                latest={
                                    key: results[-1][key]
                                    for key in (
                                        "arm",
                                        "workload",
                                        "concurrency",
                                        "throughput_rps",
                                    )
                                },
                            ),
                        )
    finally:
        if timer:
            timer.cancel()
        if proc:
            run.stop_server(proc)
        run.save(
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
