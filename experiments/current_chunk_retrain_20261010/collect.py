# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repeat controlled geometry/transfer cohorts with current worker settings."""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import threading
import time
import urllib.error
from collections import Counter
from pathlib import Path

from experiments.native_scheduler_ab_20261010 import run as baseline
from vllm.v1.kv_offload.early_readiness import (
    EXPERIMENT_DEFAULTS,
    validate_experiment_options,
)

collector = baseline.collector
SOURCE = Path(__file__).resolve().parents[2]


def prepare(root, seconds):
    baseline.CURRENT_POLICY = "D"
    policy = baseline.options(True)
    for key, value in EXPERIMENT_DEFAULTS.items():
        if key.startswith("late_"):
            policy[key] = value
    policy.update(late_admission_markers=True, late_seal_freeze_estimator=True)
    validate_experiment_options(policy)
    assert policy["late_admission_mode"] == 0
    assert policy["copy_pipeline_depth"] == 2
    assert policy["store_reuse_gpu_fence"]
    assert policy["store_reuse_allocator_preference"]
    command, config = baseline.server_command()
    command[command.index("--port") + 1] = "18461"
    extra = config["kv_connector_extra_config"]
    assert extra["cpu_cache_layout"] == "private"
    assert extra["cpu_bytes_to_use"] == 2130300354560
    assert extra["async_h2d_chunk_bytes"] == 8 * 1024**2
    assert extra["async_d2h_chunk_bytes"] == 1024**3
    assert extra["async_h2d_submission"] and extra["async_d2h_submission"]
    collector.ROOT = root
    collector.BASE = "http://127.0.0.1:18461"
    collector.options = lambda active=False: copy.deepcopy(policy)
    collector.server_command = lambda: copy.deepcopy((command, config))
    matrix = collector.build()
    checkpoint = Path("/mnt/localnvme/jeongyoon/yc5/loaded_predictor.json")
    state = json.loads(checkpoint.read_text())
    plan = dict(
        source=str(SOURCE),
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
        ).strip(),
        allocation_seconds=seconds,
        policy=policy,
        command=command,
        config=config,
        matrix_sha256=hashlib.sha256(
            json.dumps(matrix, sort_keys=True).encode()
        ).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        seeds=dict(
            pilot=11001,
            train=[11101, 11102, 11103],
            development=[11201, 11202],
            final=[11301, 11302],
        ),
        scope="Original controlled cohort matrix and transfer barriers; "
        "current worker, ordinary input path, no late admission or online fit. "
        "Training first-use repeat 0 is excluded during measurement audit. "
        "Fresh seeds; shorter budget than historical collection.",
    )
    collector.save(root / "plan.json", plan)
    collector.save(root / "matrix.json", matrix)
    return matrix, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=10800)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "plan.json").exists(), "Use a fresh output root"
    matrix, state = prepare(root, args.seconds)
    if args.preflight_only:
        print("Current-worker controlled collection preflight passed", flush=True)
        return
    shutil.copy2(
        "/mnt/localnvme/jeongyoon/glm53_dataset_coverage_20261007/"
        "libseal_execution_observer.so",
        root / "libseal_execution_observer.so",
    )
    started = time.monotonic()
    proc = timer = None
    results = []
    counts, positions = Counter(), Counter()

    def cell(family, seed, repeat, split):
        assert proc.poll() is None, "Server exited during collection"
        result = collector.cohort(family, seed, repeat, split)
        results.append(
            dict(
                path=result["path"],
                accepted=result["accepted"],
                valid_forecasts=result["valid_forecasts"],
            )
        )
        collector.save(
            root / "collection_progress.json",
            dict(
                counts=counts,
                positions=positions,
                cells=len(results),
                accepted=sum(r["accepted"] for r in results),
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        return result

    try:
        proc = collector.launch()
        timer = threading.Timer(args.seconds - 60, collector.stop_server, (proc,))
        timer.daemon = True
        timer.start()
        while time.monotonic() - started < 1800:
            assert proc.poll() is None, "Server exited during startup"
            try:
                collector.call("/health", timeout=2)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(2)
        else:
            raise RuntimeError("Startup timed out")
        collector.save(root / "server/policy.json", collector.configure(state=state))
        collector.log("server_ready", seconds=time.monotonic() - started)
        pilot = [
            matrix["train"][0],
            matrix["train"][6],
            next(f for f in matrix["train"] if f["block"] == "composition"),
            next(
                f
                for f in matrix["train"]
                if f["block"] == "transfer" and f["background_loads"] == 2
            ),
        ]
        for i, family in enumerate(pilot):
            result = cell(family, 11001, i, "pilot")
            assert result["accepted"], result["path"]
        collector.log("pilot_complete")
        blocks = ("geometry", "composition", "transfer", "familiar")
        families = {b: [f for f in matrix["train"] if f["block"] == b] for b in blocks}
        targets = dict(geometry=250, composition=250, transfer=250, familiar=750)
        while time.monotonic() - started < args.seconds * 0.66:
            available = [b for b in blocks if counts[b] < targets[b]]
            if not available:
                break
            block = min(available, key=lambda b: counts[b] / targets[b])
            i = positions[block]
            repeat = i // len(families[block])
            family = families[block][i % len(families[block])]
            result = cell(family, 11101 + repeat % 3, repeat, "train")
            if repeat > 0:
                counts[block] += result["valid_forecasts"]
            positions[block] += 1
        for split, seeds, limit in [
            ("development", [11201, 11202], 0.82),
            ("final", [11301, 11302], 0.95),
        ]:
            for seed in seeds:
                for i, family in enumerate(matrix["challenge"] + matrix["train"][::8]):
                    if time.monotonic() - started >= args.seconds * limit:
                        break
                    cell(family, seed, i, split)
        collector.log("collection_complete", counts=counts, cells=len(results))
    except BaseException as error:
        collector.log("failed", error=repr(error))
        raise
    finally:
        if timer is not None:
            timer.cancel()
        if proc is not None:
            collector.stop_server(proc)
        collector.save(
            root / "allocation.json",
            dict(
                total_seconds=time.monotonic() - started,
                server_pid=None if proc is None else proc.pid,
                server_exit_code=None if proc is None else proc.poll(),
                counts=counts,
                positions=positions,
                results=results,
            ),
        )


if __name__ == "__main__":
    main()
