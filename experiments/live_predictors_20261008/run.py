# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen predictor comparison on one GLM TP8 server and identical seal rules."""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_live_predictors_20261008")
COVERAGE = SOURCE / "experiments/dataset_coverage_20261007"
sys.path.insert(0, str(COVERAGE))
# The inherited helpers import their collector as 'run'. This driver runs as main.
spec = importlib.util.spec_from_file_location("run", COVERAGE / "run.py")
collector = importlib.util.module_from_spec(spec)
sys.modules["run"] = collector
spec.loader.exec_module(collector)
import live  # noqa: E402

collector.ROOT = ROOT
collector.BASE = "http://127.0.0.1:18408"
original_command = collector.server_command


def server_command():
    command, cfg = original_command()
    command[command.index("--port") + 1] = "18408"
    return command, cfg


collector.server_command = server_command
PATHS = {
    "trees": Path(
        "/mnt/localnvme/glm53_dataset_coverage_20261007/study/models/combined/seed73/serving.json"
    ),
    "guarded_sparse_mlp": Path(
        "/mnt/localnvme/glm53_predictor_expanded_20261007/exports/improvement/guarded_sparse_mlp/seed73.json"
    ),
    "envelope_sparse_mlp": Path(
        "/mnt/localnvme/glm53_envelope_blend_20261007/exports/envelope_sparse_mlp/seed73.json"
    ),
}


def models():
    tree = json.loads(PATHS["trees"].read_text())["state"]
    result = {"trees": tree}
    for name in list(PATHS)[1:]:
        state = copy.deepcopy(tree)
        state.pop("refinement_surface", None)
        state["schema"] = 2
        state["latency_surface"] = dict(
            schema=4,
            architecture=name,
            checkpoint=json.loads(PATHS[name].read_text()),
            target_schema="critical_end_after_all_rank_start.v1",
            provenance=dict(
                general_prefill=True,
                seed=73,
                checkpoint_sha256=hashlib.sha256(PATHS[name].read_bytes()).hexdigest(),
            ),
        )
        result[name] = state
    for name, state in result.items():
        collector.save(ROOT / "models" / f"{name}.json", state)
    return result


def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "allocation.json").exists():
        raise RuntimeError("Use a fresh artifact root for another server allocation")
    shutil.copy2(
        "/mnt/localnvme/glm53_dataset_coverage_20261007/libseal_execution_observer.so",
        ROOT / "libseal_execution_observer.so",
    )
    selected = models()
    workloads = ["familiar60k", "short16k", "heterogeneous"]
    collector.save(
        ROOT / "plan.json",
        dict(
            model="GLM 5.3 NVFP4",
            parallelism="TP8",
            concurrency=32,
            requests_per_episode=75,
            repeats=4,
            requests_per_predictor_workload=300,
            workloads=workloads,
            predictors=list(selected),
            policy=collector.options(True),
            comparison_scope=(
                "Predictor checkpoints differ in coverage as well as architecture; "
                "all other serving settings match."
            ),
            original_checkpoint_sha256={
                name: hashlib.sha256(p.read_bytes()).hexdigest()
                for name, p in PATHS.items()
            },
            maximum_allocation_seconds=14400,
        ),
    )
    proc = None
    timer = None
    started = time.monotonic()
    results = []
    try:
        collector.log("comparison_started")
        preflight = (
            SOURCE / "experiments/coalesced_notify_20261004/environment_preflight.py"
        )
        spec = importlib.util.spec_from_file_location("preflight", preflight)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import subprocess

        uuids = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).splitlines()
        module.check(SOURCE, ROOT, uuids)
        proc = collector.launch()
        timer = threading.Timer(14340, collector.stop_server, (proc,))
        timer.daemon = True
        timer.start()
        while time.monotonic() - started < 1800:
            if proc.poll() is not None:
                raise RuntimeError("Server exited during startup")
            try:
                collector.call("/health", timeout=2)
                break
            except (OSError, TimeoutError):
                time.sleep(2)
        else:
            raise RuntimeError("Startup timeout")
        collector.log("server_ready")
        # Exclude a complete shape warmup for each policy and workload.
        for workload in workloads:
            for name, state in selected.items():
                live.cell(name, state, workload, 901, "warmup", requests_count=32)
        names = list(selected)
        for repeat in range(4):
            for workload in workloads:
                order = names[repeat % 3 :] + names[: repeat % 3]
                if repeat == 3:
                    order = list(reversed(names))
                for name in order:
                    if time.monotonic() - started > 13800:
                        raise RuntimeError(
                            "Budget exhausted before completing comparison"
                        )
                    results.append(
                        live.cell(name, selected[name], workload, 1201 + repeat, repeat)
                    )
                    collector.save(
                        ROOT / "progress.json",
                        dict(
                            cells=len(results),
                            expected_cells=36,
                            latest=dict(name=name, workload=workload, repeat=repeat),
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
            ROOT / "summary.json", dict(complete=len(results) == 36, cells=results)
        )
        collector.log("comparison_stopped", cells=len(results))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
