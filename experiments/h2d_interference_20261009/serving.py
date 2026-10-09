# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small serving validation with the latest frozen predictor and seal policy."""

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path


def state_for_chunk(original, chunk_mib):
    """Retarget only transfer identity for this controlled frozen-weight study."""
    assert original["context"]["h2d_chunk_bytes"] == 128 * 1024**2
    state = copy.deepcopy(original)
    state["context"]["h2d_chunk_bytes"] = chunk_mib * 1024**2
    assert {k: v for k, v in state.items() if k != "context"} == {
        k: v for k, v in original.items() if k != "context"
    }
    assert {k: v for k, v in state["context"].items() if k != "h2d_chunk_bytes"} == {
        k: v for k, v in original["context"].items() if k != "h2d_chunk_bytes"
    }
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, required=True)
    parser.add_argument("--depth", type=int, choices=[1, 2], default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--requests", type=int, default=32)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "plan.json").exists(), "Use a fresh artifact root"
    spec = importlib.util.spec_from_file_location(
        "h2d_wait_driver", source / "experiments/efficient_wait_ab_20261009/run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    collector = module.collector
    module.ROOT = module.driver.ROOT = collector.ROOT = root
    collector.BASE = "http://127.0.0.1:18419"
    original_command = module.server_command
    original_options = module.options

    def server_command():
        command, config = original_command()
        command[0] = str(source / ".venv/bin/python")
        command[command.index("--port") + 1] = "18419"
        config["kv_connector_extra_config"]["async_h2d_chunk_bytes"] = (
            args.chunk_mib * 1024**2
        )
        command[command.index("--kv-transfer-config") + 1] = json.dumps(config)
        return command, config

    def options(active=False):
        result = original_options(active)
        result["copy_pipeline_depth"] = args.depth
        return result

    collector.server_command = server_command
    collector.options = options
    shutil.copy2(
        "/mnt/localnvme/glm53_dataset_coverage_20261007/libseal_execution_observer.so",
        root / "libseal_execution_observer.so",
    )
    checkpoint = module.driver.MODEL_ROOT / "models/matched_offset.json"
    freeze = json.loads((module.driver.MODEL_ROOT / "freeze.json").read_text())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == freeze["models"]["matched_offset"]
    original_state = json.loads(checkpoint.read_text())
    state = state_for_chunk(original_state, args.chunk_mib)
    collector.save(root / "loaded_predictor.json", state)
    collector.save(
        root / "predictor_compatibility.json",
        dict(
            original_file_sha256=digest,
            changed_context_fields={
                k: [original_state["context"][k], v]
                for k, v in state["context"].items()
                if original_state["context"][k] != v
            },
            learned_state_unchanged=True,
            note="Snapshot remains trained at 128 MiB; explicitly retarget only "
            "h2d_chunk_bytes identity for a frozen-weight configuration comparison. "
            "Production compatibility validation remains active; no retraining.",
        ),
    )
    workloads = ("short16k", "familiar60k", "heterogeneous")
    collector.save(
        root / "source_provenance.json",
        dict(
            source=str(source),
            branch=subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=source, text=True
            ).strip(),
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip(),
        ),
    )
    collector.save(
        root / "plan.json",
        dict(
            chunk_mib=args.chunk_mib,
            depth=args.depth,
            workloads=workloads,
            repeats=args.repeats,
            requests=args.requests,
            concurrency=32,
            checkpoint_sha256=digest,
            policy=options(True),
            scope="Small reset/primed serving episodes; separate allocation per chunk "
            "configuration, not a continuous 300-request throughput study.",
        ),
    )
    proc = timer = None
    results = []
    started = time.monotonic()
    try:
        proc = collector.launch()
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
        for workload in workloads:
            result = module.driver.live.cell(
                "full", state, workload, 7090, "warmup", requests_count=args.requests
            )
            assert result["accepted"], result
        for repeat in range(args.repeats):
            for workload in workloads:
                result = module.driver.live.cell(
                    "full",
                    state,
                    workload,
                    7101 + repeat,
                    repeat,
                    requests_count=args.requests,
                )
                assert result["accepted"], result
                results.append(result)
                collector.save(root / "progress.json", dict(cells=len(results)))
    finally:
        if timer:
            timer.cancel()
        if proc:
            collector.stop_server(proc)
        collector.save(
            root / "summary.json",
            dict(
                complete=len(results) == len(workloads) * args.repeats,
                results=results,
                seconds=time.monotonic() - started,
                stopped=proc is None or proc.poll() is not None,
            ),
        )


if __name__ == "__main__":
    main()
