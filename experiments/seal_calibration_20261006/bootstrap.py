# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recover one frozen training model and evaluate calibration on later traces."""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.usage.usage_lib import UsageContext  # noqa: E402
from vllm.v1.kv_offload.seal_predictor import (  # noqa: E402
    ComputeEstimator,
    estimator_context,
)


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def context_from_launch(root):
    command = json.loads((root / "actual-launch.json").read_text())["command"]

    def option(name, convert=str):
        return convert(command[command.index("--" + name) + 1])

    config = EngineArgs(
        model=command[command.index("serve") + 1],
        tensor_parallel_size=option("tensor-parallel-size", int),
        kv_cache_dtype=option("kv-cache-dtype"),
        block_size=option("block-size", int),
        max_model_len=option("max-model-len", int),
        max_num_batched_tokens=option("max-num-batched-tokens", int),
        max_num_seqs=option("max-num-seqs", int),
        kv_cache_memory_bytes=option("kv-cache-memory-bytes", int),
        gpu_memory_utilization=option("gpu-memory-utilization", float),
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enable_flashinfer_autotune=False,
        async_scheduling=True,
        kv_transfer_config=json.loads(option("kv-transfer-config")),
    ).create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
    return estimator_context(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("previous", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.previous / "canary"
    training = root / "training/p61440_f95_h100/r0_current/engine.jsonl"
    cutoff = max(
        e["time_ns"] for e in events(training) if e["event"] == "step_completion"
    )
    model = ComputeEstimator()
    observed = 0
    for path in sorted((root / "gpu/server/timing").glob("engine-*.jsonl")):
        for event in events(path):
            if event["event"] == "step_completion" and event["time_ns"] <= cutoff:
                model.observe(event, event.get("execution_ranks"))
                model.observe_preparation(event.get("ranks"), event)
                observed += 1
    model.frozen = True
    snapshot = model.state_dict(context_from_launch(root))
    checked = 0
    summaries = []
    for directory in sorted((root / "gpu/p61440_f95_h100").glob("r*")):
        if not (directory / "result.json").exists():
            continue
        done = {
            e["step"]: e
            for e in events(directory / "engine.jsonl")
            if e["event"] == "step_completion" and e.get("selected")
        }
        model.calibration_mode = "legacy"
        for record in json.loads((directory / "late.json").read_text())["records"]:
            if record["event"] != "seal_prediction":
                continue
            actual = model.predict(done[record["predecessor_step"]])
            assert actual == record["predicted_duration_ns"], (actual, record)
            checked += 1
        for mode in ("legacy", "raw", "shape"):
            model.calibration_mode = mode
            errors = [
                (model.predict(e) - max(r[6] for r in e["execution_ranks"])) / 1e6
                for e in done.values()
            ]
            summaries.append(
                dict(
                    cell=directory.name,
                    mode=mode,
                    samples=len(errors),
                    bias_ms=statistics.mean(errors),
                    mae_ms=statistics.mean(abs(e) for e in errors),
                    p95_absolute_error_ms=float(np.quantile(np.abs(errors), 0.95)),
                    underpredicted=sum(e < 0 for e in errors),
                )
            )
    assert checked > 0
    args.output.mkdir(parents=True, exist_ok=True)
    payload = dict(
        state=snapshot,
        provenance=dict(
            training_source=str(args.previous),
            cutoff_ns=cutoff,
            training_trace_sha256=hashlib.sha256(training.read_bytes()).hexdigest(),
            observed_steps=observed,
            matched_frozen_predictions=checked,
            scope="Reconstructed from prior training, before frozen measured episodes.",
        ),
    )
    (args.output / "trained_model.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    (args.output / "offline_calibration.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    print(json.dumps(payload["provenance"], indent=2))
    for mode in ("legacy", "raw", "shape"):
        rows = [r for r in summaries if r["mode"] == mode]
        print(
            mode,
            {
                key: statistics.mean(r[key] for r in rows)
                for key in ("bias_ms", "mae_ms", "p95_absolute_error_ms")
            },
        )


if __name__ == "__main__":
    main()
