# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit reusable frozen models and compare live calibration accuracy."""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))
from vllm.v1.kv_offload.seal_predictor import ComputeEstimator  # noqa: E402


def accuracy(errors):
    return dict(
        count=len(errors),
        bias_ms=statistics.mean(errors),
        mae_ms=statistics.mean(abs(e) for e in errors),
        stddev_ms=statistics.stdev(errors),
        p95_absolute_error_ms=float(np.quantile(np.abs(errors), 0.95)),
        underpredicted=sum(e < 0 for e in errors),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "deadline_analysis", SOURCE / "experiments/seal_deadline_20261006/analyze.py"
    )
    deadline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deadline)
    saved_argv = sys.argv
    sys.argv = [sys.argv[0], str(args.root), "--control", "legacy5"]
    try:
        deadline.main()
    finally:
        sys.argv = saved_argv
    report = json.loads((args.root / "serving_comparison.json").read_text())
    state = json.loads(args.model.read_text())["state"]
    digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    assert not (args.root / "canary/training").exists(), (
        "reused run retrained the estimator"
    )
    groups = defaultdict(list)
    all_errors, issued_errors = defaultdict(list), defaultdict(list)
    for cell in report["cells"]:
        directory = Path(cell["path"])
        setting = json.loads((directory / "policy_setting.json").read_text())
        reuse = json.loads((directory / "model_reuse.json").read_text())
        assert reuse["frozen_state_unchanged"] and reuse["state_sha256"] == digest
        assert (
            json.loads((directory / "estimator_export.json").read_text())["state"]
            == state
        )
        model = ComputeEstimator.from_state_dict(state, state["context"])
        mode = setting["late_seal_calibration"]
        model.calibration_mode = mode
        groups[mode].append(cell)
        events = [
            json.loads(s) for s in (directory / "engine.jsonl").read_text().splitlines()
        ]
        done = {
            e["step"]: e
            for e in events
            if e["event"] == "step_completion" and e.get("selected")
        }
        errors = {
            step: (model.predict(e) - max(r[6] for r in e["execution_ranks"])) / 1e6
            for step, e in done.items()
        }
        all_errors[mode].extend(errors.values())
        for r in json.loads((directory / "late.json").read_text())["records"]:
            if r["event"] == "seal_prediction":
                step = r["predecessor_step"]
                assert model.predict(done[step]) == r["predicted_duration_ns"]
                issued_errors[mode].append(errors[step])
    control = {r["cell"].split("_")[0]: r for r in groups["legacy"]}
    aggregate = []
    for mode, rows in groups.items():
        assert {r["cell"].split("_")[0] for r in rows} == set(control)
        assert len(rows) == 4
        ratios = [
            r["throughput"] / control[r["cell"].split("_")[0]]["throughput"]
            for r in rows
        ]
        logs = [math.log(r) for r in ratios]
        center = statistics.mean(logs)
        radius = 3.182446305284263 * statistics.stdev(logs) / math.sqrt(4)
        aggregate.append(
            dict(
                mode=mode,
                episodes=len(rows),
                requests=sum(r["requests"] for r in rows),
                throughput=sum(r["requests"] for r in rows)
                / sum(r["requests"] / r["throughput"] for r in rows),
                mean_ttft_ms=statistics.mean(r["mean_ttft_ms"] for r in rows),
                mean_cell_p95_ttft_ms=statistics.mean(r["p95_ttft_ms"] for r in rows),
                compute_rounds=sum(r["compute_rounds"] for r in rows),
                late_admissions=sum(r["late_admissions"] for r in rows),
                recovered_compute_opportunities=sum(
                    r["recovered_compute_opportunities"] for r in rows
                ),
                kv_ready_during_compute=sum(r["kv_ready_during_compute"] for r in rows),
                mean_gap_lower_ms=statistics.mean(r["gap_lower_ms"] for r in rows),
                mean_gap_upper_ms=statistics.mean(r["gap_upper_ms"] for r in rows),
                mean_non_load_late_enqueue_lower_ms=statistics.mean(
                    r["non_load_late_enqueue_lower_ms"] for r in rows
                ),
                pair_gains_pct=[100 * (r - 1) for r in ratios],
                paired_95pct_gain_bounds=[
                    100 * (math.exp(center - radius) - 1),
                    100 * (math.exp(center + radius) - 1),
                ],
                all_compute_prediction=accuracy(all_errors[mode]),
                issued_predictions=accuracy(issued_errors[mode]),
            )
        )
    output = dict(
        source_commit=report["source_commit"],
        model=str(args.model),
        state_sha256=digest,
        no_gpu_estimator_training=True,
        aggregate=aggregate,
        scope=(
            "Four paired 75-request episodes per mode; 300 requests per mode. "
            "CPU cache priming and server initialization are excluded. "
            "All modes reuse identical frozen coefficients and training histories. "
            "Intervals use Student t on four log ratios."
        ),
    )
    (args.root / "calibration_comparison.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
