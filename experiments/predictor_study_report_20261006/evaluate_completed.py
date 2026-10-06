# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate completed cells and pool distinct workloads for layer diagnostics."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SERVING = Path("/mnt/localnvme/vllm_glm53_predictor_serving_analysis_20261006")
LAYER = Path("/mnt/localnvme/vllm_glm53_predictor_layer_features_20261006")
ISSUED = Path("/mnt/localnvme/vllm_glm53_predictor_issued_audit_20261006")


def run(root, label, source, *args):
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    with (root / f"{label}.log").open("x") as log:
        subprocess.run(
            [sys.executable, str(source), *map(str, args)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def phase(root, name, count, control):
    destination = root / name
    run(
        root,
        f"{name}_frozen_report",
        SERVING / "experiments/predictor_two_stage_20261006/phase_report.py",
        destination,
        "--control",
        control,
        "--expected-cells",
        count,
        "--require-cpu-cache",
    )
    run(
        destination,
        "issued_accuracy",
        ISSUED / "experiments/predictor_issued_audit_20261006/issued.py",
        "--corpus",
        destination / "frozen_evaluation/forecasts/forecasts.jsonl",
        "--model",
        root / "boosted_residual/model.json",
        "--output",
        destination / "issued_accuracy.json",
    )


def dense(root):
    decision = root / "realized_augmentation/serving_decision.json"
    if not decision.is_file() or not json.loads(decision.read_text())["ran"]:
        return
    options = json.loads(decision.read_text())
    actual = root / "dense_realized/validation"
    (actual / "provenance.json").symlink_to(
        root / "dense_realized/canary/gpu/provenance.json"
    )
    alias = root / "dense_validation/canary"
    alias.mkdir(parents=True)
    (alias / "gpu").symlink_to(actual)
    phase(root, "dense_validation", 8 * options["repeats"], "control")


def online(root):
    phase(root, "online_live", 8, "boosted")
    included = [root / "online_live/canary/gpu"]
    additional = root / "online_live/canary/generalization/results.json"
    if additional.is_file():
        phase(root, "generalization", 6, "boosted")
        included.append(root / "generalization/canary/gpu")
    joint = root / "layer_joint/canary/gpu"
    joint.mkdir(parents=True)
    (joint / "provenance.json").symlink_to(included[0] / "provenance.json")
    count = 0
    for path in included:
        for cell in path.glob("p*/*/result.json"):
            if not json.loads(cell.read_text())["accepted"]:
                raise ValueError("joint diagnostics require accepted complete phases")
            directory = joint / cell.parent.parent.name
            directory.mkdir(exist_ok=True)
            (directory / cell.parent.name).symlink_to(cell.parent.resolve())
            count += 1
    run(
        root,
        "joint_layer_features",
        LAYER / "experiments/predictor_layer_submission_20261006/pipeline.py",
        "--root",
        root / "layer_joint",
        "--expected-cells",
        count,
        "--output",
        root / "layer_joint/analysis",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=["dense", "online"], required=True)
    args = parser.parse_args()
    {"dense": dense, "online": online}[args.phase](args.root)
