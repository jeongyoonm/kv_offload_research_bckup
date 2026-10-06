# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consolidate completed experiments without fitting or modifying checkpoints."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from report import PHASES

TOOLS = Path("/mnt/localnvme/vllm_glm53_predictor_corpus_lineage_20261006")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    models = [Path("/mnt/localnvme/glm53_seal_calibration_20261006/trained_model.json")]
    models.extend(
        root / name / "model.json"
        for name in [
            "expanded",
            "refreshed",
            "boosted_residual",
            "boosted_base",
            "realized_augmentation",
            "realized_augmentation/boosted",
        ]
    )
    models = [p for p in models if p.is_file()]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for phase in PHASES:
        directory = root / phase / "frozen_evaluation"
        if not (directory / "completed.json").is_file():
            continue
        output = directory / "final_models_accuracy.json"
        if output.is_file():
            continue
        with (directory / "final_models_accuracy.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "experiments/predictor_portable_20261006/score.py"),
                    "--corpus",
                    str(directory / "forecasts/forecasts.jsonl"),
                    "--models",
                    *map(str, models),
                    "--output",
                    str(output),
                ],
                check=True,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("report.py")),
            "--root",
            str(root),
            "--output",
            str(root / "report"),
        ],
        check=True,
        env=environment,
    )
    inventory = dict(
        study=json.loads((root / "study.json").read_text()),
        checkpoints=[str(p) for p in models],
        branches=json.loads((root / "branches.json").read_text()),
        decisions={
            str(p.relative_to(root)): json.loads(p.read_text())
            for p in root.glob("*/decision.json")
        },
        layer_outcomes=(
            json.loads((root / "online_live/analysis/outcomes.json").read_text())
            if (root / "online_live/analysis/outcomes.json").is_file()
            else None
        ),
    )
    (root / "report/experiment_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
