# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare a selected learned allowance with fixed5 and fixed50 on fresh seeds."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from experiments.current_chunk_retrain_20261010 import serving

A_STATE = Path(
    "/mnt/localnvme/jeongyoon/current_chunk_retrain_20261010/"
    "envelope/envelope_boost.json"
)
MODELS = Path("/mnt/localnvme/jeongyoon/setup_prediction_20261011/frozen_models")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocation-seconds", type=int, required=True)
    parser.add_argument("--model", choices=("mean", "uncertainty"), required=True)
    parser.add_argument(
        "--workloads", nargs="+", default=["short16k", "mixed_supported"]
    )
    parser.add_argument("--seed-start", type=int, default=12311)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "reference_plan.json").exists()
    setup_path = MODELS / (args.model + ".json")
    state = json.loads(A_STATE.read_text())
    state["latency_surface"]["provenance"]["setup_predictor"] = json.loads(
        setup_path.read_text()
    )
    learned = "learned_mean" if args.model == "mean" else "learned_q90"
    path = args.output / (learned + ".json")
    path.write_text(json.dumps(state))
    (args.output / "reference_plan.json").write_text(
        json.dumps(
            dict(
                purpose="Isolate benefit beyond selecting a smaller constant lead; "
                "mixed_supported includes two context/suffix combinations "
                "absent from the homogeneous setup training episodes.",
                selected_model=args.model,
                sha256=hashlib.sha256(setup_path.read_bytes()).hexdigest(),
                fixed_leads_ms=[50, 5],
                A_weights_unchanged=True,
                other_policies_unchanged=True,
            ),
            indent=2,
        )
        + "\n"
    )
    stripped = copy.deepcopy(state)
    del stripped["latency_surface"]["provenance"]["setup_predictor"]
    assert stripped == json.loads(A_STATE.read_text())
    sys.argv = [
        "serving",
        "--output",
        str(args.output),
        "--allocation-seconds",
        str(args.allocation_seconds),
        "--seed-start",
        str(args.seed_start),
        "--repeats",
        str(args.repeats),
        "--setup-reference-leads",
        "--workloads",
        *args.workloads,
        "--arm",
        "fixed50=" + str(A_STATE),
        "--arm",
        "fixed5=" + str(A_STATE),
        "--arm",
        learned + "=" + str(path),
        "--arm-lead-us",
        "fixed50=50000",
        "--arm-lead-us",
        "fixed5=5000",
        "--arm-lead-us",
        learned + "=50000",
    ]
    if args.preflight_only:
        sys.argv.append("--preflight-only")
    serving.main()


if __name__ == "__main__":
    main()
