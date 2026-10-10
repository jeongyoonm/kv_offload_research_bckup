# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate setup predictor infrastructure from its deadline decisions."""

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
SETUP = Path(
    "/mnt/localnvme/jeongyoon/setup_prediction_20261011/frozen_models/mean.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocation-seconds", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=12321)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "shadow_plan.json").exists()
    base = json.loads(A_STATE.read_text())
    model = json.loads(SETUP.read_text())
    paths = dict(fixed50=A_STATE)
    for name, shadow in (("shadow_mean50", True), ("learned_mean", False)):
        state = copy.deepcopy(base)
        setup = copy.deepcopy(model)
        if shadow:
            setup["shadow_only"] = True
        state["latency_surface"]["provenance"]["setup_predictor"] = setup
        stripped = copy.deepcopy(state)
        del stripped["latency_surface"]["provenance"]["setup_predictor"]
        assert stripped == base
        path = args.output / (name + ".json")
        path.write_text(json.dumps(state))
        paths[name] = path
    (args.output / "shadow_plan.json").write_text(
        json.dumps(
            dict(
                purpose="Measure the same frozen setup model in shadow with "
                "fixed50 decisions; compare with absent setup and learned decisions.",
                setup_sha256=hashlib.sha256(SETUP.read_bytes()).hexdigest(),
                A_sha256=hashlib.sha256(A_STATE.read_bytes()).hexdigest(),
                A_weights_unchanged=True,
                worker_and_other_policies_unchanged=True,
                workloads=["short16k", "mid32k"],
                seeds=[args.seed_start + i for i in range(args.repeats)],
                motivation="Primary slow cells enter a burst-and-load-only "
                "regime; first-compute load coverage also differs. Isolate "
                "prediction infrastructure before attributing the entire gain "
                "to deadline accuracy.",
            ),
            indent=2,
        )
        + "\n"
    )
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
        "--lead-us",
        "50000",
        "--workloads",
        "short16k",
        "mid32k",
    ]
    for name, path in paths.items():
        sys.argv += ["--arm", name + "=" + str(path)]
    if args.preflight_only:
        sys.argv.append("--preflight-only")
    serving.main()


if __name__ == "__main__":
    main()
