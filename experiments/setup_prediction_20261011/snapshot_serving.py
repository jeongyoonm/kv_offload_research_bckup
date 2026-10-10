# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched comparison of original and physical seal-snapshot setup predictors."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from experiments.current_chunk_retrain_20261010 import serving

ROOT = Path("/mnt/localnvme/jeongyoon")
CAMPAIGN = ROOT / "setup_prediction_20261011"
A_STATE = ROOT / "current_chunk_retrain_20261010/envelope/envelope_boost.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocation-seconds", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=12331)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "snapshot_plan.json").exists()
    base = json.loads(A_STATE.read_text())
    paths, checksums = {}, {}
    for name, directory in (
        ("old_mean", "frozen_models"),
        ("snapshot_mean", "seal_snapshot_models"),
    ):
        source = CAMPAIGN / directory / "mean.json"
        state = copy.deepcopy(base)
        state["latency_surface"]["provenance"]["setup_predictor"] = json.loads(
            source.read_text()
        )
        stripped = copy.deepcopy(state)
        del stripped["latency_surface"]["provenance"]["setup_predictor"]
        assert stripped == base
        path = args.output / (name + ".json")
        path.write_text(json.dumps(state))
        paths[name] = path
        checksums[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (args.output / "snapshot_plan.json").write_text(
        json.dumps(
            dict(
                purpose="Original prospective training versus matching physical "
                "last-snapshot training; only frozen setup weights differ.",
                setup_sha256=checksums,
                A_sha256=hashlib.sha256(A_STATE.read_bytes()).hexdigest(),
                A_weights_unchanged=True,
                worker_and_other_policies_unchanged=True,
                workloads=["short16k", "mixed_supported"],
                seeds=[args.seed_start + i for i in range(args.repeats)],
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
        "mixed_supported",
    ]
    for name, path in paths.items():
        sys.argv += ["--arm", name + "=" + str(path)]
    if args.preflight_only:
        sys.argv.append("--preflight-only")
    serving.main()


if __name__ == "__main__":
    main()
