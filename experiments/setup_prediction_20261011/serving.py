# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare learned B setup leads with fixed50 while keeping A weights identical."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from experiments.current_chunk_retrain_20261010 import serving

ROOT = Path("/mnt/localnvme/jeongyoon/setup_prediction_20261011")
A_STATE = Path(
    "/mnt/localnvme/jeongyoon/current_chunk_retrain_20261010/"
    "envelope/envelope_boost.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocation-seconds", type=int, required=True)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["short16k", "mid32k", "unseen48k", "familiar60k"],
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=12301)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "setup_plan.json").exists()
    state = json.loads(A_STATE.read_text())
    assert "setup_predictor" not in state["latency_surface"]["provenance"]
    paths = dict(fixed50=A_STATE)
    for name, filename in (("learned_mean", "mean"), ("learned_q90", "uncertainty")):
        setup_state = json.loads(
            (ROOT / "frozen_models" / (filename + ".json")).read_text()
        )
        proposed = copy.deepcopy(state)
        proposed["latency_surface"]["provenance"]["setup_predictor"] = setup_state
        stripped = copy.deepcopy(proposed)
        del stripped["latency_surface"]["provenance"]["setup_predictor"]
        assert stripped == state
        paths[name] = args.output / (name + ".json")
        paths[name].write_text(json.dumps(proposed))
    (args.output / "setup_plan.json").write_text(
        json.dumps(
            dict(
                A_sha256=hashlib.sha256(A_STATE.read_bytes()).hexdigest(),
                A_unchanged=True,
                design="Deadline = predicted A finish - predicted B remaining setup. "
                "B proposal updates may revise an unexpired deadline; safety, "
                "candidate readiness and load-only early exits are unchanged. "
                "Fixed50 is retained on unavailable or unsupported setup inputs.",
                data=ROOT / "data/pool.json.gz",
                selection=ROOT / "frozen_models/selection.json",
                serving_seeds=[args.seed_start + i for i in range(args.repeats)],
            ),
            indent=2,
            default=str,
        )
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
        *args.workloads,
    ]
    for name, path in paths.items():
        sys.argv += ["--arm", name + "=" + str(path)]
    if args.preflight_only:
        sys.argv.append("--preflight-only")
    serving.main()


if __name__ == "__main__":
    main()
