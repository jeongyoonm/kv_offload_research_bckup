# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched full predicted wait, margin, and fine polling experiment."""

import importlib.util
import json
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_full_predicted_wait_20261004")
sys.path.insert(0, str(SOURCE / "experiments/poll_prediction_20261003"))
spec = importlib.util.spec_from_file_location(
    "poll_wrapper", SOURCE / "experiments/poll_prediction_20261003/run.py"
)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
runner = wrapper.run
runner.SOURCE = SOURCE
runner.ROOT = ROOT
base = dict(
    runner.POLICIES["hybrid_50ms"],
    late_seal_prediction_to_readiness=True,
    late_seal_load_only_target_candidates=4,
    late_seal_wait_trace=True,
)
runner.POLICIES = {
    "capped50_fine2": base,
    "readiness2": dict(runner.POLICIES["readiness_2ms"], late_seal_wait_trace=True),
}
for margin in (50, 0):
    for fine in (2, 5, 10):
        runner.POLICIES[f"full_margin{margin}_fine{fine}"] = dict(
            base,
            late_seal_full_prediction_wait=True,
            late_seal_hybrid_guard_us=margin * 1000,
            late_seal_adaptive_poll_us=fine * 1000,
        )
original_cells = runner.run_cells


def cells(*args, **kwargs):
    kwargs["seed"] = kwargs.get("seed", 57)
    return original_cells(*args, **kwargs)


def check(directory, policies, *args, **kwargs):
    kwargs["prefix_len"] = 11776
    return wrapper.original_check(directory, policies, *args, **kwargs)


runner.run_cells = cells
wrapper.correctness.check = check


if __name__ == "__main__":
    (ROOT / "policies.json").write_text(json.dumps(runner.POLICIES, indent=2) + "\n")
    with (ROOT / "process_log.jsonl").open("a") as log:
        log.write(
            json.dumps(dict(event="driver_started", time=time.time(), argv=sys.argv))
            + "\n"
        )
    try:
        runner.main()
    except BaseException as error:
        with (ROOT / "process_log.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    dict(event="driver_failed", time=time.time(), error=repr(error))
                )
                + "\n"
            )
        raise
    else:
        with (ROOT / "process_log.jsonl").open("a") as log:
            log.write(
                json.dumps(dict(event="driver_completed", time=time.time())) + "\n"
            )
