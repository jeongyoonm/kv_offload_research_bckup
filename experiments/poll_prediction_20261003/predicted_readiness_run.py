# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prediction schedules checking; readiness/completion authorizes the seal."""

import json
import time

import run as wrapper

runner = wrapper.run
runner.SOURCE = wrapper.SOURCE
runner.POLICIES["predicted_readiness_50ms"] = dict(
    runner.POLICIES["hybrid_50ms"],
    late_seal_prediction_to_readiness=True,
    late_seal_load_only_target_candidates=4,
)
original_cells = runner.run_cells


def cells(*args, **kwargs):
    kwargs["seed"] = 56
    return original_cells(*args, **kwargs)


def check(directory, policies, *args, **kwargs):
    kwargs["prefix_len"] = 11776
    return wrapper.original_check(directory, policies, *args, **kwargs)


runner.run_cells = cells
wrapper.correctness.check = check
if __name__ == "__main__":
    root = wrapper.ROOT
    (root / "predicted_readiness_policies.json").write_text(
        json.dumps(runner.POLICIES, indent=2) + "\n"
    )
    with (root / "process_log.jsonl").open("a") as log:
        log.write(
            json.dumps(
                dict(
                    event="predicted_readiness_driver_started",
                    time=time.time(),
                    source=str(wrapper.SOURCE),
                    seed=56,
                )
            )
            + "\n"
        )
    try:
        runner.main()
    except BaseException as error:
        with (root / "process_log.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    dict(
                        event="predicted_readiness_driver_failed",
                        time=time.time(),
                        error=repr(error),
                    )
                )
                + "\n"
            )
        raise
    else:
        with (root / "process_log.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    dict(
                        event="predicted_readiness_driver_completed",
                        time=time.time(),
                    )
                )
                + "\n"
            )
