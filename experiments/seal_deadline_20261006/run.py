# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched GLM TP8 deadline leads atop compact pinned preparation."""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "compact_canary", SOURCE / "experiments/cached_pinned_prep_20261005/canary.py"
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
runner = prep.runner
compact = runner.POLICIES["compact_pinned"]
runner.POLICIES = {
    "compact_shadow": dict(compact, late_seal_compute_deadline_shadow=True),
    **{
        f"deadline{lead}": dict(
            compact,
            late_seal_compute_deadline=True,
            late_seal_compute_deadline_lead_us=lead * 1000,
        )
        for lead in (0, 5, 10, 20)
    },
    "current": runner.POLICIES["current"],
}


if __name__ == "__main__":
    os.environ["VLLM_DEFERRAL_NATIVE_OBSERVER"] = (
        "/mnt/localnvme/glm53_coalesced_notify_20261004/build/"
        "libseal_execution_observer.so"
    )
    root = prep.ROOT
    with (root / "process_log.jsonl").open("a") as log:
        log.write(
            json.dumps(dict(event="started", time=time.time(), argv=sys.argv)) + "\n"
        )
    try:
        if any((root / "gpu").glob("p*/*/result.json")):
            raise RuntimeError("Existing serving cells require a new artifact root")
        sys.path.insert(0, str(SOURCE / "experiments/coalesced_notify_20261004"))
        from environment_preflight import check

        check(SOURCE, root, runner.glm.UUIDS)
        runner.main()
    except BaseException as error:
        with (root / "process_log.jsonl").open("a") as log:
            log.write(
                json.dumps(dict(event="failed", time=time.time(), error=repr(error)))
                + "\n"
            )
        raise
    else:
        with (root / "process_log.jsonl").open("a") as log:
            log.write(json.dumps(dict(event="completed", time=time.time())) + "\n")
