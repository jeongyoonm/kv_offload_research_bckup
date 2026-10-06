# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched serving cells with drained, frozen per-policy predictor snapshots."""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path(
    os.environ.get(
        "PREDICTOR_COMPARISON_ROOT",
        "/mnt/localnvme/glm53_predictor_overnight_20261006/live_comparison",
    )
)
os.environ["CACHED_PINNED_ARTIFACT_ROOT"] = str(ROOT)
os.environ["VLLM_DEFERRAL_NATIVE_OBSERVER"] = (
    "/mnt/localnvme/glm53_coalesced_notify_20261004/build/libseal_execution_observer.so"
)
spec = importlib.util.spec_from_file_location(
    "calibration_run", SOURCE / "experiments/seal_calibration_20261006/run.py"
)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)
runner = calibration.runner
ARTIFACT = Path("/mnt/localnvme/glm53_predictor_overnight_20261006")
MODEL_FILES = {
    "reference": Path(
        "/mnt/localnvme/glm53_seal_calibration_20261006/trained_model.json"
    ),
    "dispatch": ARTIFACT / "portable/dispatch_model.json",
    "causal": ARTIFACT / "causal_portable/causal_aligned_model.json",
    "two_stage": ARTIFACT / "two_stage/model.json",
}
MODELS = {
    name: json.loads(path.read_text())["state"] for name, path in MODEL_FILES.items()
}
reference = dict(runner.POLICIES["legacy5"], late_seal_freeze_estimator=True)
runner.POLICIES = {name: dict(reference) for name in MODEL_FILES}
runner.PRIME = dict(runner.PRIME, late_seal_freeze_estimator=True)
calibration.reuse_model = True
calibration.model_path = MODEL_FILES["reference"]
calibration.model_state = MODELS["reference"]
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "models.json").write_text(
    json.dumps({name: str(path) for name, path in MODEL_FILES.items()}, indent=2) + "\n"
)


def configure(directory, options, label):
    for name, candidate in runner.POLICIES.items():
        if options is candidate:
            calibration.loaded = False
            calibration.model_path = MODEL_FILES[name]
            calibration.model_state = MODELS[name]
            break
    return calibration.configure(directory, options, label)


runner.configure = configure


def cells(run, server_dir, workloads, counts, policies, repeats, seed=55):
    return calibration.prep.original_cells(
        run, server_dir, workloads, counts, policies, repeats, seed=87
    )


runner.run_cells = cells


if __name__ == "__main__":
    sys.path.insert(0, str(SOURCE / "experiments/coalesced_notify_20261004"))
    from environment_preflight import check

    with (ROOT / "process_log.jsonl").open("a") as stream:
        stream.write(
            json.dumps(dict(event="started", time=time.time(), argv=sys.argv)) + "\n"
        )
    check(SOURCE, ROOT, runner.glm.UUIDS)
    try:
        runner.main()
    except BaseException as error:
        with (ROOT / "process_log.jsonl").open("a") as stream:
            stream.write(
                json.dumps(dict(event="failed", time=time.time(), error=repr(error)))
                + "\n"
            )
        raise
    else:
        with (ROOT / "process_log.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event="completed", time=time.time())) + "\n")
