# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same frozen predictor and drained workload, changing only the CPU evaluator."""

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
ARTIFACT = Path("/mnt/localnvme/glm53_predictor_overnight_20261006")
ROOT = ARTIFACT / "fast_forest_serving"
LIBRARY = ARTIFACT / "fast_forest/build/libpredictor_forest_walk.so"
os.environ["PREDICTOR_COMPARISON_ROOT"] = str(ROOT)
os.environ["VLLM_DEFERRAL_FAST_FOREST"] = str(LIBRARY)
spec = importlib.util.spec_from_file_location(
    "comparison", SOURCE / "experiments/predictor_two_stage_20261006/run.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
runner = comparison.runner
model = ARTIFACT / "expanded/model.json"
comparison.MODEL_FILES = dict(numpy=model, native=model)
comparison.MODELS = {
    name: json.loads(path.read_text())["state"]
    for name, path in comparison.MODEL_FILES.items()
}
runner.POLICIES = {
    name: dict(comparison.reference, late_seal_native_predictor=name == "native")
    for name in comparison.MODEL_FILES
}
(ROOT / "models.json").write_text(
    json.dumps(
        {name: str(path) for name, path in comparison.MODEL_FILES.items()}, indent=2
    )
    + "\n"
)
(ROOT / "native_library.json").write_text(
    json.dumps(
        dict(
            path=str(LIBRARY), sha256=hashlib.sha256(LIBRARY.read_bytes()).hexdigest()
        ),
        indent=2,
    )
    + "\n"
)


def cells(run, server_dir, workloads, counts, policies, repeats, seed=55):
    return comparison.calibration.prep.original_cells(
        run, server_dir, workloads, counts, policies, repeats, seed=107
    )


runner.cells = cells


if __name__ == "__main__":
    sys.path.insert(0, str(SOURCE / "experiments/coalesced_notify_20261004"))
    from environment_preflight import check

    check(SOURCE, ROOT, runner.glm.UUIDS)
    with (ROOT / "process_log.jsonl").open("a") as stream:
        stream.write(
            json.dumps(dict(event="started", time=time.time(), argv=sys.argv)) + "\n"
        )
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
