# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare calibrations using one saved estimator, with no repeated training."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "deadline_run", SOURCE / "experiments/seal_deadline_20261006/run.py"
)
deadline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deadline)
prep, runner = deadline.prep, deadline.prep.runner
ROOT = prep.ROOT
reference = runner.POLICIES["deadline5"]
training_policy = runner.POLICIES["current"]
runner.POLICIES = {
    f"{mode}5": dict(reference, late_seal_calibration=mode)
    for mode in ("legacy", "raw", "shape")
}
runner.POLICIES["current"] = dict(training_policy)
original_configure, original_cell = runner.configure, runner.cell
model_state = None
loaded = False
reuse_model = False
model_path = None


def estimator_request(directory, options, action, state=None):
    request = {"action": action}
    if state is not None:
        request["state"] = state
    result = json.loads(
        runner.glm.call(
            "/experimental/kv-load-scheduling",
            dict(
                early_load_completion=True,
                experiment_options=options,
                seal_estimator=request,
            ),
        )
    )["seal_estimator"]
    (directory / f"estimator_{action}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def configure(directory, options, label):
    global loaded
    if reuse_model:
        options = dict(options, late_seal_freeze_estimator=True)
    result = original_configure(directory, options, label)
    if reuse_model and not loaded:
        estimator_request(directory, options, "load", model_state)
        assert estimator_request(directory, options, "export")["state"] == model_state
        loaded = True
    return result


runner.configure = configure


def cell(directory, workload, options, server_dir, manifest):
    result = original_cell(directory, workload, options, server_dir, manifest)
    if model_state is not None:
        exported = estimator_request(directory, runner.PRIME, "export")["state"]
        assert exported == model_state, "frozen estimator changed during serving"
        (directory / "model_reuse.json").write_text(
            json.dumps(
                dict(
                    trained_model=str(model_path),
                    state_sha256=hashlib.sha256(
                        json.dumps(model_state, sort_keys=True).encode()
                    ).hexdigest(),
                    frozen_state_unchanged=True,
                ),
                indent=2,
            )
            + "\n"
        )
    return result


runner.cell = cell


def cells(run, server_dir, workloads, counts, policies, repeats, seed=55):
    global model_state
    if not reuse_model:
        prep.original_cells(
            ROOT / "training", server_dir, workloads, counts, ["current"], 1, seed=61
        )
        runner.PRIME = dict(runner.PRIME, late_seal_freeze_estimator=True)
        original_configure(run, runner.PRIME, "freeze_setting")
        model_state = estimator_request(run, runner.PRIME, "export")["state"]
        payload = dict(
            state=model_state,
            provenance=dict(training_source=str(ROOT), live_export=True),
        )
        temporary = model_path.with_suffix(model_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(model_path)
    else:
        assert loaded
    runner.PRIME = dict(runner.PRIME, late_seal_freeze_estimator=True)
    runner.POLICIES = {
        name: dict(options, late_seal_freeze_estimator=True)
        for name, options in runner.POLICIES.items()
    }
    (ROOT / "policies.json").write_text(json.dumps(runner.POLICIES, indent=2) + "\n")
    return prep.original_cells(
        run, server_dir, workloads, counts, policies, repeats, seed=62
    )


runner.run_cells = cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trained-model", type=Path)
    group.add_argument("--save-trained-model", type=Path)
    args, remaining = parser.parse_known_args()
    if not any(
        arg == "--policies" or arg.startswith("--policies=") for arg in remaining
    ):
        remaining += ["--policies", "legacy5,shape5"]
    reuse_model = args.trained_model is not None
    model_path = args.trained_model or args.save_trained_model
    if reuse_model:
        model_state = json.loads(model_path.read_text())["state"]
        runner.PRIME = dict(runner.PRIME, late_seal_freeze_estimator=True)
        runner.POLICIES = {
            name: dict(options, late_seal_freeze_estimator=True)
            for name, options in runner.POLICIES.items()
        }
    elif model_path.exists():
        raise RuntimeError(
            "saved model already exists; use --trained-model to reuse it"
        )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_DEFERRAL_NATIVE_OBSERVER"] = (
        "/mnt/localnvme/glm53_coalesced_notify_20261004/build/libseal_execution_observer.so"
    )
    with (ROOT / "process_log.jsonl").open("a") as log:
        log.write(
            json.dumps(
                dict(
                    event="started",
                    time=time.time(),
                    argv=sys.argv,
                    reused_model=reuse_model,
                    model=str(model_path),
                )
            )
            + "\n"
        )
    sys.argv = [sys.argv[0], *remaining]
    try:
        if any((ROOT / "gpu").glob("p*/*/result.json")):
            raise RuntimeError("existing serving cells require a new artifact root")
        sys.path.insert(0, str(SOURCE / "experiments/coalesced_notify_20261004"))
        from environment_preflight import check

        check(SOURCE, ROOT, runner.glm.UUIDS)
        runner.main()
    except BaseException as error:
        with (ROOT / "process_log.jsonl").open("a") as log:
            log.write(
                json.dumps(dict(event="failed", time=time.time(), error=repr(error)))
                + "\n"
            )
        raise
    else:
        with (ROOT / "process_log.jsonl").open("a") as log:
            log.write(json.dumps(dict(event="completed", time=time.time())) + "\n")
