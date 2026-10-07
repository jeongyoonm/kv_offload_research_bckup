# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use training-only geometry/load bounds to guard neural extrapolation."""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import improve
import joblib
import numpy as np
import study
from threadpoolctl import threadpool_limits

INDICES = [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12]
NAMES = ["guarded_sparse_mlp", "guarded_sparse_blend"]


def states_for(train, source, fixed):
    x = study.arch.runtime_features(train)[:, INDICES]
    bound = dict(
        indices=INDICES,
        minimum=x.min(0).tolist(),
        maximum=x.max(0).tolist(),
        fallback_model_index=1,
    )
    return {
        name: dict(
            models=[source["models"][0], fixed],
            weights=weights,
            support_guard=bound,
            sparse_geometry=True,
        )
        for name, weights in zip(NAMES, [[1.0, 0.0], [0.5, 0.5]])
    }


def run_task(task):
    directory = study.ROOT / "guards" / task["name"]
    complete = directory / "completed.json"
    if complete.exists():
        return json.loads(complete.read_text())
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    pool = joblib.load(task["dataset"])
    train = [pool["forecasts"][i] for i in task["train"]]
    test = [pool["forecasts"][i] for i in task["test"]]
    assert not {r["episode"] for r in train} & {r["episode"] for r in test}
    parent = study.ROOT / "improvements" / task["name"]
    checkpoints = json.loads((parent / "checkpoints.json").read_text())
    fixed = json.loads(
        (study.ROOT / "folds" / task["name"] / "checkpoints.json").read_text()
    )["trees_fixed"]
    states = states_for(train, checkpoints["sparse_mlp"], fixed)
    bound = states[NAMES[0]]["support_guard"]
    x = study.arch.runtime_features(test)[:, INDICES]
    outside = ((x < bound["minimum"]) | (x > bound["maximum"])).any(1)
    with np.load(parent / "predictions.npz") as f:
        arrays = {k: f[k] for k in f.files}
    errors = {}
    with threadpool_limits(limits=1):
        for name, source in zip(NAMES, ["sparse_mlp", "sparse_mlp_blend_fixed"]):
            arrays[name] = np.where(outside, arrays["trees_fixed"], arrays[source])
            # Check both routes, including late-index out-of-support examples.
            indices = sorted(
                set(range(min(100, len(test)))) | set(np.flatnonzero(outside)[:100])
            )
            actual = improve.predict_state(states[name], [test[i] for i in indices])
            error = float(np.max(np.abs(actual - arrays[name][indices])))
            assert error < 1e-4, (name, error)
            errors[name] = error
    temporary = directory / "merged.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(parent / "predictions.npz")
    checkpoints.update(states)
    study.arch.dump(parent / "checkpoints.json", checkpoints)
    study.arch.dump(directory / "checkpoints.json", states)
    result = dict(
        task=task["name"],
        seconds=time.perf_counter() - started,
        fallback_samples=int(outside.sum()),
        total_samples=len(test),
        export_equivalence_max_ms=errors,
        test_training_episode_overlap=0,
        bounds_fitted_without_test_rows=True,
    )
    study.arch.dump(complete, result)
    print(json.dumps(dict(event="guard_task_complete", **result)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = json.loads((study.ROOT / "tasks.json").read_text())
    study.arch.dump(
        study.ROOT / "guard_plan.json",
        dict(
            reason=(
                "Neural models led interpolation but trees were safer on "
                "all withheld ranges"
            ),
            feature_indices=INDICES,
            features=[
                (study.arch.surface.FEATURE_NAMES + study.arch.surface.FORECAST_NAMES)[
                    i
                ]
                for i in INDICES
            ],
            training_bounds_only=True,
            thresholds_tuned=False,
            fitting_required=False,
            benchmark_used=False,
            joint_support_guaranteed=False,
            serving_integration_required=True,
        ),
    )
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        for future in as_completed([executor.submit(run_task, t) for t in tasks]):
            future.result()
    proposal = json.loads((study.ROOT / "improvement_plan.json").read_text())
    proposal["candidates"] = list(dict.fromkeys(proposal["candidates"] + NAMES))
    proposal["reasons"]["support_guard"] = (
        "Use fixed trees when geometry/load footprint exceeds training bounds"
    )
    study.arch.dump(study.ROOT / "improvement_plan.json", proposal)
    improve.summarize(tasks, proposal)


if __name__ == "__main__":
    main()
