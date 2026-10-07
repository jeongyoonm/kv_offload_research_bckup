# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remove fresh training rows while holding each fold's capacity fixed."""

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import improve
import joblib
import numpy as np
import report
import study
from threadpoolctl import threadpool_limits


def run_task(task):
    directory = study.ROOT / "coverage_controls" / task["name"]
    completed = directory / "completed.json"
    if completed.exists():
        return json.loads(completed.read_text())
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    pool = joblib.load(study.ROOT / "pool.joblib")
    held = {pool["forecasts"][i]["episode"] for i in task["test"]}
    train = [
        pool["forecasts"][i]
        for i in task["train"]
        if pool["forecasts"][i]["_source"] != "fresh"
    ]
    test = [
        pool["forecasts"][i]
        for i in task["test"]
        if pool["forecasts"][i]["_source"] == "fresh"
    ]
    batches = [
        r
        for r in pool["batches"]
        if r["_source"] != "fresh" and r["episode"] not in held
    ]
    if task["regime"] == "matched":
        batches = train
    assert not held.intersection(r["episode"] for r in train + batches)
    assert not any(r["_source"] == "fresh" for r in train + batches)
    selected = json.loads(
        (study.ROOT / "folds" / task["name"] / "tuning.json").read_text()
    )["selected"]
    improve.configure()
    with threadpool_limits(limits=1):
        states = {
            name: improve.wrapped(
                improve.fit_one(name, selected, batches, train, task["seed"])
            )
            for name in study.arch.MODELS
        }
        predictions = {n: improve.predict_state(s, test) for n, s in states.items()}
    metadata = dict(
        y=study.arch.target(test),
        episode=np.array([r["episode"] for r in test]),
        workload=np.array([r["workload"] for r in test]),
    )
    np.savez_compressed(directory / "predictions.npz", **metadata, **predictions)
    study.arch.dump(directory / "checkpoints.json", states)
    result = dict(
        task=task["name"],
        seconds=time.perf_counter() - started,
        historical_training_forecasts=len(train),
        historical_training_batches=len(batches),
        fresh_test_forecasts=len(test),
        test_training_episode_overlap=0,
        configs_reused_from_expanded_training_only_inner_validation=True,
        fresh_labels_used_for_model_fitting=False,
    )
    study.arch.dump(completed, result)
    print(json.dumps(dict(event="coverage_control_complete", **result)), flush=True)
    return result


def main(workers=4):
    tasks = [
        t
        for t in json.loads((study.ROOT / "tasks.json").read_text())
        if t["range_scope"] is None
    ]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        for future in as_completed([executor.submit(run_task, t) for t in tasks]):
            future.result()
    summary = {}
    for regime in ["wide", "matched"]:
        members = [t for t in tasks if t["regime"] == regime]
        old, expanded = [], []
        for task in members:
            with np.load(
                study.ROOT / "coverage_controls" / task["name"] / "predictions.npz"
            ) as data:
                old.append({k: data[k] for k in data.files})
            with np.load(
                study.ROOT / "folds" / task["name"] / "predictions.npz"
            ) as data:
                mask = data["phase"] == "fresh"
                expanded.append({k: data[k][mask] for k in old[-1]})
            for key in ["y", "episode", "workload"]:
                np.testing.assert_array_equal(old[-1][key], expanded[-1][key])
        old = {k: np.concatenate([v[k] for v in old]) for k in old[0]}
        expanded = {k: np.concatenate([v[k] for v in expanded]) for k in expanded[0]}
        pairs = {}
        for name in study.arch.MODELS:
            comparison = dict(
                y=old["y"],
                episode=old["episode"],
                workload=old["workload"],
                trees_fixed=old[name],
                expanded=expanded[name],
            )
            pairs[name] = dict(
                sample=report.paired_interval(comparison, "expanded"),
                macro=report.paired_interval(comparison, "expanded", macro=True),
            )
        summary[regime] = dict(
            historical_only=report.scores(old),
            expanded=report.scores(expanded),
            paired_expanded_minus_historical=pairs,
        )
    study.arch.dump(
        study.ROOT / "coverage_control_summary.json",
        dict(
            scores=summary,
            benchmark_used=False,
            selection_changed=False,
            held_episodes_and_capacities_matched=True,
            caveat=(
                "Historical-only controls inherit capacities selected on expanded "
                "training folds. This isolates fitting-data coverage, rather than "
                "replaying an independently tuned historical deployment. Weights "
                "and training-only normalization are recomputed after removal."
            ),
        ),
    )


if __name__ == "__main__":
    main()
