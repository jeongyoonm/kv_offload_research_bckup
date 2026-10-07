# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conditionally test extrapolating geometry trends after range error evidence."""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import improve
import joblib
import numpy as np
import study
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

NAMES = ["trend_only", "trend_boost", "trend_absolute"]
INDICES = [0, 1, 2, 3, 8, 9, 10]


def fit_linear(rows, alpha):
    x = study.arch.features(rows)[:, INDICES]
    y, w = study.arch.target(rows), study.arch.weights(rows)
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    scale[scale < 1e-12] = 1
    model = Ridge(alpha=alpha, positive=True, solver="lbfgs", tol=1e-8).fit(
        (x - mean) / scale, y, sample_weight=w
    )
    state = dict(
        kind="linear",
        indices=INDICES,
        mean=mean.tolist(),
        scale=scale.tolist(),
        coefficients=model.coef_.tolist(),
        intercept=float(model.intercept_),
        alpha=alpha,
        positive_coefficients=True,
    )
    got = np.array(
        [
            improve.FrozenTrend(dict(base=state)).predict_one(v)
            for v in study.arch.features(rows)
        ]
    )
    np.testing.assert_allclose(got, np.maximum(1, model.predict((x - mean) / scale)))
    return state


def select_alpha(batches, forecasts):
    groups = np.array([r["episode"] for r in forecasts])
    _, validation = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=73).split(
            np.arange(len(forecasts)), groups=groups
        )
    )
    held = set(groups[validation])
    training = [r for r in batches if r["episode"] not in held]
    rows = [forecasts[i] for i in validation]
    trials = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        state = fit_linear(training, alpha)
        prediction = improve.predict_state(improve.wrapped(dict(base=state)), rows)
        trials.append(
            dict(
                alpha=alpha,
                macro_mae_ms=study.arch.objective(
                    study.arch.target(rows), prediction, rows
                ),
            )
        )
    return min(trials, key=lambda r: r["macro_mae_ms"])["alpha"], trials


def fit_variants(batches, forecasts, selected, seed):
    improve.configure()
    alpha, trials = select_alpha(batches, forecasts)
    base = fit_linear(batches, alpha)
    groups = np.array([r["episode"] for r in forecasts])
    prediction = np.full(len(forecasts), np.nan)
    for _, test in GroupKFold(n_splits=3).split(
        np.arange(len(forecasts)), groups=groups
    ):
        banned = set(groups[test])
        train = [r for r in batches if r["episode"] not in banned]
        state = fit_linear(train, alpha)
        prediction[test] = improve.predict_state(
            improve.wrapped(dict(base=state)), [forecasts[i] for i in test]
        )
    assert np.isfinite(prediction).all()
    # Keep residual capacity equal to the selected tree correction, if present.
    config = selected.get("correction_tree_tree", study.arch.FIXED_CORRECTION)
    states = {"trend_only": improve.wrapped(dict(base=base))}
    for name, loss in [
        ("trend_boost", "squared_error"),
        ("trend_absolute", "absolute_error"),
    ]:
        improve.configure(loss=loss)
        model, _ = improve.fit(
            config,
            np.column_stack((study.arch.runtime_features(forecasts), prediction)),
            study.arch.target(forecasts) - prediction,
            forecasts,
            seed,
        )
        correction = study.arch.export_component(
            model,
            study.arch.surface.FEATURE_NAMES
            + study.arch.surface.FORECAST_NAMES
            + study.arch.surface.BASE_PREDICTION_NAMES,
        )
        states[name] = improve.wrapped(dict(base=base, correction=correction))
    improve.configure()
    return states, dict(alpha=alpha, trials=trials, base_oof_episode_overlap=0)


def run_task(task):
    directory = study.ROOT / "trends" / task["name"]
    complete = directory / "completed.json"
    if complete.exists():
        return json.loads(complete.read_text())
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    pool = joblib.load(task["dataset"])
    train = [pool["forecasts"][i] for i in task["train"]]
    test = [pool["forecasts"][i] for i in task["test"]]
    held = {r["episode"] for r in test}
    batches = [r for r in pool["batches"] if r["episode"] not in held]
    if task["regime"] == "matched":
        batches = train
    assert not held.intersection(r["episode"] for r in train + batches)
    fold = study.ROOT / "folds" / task["name"]
    selected = json.loads((fold / "tuning.json").read_text())["selected"]
    with threadpool_limits(limits=1):
        states, audit = fit_variants(batches, train, selected, task["seed"])
        predictions = {n: improve.predict_state(s, test) for n, s in states.items()}
    improve_directory = study.ROOT / "improvements" / task["name"]
    with np.load(improve_directory / "predictions.npz") as f:
        arrays = {k: f[k] for k in f.files}
    arrays.update(predictions)
    temporary = directory / "merged.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(improve_directory / "predictions.npz")
    existing = json.loads((improve_directory / "checkpoints.json").read_text())
    existing.update(states)
    study.arch.dump(improve_directory / "checkpoints.json", existing)
    study.arch.dump(directory / "checkpoints.json", states)
    result = dict(task=task["name"], seconds=time.perf_counter() - started, audit=audit)
    study.arch.dump(complete, result)
    print(json.dumps(dict(event="trend_task_complete", **result)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selection = json.loads((study.ROOT / "final_selection.json").read_text())
    summary = json.loads((study.ROOT / "improvement_summary.json").read_text())
    name = selection["improvement"]
    cv = summary["episode:wide"]["scores"]["fresh"][name]["mae_ms"]
    ranges = {
        scope: summary[f"{scope}:wide"]["scores"]["fresh"][name]["mae_ms"]
        for scope in ["long_context", "long_suffix", "large_batch", "heavy_transfers"]
    }
    triggered = max(ranges.values()) > max(10, 2 * cv)
    study.arch.dump(
        study.ROOT / "trend_plan.json",
        dict(
            triggered=triggered,
            fresh_cv_mae_ms=cv,
            range_mae_ms=ranges,
            criterion="max fresh range MAE > max(10 ms, twice fresh episode-CV MAE)",
            benchmark_used_for_planning=False,
            method="positive ridge geometry trend plus bounded tree residual",
            added_serving_features=False,
            serving_integration_required=True,
        ),
    )
    if not triggered:
        return
    tasks = json.loads((study.ROOT / "tasks.json").read_text())
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
    proposal = json.loads((study.ROOT / "improvement_plan.json").read_text())
    proposal["candidates"] = list(dict.fromkeys(proposal["candidates"] + NAMES))
    proposal["reasons"]["continuous_trend"] = dict(
        fresh_cv_mae_ms=cv, range_mae_ms=ranges
    )
    study.arch.dump(study.ROOT / "improvement_plan.json", proposal)
    improve.summarize(tasks, proposal)


if __name__ == "__main__":
    main()
