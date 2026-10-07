# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test GLM's capped sparse attention and separate prefix/suffix indexing work."""

import argparse
import json
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import enqueue
import improve
import joblib
import numpy as np
import study
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

TOPK = 2048
EXTRA_NAMES = [
    "prefix_index_pairs",
    "causal_suffix_index_pairs",
    "sparse_attention_pairs",
]
NAMES = [
    "sparse_trend_only",
    "sparse_trend_boost",
    "sparse_trend_absolute",
    "sparse_mlp",
    "sparse_mlp_enqueue",
]
INDICES = [0, 1, 8, 23, 24, 25]
PROJECTED = list(range(23)) + [28, 29, 30]
BLENDS = {
    "gpu_enqueue_blend_fixed": "gpu_enqueue_features",
    "gpu_gap_blend_fixed": "gpu_enqueue_gap_only",
    "sparse_mlp_blend_fixed": "sparse_mlp",
    "sparse_mlp_enqueue_blend_fixed": "sparse_mlp_enqueue",
}


def add_blends(states, fixed):
    for name, source in BLENDS.items():
        if source not in states:
            continue
        states[name] = dict(
            models=[states[source]["models"][0], fixed],
            weights=[0.5, 0.5],
            gpu_enqueue_features=states[source].get("gpu_enqueue_features", False),
            sparse_geometry=states[source].get("sparse_geometry", False),
        )


def register_schemas():
    original = study.arch.surface.FEATURE_NAMES + study.arch.surface.FORECAST_NAMES
    for names in [
        original + EXTRA_NAMES,
        original + EXTRA_NAMES + study.arch.surface.BASE_PREDICTION_NAMES,
        original + enqueue.EXTRA_NAMES + EXTRA_NAMES,
    ]:
        if names not in study.arch.surface.FEATURE_SCHEMAS:
            study.arch.surface.FEATURE_SCHEMAS.append(names)


register_schemas()


def extra_features(rows):
    features = []
    for row in rows:
        prefix = suffix = sparse = 0
        for q, c in row["request_shapes"]:
            assert q > 0 and c >= 0
            prefix += q * c
            suffix += q * (q + 1) / 2
            n = min(q, max(0, TOPK - c))
            sparse += n * c + n * (n + 1) / 2 + (q - n) * TOPK
        features.append([prefix, suffix, sparse])
    return np.array(features, dtype=float)


def linear_features(rows):
    return np.column_stack(
        (study.arch.features(rows), np.zeros((len(rows), 12)), extra_features(rows))
    )


def fit_linear(rows, alpha):
    full = linear_features(rows)
    x, y, w = full[:, INDICES], study.arch.target(rows), study.arch.weights(rows)
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    scale[scale < 1e-12] = 1
    model = Ridge(alpha=alpha, positive=True, solver="lbfgs", tol=1e-8).fit(
        (x - mean) / scale,
        y,
        sample_weight=w,
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
    got = np.array([improve.FrozenTrend(dict(base=state)).predict_one(v) for v in full])
    np.testing.assert_allclose(got, np.maximum(1, model.predict((x - mean) / scale)))
    return state


def fit_variants(batches, forecasts, selected, seed, verify_rows=None):
    improve.configure()
    groups = np.array([r["episode"] for r in forecasts])
    _, validation = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=73,
        ).split(np.arange(len(forecasts)), groups=groups)
    )
    held = set(groups[validation])
    training = [r for r in batches if r["episode"] not in held]
    rows = [forecasts[i] for i in validation]
    trials = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        base = fit_linear(training, alpha)
        model = improve.FrozenTrend(dict(base=base))
        prediction = np.array([model.predict_one(v) for v in linear_features(rows)])
        trials.append(
            dict(
                alpha=alpha,
                macro_mae_ms=study.arch.objective(
                    study.arch.target(rows),
                    prediction,
                    rows,
                ),
            )
        )
    alpha = min(trials, key=lambda r: r["macro_mae_ms"])["alpha"]
    base = fit_linear(batches, alpha)
    oof = np.full(len(forecasts), np.nan)
    for _, test in GroupKFold(n_splits=3).split(
        np.arange(len(forecasts)),
        groups=groups,
    ):
        banned = set(groups[test])
        fitted = fit_linear([r for r in batches if r["episode"] not in banned], alpha)
        model = improve.FrozenTrend(dict(base=fitted))
        oof[test] = [
            model.predict_one(v)
            for v in linear_features(
                [forecasts[i] for i in test],
            )
        ]
    assert np.isfinite(oof).all()
    names = study.arch.surface.FEATURE_NAMES + study.arch.surface.FORECAST_NAMES
    x = np.column_stack(
        (study.arch.runtime_features(forecasts), extra_features(forecasts))
    )
    states = {
        "sparse_trend_only": dict(
            **improve.wrapped(dict(base=base, input_indices=PROJECTED)),
            sparse_geometry=True,
        )
    }
    config = selected.get("correction_tree_tree", study.arch.FIXED_CORRECTION)
    for name, loss in [
        ("sparse_trend_boost", "squared_error"),
        ("sparse_trend_absolute", "absolute_error"),
    ]:
        improve.configure(loss=loss)
        model, _ = improve.fit(
            config,
            np.column_stack((x, oof)),
            study.arch.target(forecasts) - oof,
            forecasts,
            seed,
        )
        state = dict(
            base=base,
            input_indices=PROJECTED,
            correction=study.arch.export_component(
                model, names + EXTRA_NAMES + study.arch.surface.BASE_PREDICTION_NAMES
            ),
        )
        states[name] = dict(**improve.wrapped(state), sparse_geometry=True)
    improve.configure()
    verified = forecasts[:200] if verify_rows is None else verify_rows
    audit = dict(alpha=alpha, alpha_trials=trials, base_oof_episode_overlap=0)
    for name, use_enqueue in [("sparse_mlp", False), ("sparse_mlp_enqueue", True)]:
        train_x, feature_names, indices = x, names + EXTRA_NAMES, PROJECTED
        if use_enqueue:
            train_x = np.column_stack(
                (
                    study.arch.runtime_features(forecasts),
                    enqueue.extra_features(forecasts),
                    extra_features(forecasts),
                )
            )
            feature_names = names + enqueue.EXTRA_NAMES + EXTRA_NAMES
            indices = list(range(31))
        model, _ = improve.fit(
            selected["direct_mlp"],
            train_x,
            study.arch.target(forecasts),
            forecasts,
            seed,
        )
        states[name] = dict(
            **improve.wrapped(
                dict(
                    direct=study.arch.export_component(model, feature_names),
                    input_indices=indices,
                )
            ),
            sparse_geometry=True,
            gpu_enqueue_features=use_enqueue,
        )
        vx = np.column_stack(
            (
                study.arch.runtime_features(verified),
                enqueue.extra_features(verified),
                extra_features(verified),
            )
        )
        expected = np.maximum(1, study.arch.predict(model, vx[:, indices]))
        actual = improve.predict_state(states[name], verified)
        error = float(np.max(np.abs(expected - actual)))
        assert error < 1e-4, (name, error)
        audit[name] = dict(export_equivalence_max_ms=error)
    return states, audit


def run_task(task):
    directory = study.ROOT / "sparse" / task["name"]
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
    selected = json.loads(
        (study.ROOT / "folds" / task["name"] / "tuning.json").read_text()
    )["selected"]
    with threadpool_limits(limits=1):
        states, audit = fit_variants(batches, train, selected, task["seed"], test)
        predictions = {n: improve.predict_state(s, test) for n, s in states.items()}
    parent = study.ROOT / "improvements" / task["name"]
    with np.load(parent / "predictions.npz") as f:
        arrays = {k: f[k] for k in f.files}
    arrays.update(predictions)
    temporary = directory / "merged.npz"
    existing = json.loads((parent / "checkpoints.json").read_text())
    fixed = json.loads(
        (study.ROOT / "folds" / task["name"] / "checkpoints.json").read_text()
    )["trees_fixed"]
    add_blends(states, fixed)
    for name, source in BLENDS.items():
        if source in predictions:
            arrays[name] = (predictions[source] + arrays["trees_fixed"]) / 2
    np.savez_compressed(temporary, **arrays)
    temporary.replace(parent / "predictions.npz")
    existing.update(states)
    study.arch.dump(parent / "checkpoints.json", existing)
    study.arch.dump(directory / "checkpoints.json", states)
    result = dict(
        task=task["name"],
        seconds=time.perf_counter() - started,
        audit=audit,
        test_training_episode_overlap=0,
        features_from_geometry_only=True,
    )
    study.arch.dump(complete, result)
    print(json.dumps(dict(event="sparse_task_complete", **result)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = json.loads((study.ROOT / "tasks.json").read_text())
    proposal = json.loads((study.ROOT / "improvement_plan.json").read_text())
    available_blends = []
    for task in tasks:
        parent = study.ROOT / "improvements" / task["name"]
        states = json.loads((parent / "checkpoints.json").read_text())
        with np.load(parent / "predictions.npz") as data:
            arrays = {k: data[k] for k in data.files}
        fixed = json.loads(
            (study.ROOT / "folds" / task["name"] / "checkpoints.json").read_text()
        )["trees_fixed"]
        add_blends(states, fixed)
        for name, source in BLENDS.items():
            if source in arrays:
                arrays[name] = (arrays[source] + arrays["trees_fixed"]) / 2
                available_blends.append(name)
        np.savez_compressed(parent / "predictions.npz", **arrays)
        study.arch.dump(parent / "checkpoints.json", states)
    proposal["candidates"] = list(
        dict.fromkeys(
            proposal["candidates"] + available_blends,
        )
    )
    proposal["reasons"]["feature_blends"] = (
        "The ordinary 50/50 blend led completed CV; test the same fixed weight "
        "with new features, without selecting weights on held-out labels."
    )
    study.arch.dump(study.ROOT / "improvement_plan.json", proposal)
    subprocess.run(
        [
            sys.executable,
            str(study.SOURCE / "experiments/predictor_expanded_20261007/improve.py"),
            "--summarize-only",
        ],
        check=True,
    )
    selection = json.loads((study.ROOT / "final_selection.json").read_text())
    summary = json.loads((study.ROOT / "improvement_summary.json").read_text())
    chosen = selection["improvement"]
    cv = summary["episode:wide"]["scores"]["fresh"][chosen]["mae_ms"]
    ranges = {
        s: summary[f"{s}:wide"]["scores"]["fresh"][chosen]["mae_ms"]
        for s in ["long_context", "long_suffix", "large_batch"]
    }
    triggered = max(ranges.values()) > max(10, 2 * cv)
    study.arch.dump(
        study.ROOT / "sparse_plan.json",
        dict(
            triggered=triggered,
            reason="remaining geometry-range error exceeds 10ms and twice fresh CV MAE",
            fresh_cv_mae_ms=cv,
            range_mae_ms=ranges,
            benchmark_used=False,
            topk=TOPK,
            features=EXTRA_NAMES,
            model_pair="GLM 5.3 TP8",
            serving_integration_required=True,
            sources=[
                "model config index_topk=2048",
                "vllm/model_executor/layers/sparse_attn_indexer.py",
                "vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py",
            ],
        ),
    )
    if not triggered:
        return
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        for future in as_completed([executor.submit(run_task, t) for t in tasks]):
            future.result()
    proposal = json.loads((study.ROOT / "improvement_plan.json").read_text())
    proposal["candidates"] = list(
        dict.fromkeys(
            proposal["candidates"]
            + NAMES
            + [n for n in BLENDS if n.startswith("sparse_")],
        )
    )
    proposal["reasons"]["sparse_geometry"] = dict(
        fresh_cv_mae_ms=cv, range_mae_ms=ranges
    )
    study.arch.dump(study.ROOT / "improvement_plan.json", proposal)
    subprocess.run(
        [
            sys.executable,
            str(study.SOURCE / "experiments/predictor_expanded_20261007/improve.py"),
            "--summarize-only",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
