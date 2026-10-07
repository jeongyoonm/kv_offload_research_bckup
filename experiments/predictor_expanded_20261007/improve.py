# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate data-driven follow-ups without consulting benchmark labels."""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import joblib
import numpy as np
import report
import study
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

arch = study.arch
original_fit = arch.fit
active_loss = "squared_error"


def fit(config, x, y, rows, seed):
    if config["kind"] != "boost" or active_loss == "squared_error":
        return original_fit(config, x, y, rows, seed)
    begin = time.perf_counter()
    model = HistGradientBoostingRegressor(
        loss=active_loss,
        max_iter=config["iterations"],
        max_leaf_nodes=config["leaves"],
        min_samples_leaf=config["leaf"],
        l2_regularization=config["l2"],
        random_state=seed,
        early_stopping=False,
        categorical_features=None,
    ).fit(np.asarray(x, dtype=np.float32), y, sample_weight=arch.weights(rows))
    return model, time.perf_counter() - begin


arch.fit = fit


def configure(loss="squared_error", weighting="source_mass_matched"):
    global active_loss
    active_loss = loss
    if weighting == "legacy":
        arch.weights = study.legacy_weights
    elif weighting == "fresh_double":

        def doubled(rows):
            w = study.weights(rows)
            return w * np.array([2 if r["_source"] == "fresh" else 1 for r in rows])

        arch.weights = doubled
    else:
        assert weighting == "source_mass_matched"
        arch.weights = study.weights


def fit_one(name, selected, batches, forecasts, seed):
    names = arch.surface.FEATURE_NAMES + arch.surface.FORECAST_NAMES
    if name.endswith("_direct"):
        family = name.split("_")[0]
        model, _ = fit(
            selected[f"direct_{family}"],
            arch.runtime_features(forecasts),
            arch.target(forecasts),
            forecasts,
            seed,
        )
        return {"direct": arch.export_component(model, names)}
    if name == "trees_fixed":
        base_config, correction_config = arch.FIXED_BASE, arch.FIXED_CORRECTION
    else:
        family = "tree" if name == "trees_tuned" else name.split("_")[0]
        base_config = selected[f"base_{family}"]
        correction = "tree" if name == "trees_tuned" else name.split("_")[1]
        correction_config = (
            None
            if name.endswith("_base_only")
            else selected[f"correction_{family}_{correction}"]
        )
    base, _ = fit(
        base_config, arch.features(batches), arch.target(batches), batches, seed
    )
    state = {"base": arch.export_component(base, arch.surface.FEATURE_NAMES)}
    if correction_config:
        oof, audit = arch.cross_fit(base_config, batches, forecasts, seed)
        assert all(a["episode_overlap"] == 0 for a in audit)
        model, _ = fit(
            correction_config,
            np.column_stack((arch.runtime_features(forecasts), oof)),
            arch.target(forecasts) - oof,
            forecasts,
            seed,
        )
        state["correction"] = arch.export_component(
            model, names + arch.surface.BASE_PREDICTION_NAMES
        )
    return state


def calibration_group(x):
    # Total uncached tokens and causal transfer activity are serving features.
    return f"{int(np.searchsorted([8192, 32768], x[0]))}:{int(x[11] > 0)}"


def weighted_median(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, cumulative[-1] / 2)]])


def calibrate(rows, predictions):
    residual = arch.target(rows) - predictions
    weights = arch.weights(rows)
    global_shift = weighted_median(residual, weights)
    groups = np.array([calibration_group(x) for x in arch.runtime_features(rows)])
    shifts, counts = {}, {}
    for group in sorted(set(groups)):
        mask = groups == group
        count = int(mask.sum())
        counts[group] = count
        if count >= 50:
            shift = weighted_median(residual[mask], weights[mask])
            strength = count / (count + 100)
            shifts[group] = strength * shift + (1 - strength) * global_shift
    return dict(global_shift_ms=global_shift, group_shift_ms=shifts, counts=counts)


class FrozenCandidate:
    """Compose immutable architectures, blends, and OOF-calibrated offsets."""

    def __init__(self, state, native=False):
        self.state = state
        self.models = [arch.FrozenArchitecture(s, native) for s in state["models"]]

    def predict_one(self, x):
        prediction = sum(
            w * m.predict_one(x) for w, m in zip(self.state["weights"], self.models)
        )
        calibration = self.state.get("calibration")
        if calibration:
            shift = calibration["global_shift_ms"]
            if self.state.get("group_calibration"):
                shift = calibration["group_shift_ms"].get(calibration_group(x), shift)
            prediction += shift
        return max(1, prediction)


def predict_state(state, rows):
    model = FrozenCandidate(state)
    return np.array([model.predict_one(x) for x in arch.runtime_features(rows)])


def wrapped(state):
    return dict(models=[state], weights=[1.0])


def oof_architecture(name, selected, batches, rows, seed):
    groups = np.array([r["episode"] for r in rows])
    output = np.full(len(rows), np.nan)
    audits = []
    for train, test in GroupKFold(n_splits=3).split(
        np.arange(len(rows)), groups=groups
    ):
        banned = set(groups[test])
        training_batches = [r for r in batches if r["episode"] not in banned]
        training_rows = [rows[i] for i in train]
        assert not banned.intersection(r["episode"] for r in training_batches)
        state = fit_one(name, selected, training_batches, training_rows, seed)
        output[test] = predict_state(wrapped(state), [rows[i] for i in test])
        audits.append(dict(held_episodes=len(banned), episode_overlap=0))
    assert np.isfinite(output).all()
    return output, audits


def plan():
    selection = json.loads((study.ROOT / "architecture_selection.json").read_text())
    summary = json.loads((study.ROOT / "architecture_summary.json").read_text())
    name = selection["winner"]
    scores = summary["scores"]["episode:wide"]
    fresh, historical = scores["fresh"][name], scores["historical"][name]
    candidates = [
        "selected",
        "legacy_weights",
        "global_calibration",
        "group_calibration",
    ]
    reasons = {
        "legacy_weights": "The data coverage study measured unequal source masses.",
        "global_calibration": f"Fresh CV bias is {fresh['bias_ms']:.3f} ms.",
        "group_calibration": (
            f"Fresh worst-workload MAE is {fresh['worst_workload_mae_ms']:.3f} ms; "
            "test OOF residual corrections using causal geometry/load bins."
        ),
    }
    if name != "trees_fixed":
        candidates.append("blend_fixed")
        reasons["blend_fixed"] = (
            "The selected design differs from the fixed-tree control."
        )
    if fresh["mae_ms"] > historical["mae_ms"] * 1.5:
        candidates.append("fresh_double")
        reasons["fresh_double"] = (
            f"Fresh MAE {fresh['mae_ms']:.3f} exceeds historical "
            f"{historical['mae_ms']:.3f} by more than 50%."
        )
    if name in ("trees_fixed", "trees_tuned", "mlp_tree", "tree_direct"):
        candidates.append("absolute_loss")
        reasons["absolute_loss"] = (
            "The selected architecture has a boosted-tree component trained on "
            "squared error; the selection objective is absolute error."
        )
    result = dict(
        architecture=name,
        candidates=candidates,
        reasons=reasons,
        benchmark_used_for_planning=False,
        seeds=[73, 74, 75],
        tuning_configs_reused_from_training_only_inner_splits=True,
    )
    arch.dump(study.ROOT / "improvement_plan.json", result)
    return result


def run_task(task):
    directory = study.ROOT / "improvements" / task["name"]
    completed = directory / "completed.json"
    if completed.exists():
        return json.loads(completed.read_text())
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    proposal = json.loads((study.ROOT / "improvement_plan.json").read_text())
    name = proposal["architecture"]
    pool = joblib.load(task["dataset"])
    forecasts = pool["forecasts"]
    train = [forecasts[i] for i in task["train"]]
    test = [forecasts[i] for i in task["test"]]
    held = {r["episode"] for r in test}
    batches = [r for r in pool["batches"] if r["episode"] not in held]
    if task["regime"] == "matched":
        batches = train
    assert not held.intersection(r["episode"] for r in train + batches)
    fold = study.ROOT / "folds" / task["name"]
    selected = json.loads((fold / "tuning.json").read_text())["selected"]
    existing = json.loads((fold / "checkpoints.json").read_text())
    states = {"selected": wrapped(existing[name])}
    configure()
    with threadpool_limits(limits=1):
        for variant in proposal["candidates"]:
            if variant in ("selected", "global_calibration", "group_calibration"):
                continue
            if variant == "blend_fixed":
                states[variant] = dict(
                    models=[existing[name], existing["trees_fixed"]], weights=[0.5, 0.5]
                )
                continue
            configure(
                loss="absolute_error"
                if variant == "absolute_loss"
                else "squared_error",
                weighting={
                    "legacy_weights": "legacy",
                    "fresh_double": "fresh_double",
                }.get(variant, "source_mass_matched"),
            )
            states[variant] = wrapped(
                fit_one(name, selected, batches, train, task["seed"])
            )
        configure()
        oof, audit = oof_architecture(name, selected, batches, train, task["seed"])
        calibration = calibrate(train, oof)
        for variant in ["global_calibration", "group_calibration"]:
            states[variant] = dict(
                **wrapped(existing[name]),
                calibration=calibration,
                group_calibration=variant == "group_calibration",
            )
        predictions = {n: predict_state(s, test) for n, s in states.items()}
    with np.load(fold / "predictions.npz") as previous:
        np.testing.assert_allclose(predictions["selected"], previous[name], atol=1e-4)
        metadata = {
            k: previous[k] for k in ["y", "episode", "workload", "phase", "features"]
        }
        predictions["trees_fixed"] = previous["trees_fixed"].copy()
    np.savez_compressed(directory / "predictions.npz", **metadata, **predictions)
    arch.dump(directory / "checkpoints.json", states)
    result = dict(
        task=task["name"],
        seconds=time.perf_counter() - started,
        calibration_audit=audit,
        test_training_episode_overlap=0,
        scores={
            n: arch.metrics(arch.target(test), p, test) for n, p in predictions.items()
        },
    )
    arch.dump(completed, result)
    print(json.dumps(dict(event="improvement_task_complete", **result)), flush=True)
    return result


def summarize(tasks, proposal):
    results = {}
    grouped = {}
    for task in tasks:
        key = f"{task['range_scope'] or 'episode'}:{task['regime']}"
        grouped.setdefault(key, []).append(task)
    for key, members in grouped.items():
        pieces = []
        for task in members:
            with np.load(
                study.ROOT / "improvements" / task["name"] / "predictions.npz"
            ) as f:
                pieces.append({k: f[k] for k in f.files})
        data = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        rows = [
            dict(episode=e, workload=w)
            for e, w in zip(data["episode"], data["workload"])
        ]
        metrics = {}
        for phase, mask in [
            ("overall", np.ones(len(rows), dtype=bool)),
            ("fresh", data["phase"] == "fresh"),
            ("historical", data["phase"] != "fresh"),
        ]:
            metrics[phase] = (
                {
                    n: arch.metrics(
                        data["y"][mask],
                        data[n][mask],
                        [r for r, use in zip(rows, mask) if use],
                    )
                    for n in proposal["candidates"] + ["trees_fixed"]
                }
                if mask.any()
                else {}
            )
        results[key] = dict(
            scores=metrics,
            paired={n: report.paired_interval(data, n) for n in proposal["candidates"]},
        )
    cv = results["episode:wide"]["scores"]
    limit = cv["historical"]["trees_fixed"]["mae_ms"] * 1.1
    eligible = [
        n for n in proposal["candidates"] if cv["historical"][n]["mae_ms"] <= limit
    ]
    winner = min(eligible, key=lambda n: cv["fresh"][n]["workload_macro_mae_ms"])
    selection = dict(
        architecture=proposal["architecture"],
        improvement=winner,
        benchmark_used_for_selection=False,
        criterion="fresh episode-CV workload-macro MAE, historical regression <=10%",
        historical_limit_ms=limit,
    )
    arch.dump(study.ROOT / "improvement_summary.json", results)
    arch.dump(study.ROOT / "final_selection.json", selection)
    print(json.dumps(selection), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    proposal = plan()
    tasks = json.loads((study.ROOT / "tasks.json").read_text())
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
    summarize(tasks, proposal)


if __name__ == "__main__":
    main()
