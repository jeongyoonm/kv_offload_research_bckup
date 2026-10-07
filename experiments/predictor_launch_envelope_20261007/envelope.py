# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU study of max(GPU work, host submission minus observed queue lead)."""

import argparse
import importlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

os.environ["CUDA_VISIBLE_DEVICES"] = ""
SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "experiments/predictor_expanded_20261007"))
improve = importlib.import_module("improve")
sparse = importlib.import_module("sparse")
expanded = importlib.import_module("study")

ROOT = Path("/mnt/localnvme/glm53_launch_envelope_20261007")
DATA = expanded.ROOT
ARCH = expanded.arch
CORE_INDEX = [0, 1, 8, 23, 24, 25, 11, 12]
CORE_SCALE = [1000, 16, 1000, 1e8, 1e8, 1e7, 8, 1e6]
HOST_INDEX = [1, 8, 11, 12, 13, 15, 16, 18]
HOST_SCALE = [16, 1000, 8, 1e6, 1000, 1000, 8, 8]
FEATURE_NAMES = (
    ARCH.surface.FEATURE_NAMES
    + ARCH.surface.FORECAST_NAMES
    + sparse.EXTRA_NAMES
    + ["all_rank_start_minus_last_enqueue_ms"]
)
SCHEMA = FEATURE_NAMES + ARCH.surface.BASE_PREDICTION_NAMES
if SCHEMA not in ARCH.surface.FEATURE_SCHEMAS:
    ARCH.surface.FEATURE_SCHEMAS.append(SCHEMA)
NAMES = ["envelope_only", "envelope_boost", "envelope_absolute", "envelope_mlp"]


def features(rows):
    leads = []
    for row in rows:
        ranks = row["execution_ranks"]
        assert len(ranks) == 8 and all(
            0 < r[1] <= r[3] <= row["forecast_time_ns"] for r in ranks
        )
        leads.append((max(r[3] for r in ranks) - max(r[1] for r in ranks)) / 1e6)
    return np.column_stack(
        (ARCH.runtime_features(rows), sparse.extra_features(rows), leads)
    )


def matrices(x):
    return (
        np.column_stack((np.ones(len(x)), x[:, CORE_INDEX] / CORE_SCALE)),
        np.column_stack((np.ones(len(x)), x[:, HOST_INDEX] / HOST_SCALE)),
    )


def fit(rows, regularization):
    x, y, w = features(rows), ARCH.target(rows), ARCH.weights(rows)
    c, h = matrices(x)
    lead = x[:, -1]
    n = c.shape[1]
    bounds = [(1.0, None)] + [(0.0, None)] * (n - 1)
    bounds += [(0.0, None)] * h.shape[1]
    for i in [5, 6]:
        bounds[n + i] = (None, None)
    penalty = np.ones(n + h.shape[1])
    penalty[[0, n]] = 0

    def objective(theta):
        core, host = c @ theta[:n], h @ theta[n:] - lead
        mask = core >= host
        error = np.maximum(core, host) - y
        root = np.sqrt(1 + (error / 5) ** 2)
        loss = np.sum(w * 25 * (root - 1)) / w.sum()
        slope = w * error / root / w.sum()
        gradient = np.r_[c.T @ (slope * mask), h.T @ (slope * ~mask)]
        loss += regularization * np.sum(penalty * theta**2) / 2
        gradient += regularization * penalty * theta
        return loss, gradient

    trials = []
    for core_intercept, token_rate, host_intercept in [
        (30, 18, 125),
        (50, 15, 150),
        (10, 25, 100),
    ]:
        initial = np.zeros(n + h.shape[1])
        initial[0], initial[1], initial[n], initial[n + 3] = (
            core_intercept,
            token_rate,
            host_intercept,
            180,
        )
        fitted = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=1500, ftol=1e-11, gtol=1e-7, maxls=50),
        )
        assert np.isfinite(fitted.x).all()
        trials.append(fitted)
    best = min(trials, key=lambda t: t.fun)
    return dict(
        kind="launch_envelope",
        core_coefficients=best.x[:n].tolist(),
        host_coefficients=best.x[n:].tolist(),
        regularization=regularization,
        fit_loss=float(best.fun),
        convergence=[
            dict(
                success=bool(t.success),
                message=str(t.message),
                iterations=int(t.nit),
                objective=float(t.fun),
            )
            for t in trials
        ],
    )


class Frozen:
    def __init__(self, state, native=False):
        self.state = state
        self.core = np.array(state["base"]["core_coefficients"])
        self.host = np.array(state["base"]["host_coefficients"])
        self.correction = (
            ARCH.FrozenComponent(state["correction"], native)
            if (state.get("correction"))
            else None
        )

    def predict_one(self, x):
        core = np.r_[1, x[CORE_INDEX] / CORE_SCALE] @ self.core
        host = np.r_[1, x[HOST_INDEX] / HOST_SCALE] @ self.host - x[-1]
        value = max(1, float(core), float(host))
        if self.correction:
            value += self.correction.predict_one(np.r_[x, value])
        return max(1, value)


def predict(state, rows):
    model = Frozen(state)
    return np.array([model.predict_one(v) for v in features(rows)])


def fit_variants(train, selected, seed):
    improve.configure()
    groups = np.array([r["episode"] for r in train])
    ti, vi = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=73).split(
            np.arange(len(train)), groups=groups
        )
    )
    inner, validation = [train[i] for i in ti], [train[i] for i in vi]
    assert not {r["episode"] for r in inner} & {r["episode"] for r in validation}
    trials = []
    for regularization in [0.0, 1e-4, 1e-2, 1.0]:
        state = fit(inner, regularization)
        score = ARCH.objective(
            ARCH.target(validation), predict(dict(base=state), validation), validation
        )
        trials.append(dict(regularization=regularization, macro_mae_ms=score))
    reg = min(trials, key=lambda r: r["macro_mae_ms"])["regularization"]
    base = fit(train, reg)
    oof = np.full(len(train), np.nan)
    for it, iv in GroupKFold(n_splits=3).split(np.arange(len(train)), groups=groups):
        tr, va = [train[i] for i in it], [train[i] for i in iv]
        assert not {r["episode"] for r in tr} & {r["episode"] for r in va}
        oof[iv] = predict(dict(base=fit(tr, reg)), va)
    assert np.isfinite(oof).all()
    x = np.column_stack((features(train), oof))
    states = {"envelope_only": dict(base=base)}
    for name, family, loss in [
        ("envelope_boost", "tree", "squared_error"),
        ("envelope_absolute", "tree", "absolute_error"),
        ("envelope_mlp", "mlp", "squared_error"),
    ]:
        improve.configure(loss=loss)
        model, _ = improve.fit(
            selected[f"correction_tree_{family}"],
            x,
            ARCH.target(train) - oof,
            train,
            seed,
        )
        states[name] = dict(base=base, correction=ARCH.export_component(model, SCHEMA))
        bx = predict(dict(base=base), train[:100])
        expected = np.maximum(
            1, bx + ARCH.predict(model, np.column_stack((features(train[:100]), bx)))
        )
        np.testing.assert_allclose(
            predict(states[name], train[:100]), expected, atol=1e-4, rtol=0
        )
    improve.configure()
    return states, dict(
        regularization_trials=trials,
        base_oof_episode_overlap=0,
        correction_capacity_reused_from_training_only_tuning=True,
    )


def run_task(task):
    directory = ROOT / "folds" / task["name"]
    completed = directory / "completed.json"
    if completed.exists():
        return json.loads(completed.read_text())
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    pool = joblib.load(task["dataset"])
    train, test = ([pool["forecasts"][i] for i in task[k]] for k in ["train", "test"])
    assert not {r["episode"] for r in train} & {r["episode"] for r in test}
    selected = json.loads((DATA / "folds" / task["name"] / "tuning.json").read_text())[
        "selected"
    ]
    with threadpool_limits(limits=1):
        states, audit = fit_variants(train, selected, task["seed"])
        predictions = {n: predict(s, test) for n, s in states.items()}
    np.savez_compressed(
        directory / "predictions.npz",
        y=ARCH.target(test),
        episode=np.array([r["episode"] for r in test]),
        workload=np.array([r["workload"] for r in test]),
        phase=np.array([r["_phase"] for r in test]),
        **predictions,
    )
    ARCH.dump(directory / "checkpoints.json", states)
    result = dict(
        task=task["name"],
        seconds=time.perf_counter() - started,
        audit=audit,
        scores={
            n: ARCH.metrics(ARCH.target(test), v, test) for n, v in predictions.items()
        },
    )
    ARCH.dump(completed, result)
    print(
        json.dumps(
            dict(
                event="envelope_completed", task=task["name"], seconds=result["seconds"]
            )
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    tasks = json.loads((DATA / "tasks.json").read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        result = run_task(
            next(t for t in tasks if t["name"] == "episode_0_matched_s73")
        )
        print(
            json.dumps({n: round(v["mae_ms"], 3) for n, v in result["scores"].items()})
        )
        return
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        for future in as_completed([executor.submit(run_task, t) for t in tasks]):
            future.result()


if __name__ == "__main__":
    main()
