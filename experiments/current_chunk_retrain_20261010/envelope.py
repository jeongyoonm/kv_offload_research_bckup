# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refit the later constrained submission envelope and sparse-MLP blend."""

import argparse
import copy
import gzip
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from experiments.current_chunk_retrain_20261010 import architectures as study

arch = study.arch
CORE_INDEX, CORE_SCALE = study.runtime.CORE_INDEX, study.runtime.CORE_SCALE
HOST_INDEX, HOST_SCALE = study.runtime.HOST_INDEX, study.runtime.HOST_SCALE
FrozenSealPredictor = study.FrozenSealPredictor


def matrices(x):
    return (
        np.column_stack((np.ones(len(x)), x[:, CORE_INDEX] / CORE_SCALE)),
        np.column_stack((np.ones(len(x)), x[:, HOST_INDEX] / HOST_SCALE)),
    )


def fit_base(rows, regularization):
    x, y, w = study.full_features(rows), arch.target(rows), arch.weights(rows)
    c, h = matrices(x)
    n = c.shape[1]
    bounds = [(1.0, None)] + [(0.0, None)] * (n - 1)
    bounds += [(0.0, None)] * h.shape[1]
    for i in [5, 6]:
        bounds[n + i] = (None, None)
    penalty = np.ones(n + h.shape[1])
    penalty[[0, n]] = 0

    def objective(theta):
        core, host = c @ theta[:n], h @ theta[n:] - x[:, -1]
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
    best = min(trials, key=lambda trial: trial.fun)
    return dict(
        core_coefficients=best.x[:n].tolist(),
        host_coefficients=best.x[n:].tolist(),
        regularization=regularization,
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


def base_predict(base, rows):
    x = study.full_features(rows)
    c, h = matrices(x)
    return np.maximum(
        1,
        np.maximum(
            c @ np.array(base["core_coefficients"]),
            h @ np.array(base["host_coefficients"]) - x[:, -1],
        ),
    )


def fit(rows, seed, correction_config=None):
    groups = np.array([r["episode"] for r in rows])
    it, iv = next(
        GroupShuffleSplit(1, test_size=0.2, random_state=73).split(rows, groups=groups)
    )
    tr, va = [rows[i] for i in it], [rows[i] for i in iv]
    assert not {r["episode"] for r in tr} & {r["episode"] for r in va}
    trials = []
    for reg in [0.0, 1e-4, 1e-2, 1.0]:
        base = fit_base(tr, reg)
        trials.append(
            dict(
                regularization=reg,
                macro_mae_ms=arch.objective(
                    arch.target(va), base_predict(base, va), va
                ),
            )
        )
    reg = min(trials, key=lambda t: t["macro_mae_ms"])["regularization"]
    base = fit_base(rows, reg)
    oof = np.full(len(rows), np.nan)
    for it, iv in GroupKFold(3).split(rows, groups=groups):
        tr, va = [rows[i] for i in it], [rows[i] for i in iv]
        assert not {r["episode"] for r in tr} & {r["episode"] for r in va}
        oof[iv] = base_predict(fit_base(tr, reg), va)
    assert np.isfinite(oof).all()
    x = np.column_stack((study.full_features(rows), oof))
    correction, seconds = arch.fit(
        correction_config or arch.FIXED_CORRECTION,
        x,
        arch.target(rows) - oof,
        rows,
        seed,
    )
    names = (
        arch.surface.FEATURE_NAMES
        + arch.surface.FORECAST_NAMES
        + arch.surface.SPARSE_NAMES
        + arch.surface.ENVELOPE_NAMES
        + arch.surface.BASE_PREDICTION_NAMES
    )
    exported = dict(base=base, correction=arch.export_component(correction, names))
    bp = base_predict(base, rows)
    validation_x = np.column_stack((study.full_features(rows), bp))
    expected = np.maximum(1, bp + arch.predict(correction, validation_x))
    component = arch.FrozenComponent(exported["correction"])
    actual = np.maximum(
        1, bp + np.array([component.predict_one(v) for v in validation_x])
    )
    error = float(np.max(np.abs(expected - actual)))
    assert error < 2e-6, dict(export_error_ms=error)
    return exported, dict(
        regularization_trials=trials,
        oof_episode_overlap=0,
        correction_seconds=seconds,
        export_error_ms=error,
    )


def wrap(envelope, neural, fallback, rows, blend):
    x = study.full_features(rows)
    checkpoint = dict(
        envelope=dict(
            envelope=envelope,
            fallback=fallback["models"][0],
            support=dict(
                indices=[5],
                minimum=[float(x[:, 5].min())],
                maximum=[float(x[:, 5].max())],
            ),
        ),
        neural=neural,
        weights=[0.5, 0.5] if blend else [0.0, 1.0],
        excluded_feature_indices=[21],
    )
    state = study.wrap(checkpoint)
    state["architecture"] = (
        "envelope_sparse_mlp_offset" if blend else "submission_envelope_offset"
    )
    return state


def predict(state, rows):
    model = FrozenSealPredictor(state)
    return np.array(
        [model.predict_features_ms(x, native=False) for x in study.full_features(rows)]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-state", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arch.weights = study.weights
    pool = json.loads(gzip.decompress((args.study / "pool.json.gz").read_bytes()))
    forecasts = [r for r in pool["forecasts"] if r.get("split", "train") == "train"]
    development = [r for r in pool["forecasts"] if r.get("split") == "development"]
    final = [r for r in pool["forecasts"] if r.get("split") == "final"]
    original = json.loads(args.old_state.read_text())
    fallback = json.loads((args.study / "trees_fixed.json").read_text())[
        "latency_surface"
    ]["checkpoint"]
    neural = json.loads((args.study / "sparse_mlp.json").read_text())[
        "latency_surface"
    ]["checkpoint"]
    study.train.save(
        args.output / "protocol.json",
        dict(
            seed=73,
            envelope_method="Historical constrained pseudo-Huber envelope; "
            "training-only regularization selection; purged OOF boosted residual",
            elapsed_mask=21,
            blend_weights=[0.5, 0.5],
            suffix_support_guard=True,
            external_shape_calibration=False,
        ),
    )
    with threadpool_limits(limits=1):
        started = time.monotonic()
        groups = np.array([r["episode"] for r in forecasts])
        cv = []
        for fold, (it, iv) in enumerate(GroupKFold(3).split(forecasts, groups=groups)):
            path = args.output / f"cv_{fold}.json"
            if path.exists():
                cv.append(json.loads(path.read_text()))
                continue
            tr, va = [forecasts[i] for i in it], [forecasts[i] for i in iv]
            assert not {r["episode"] for r in tr} & {r["episode"] for r in va}
            read = lambda name, fold=fold: json.loads(
                (args.study / f"cv_{fold}_{name}_checkpoint.json").read_text()
            )
            correction_config = json.loads(
                (args.study / f"cv_{fold}_trees_tuned.json").read_text()
            )["audit"]["correction"]
            fitted, fold_audit = fit(tr, 73, correction_config)
            fold_scores = {}
            for blend, name in [
                (False, "envelope_boost"),
                (True, "envelope_sparse_mlp"),
            ]:
                surface = wrap(
                    fitted, read("sparse_mlp"), read("trees_fixed"), tr, blend
                )
                fold_scores[name] = arch.metrics(
                    arch.target(va), predict(surface, va), va
                )
            result = dict(fold=fold, scores=fold_scores, audit=fold_audit)
            study.train.save(path, result)
            cv.append(result)
        correction_config = json.loads(
            (args.study / "trees_tuned_fit_audit.json").read_text()
        )["correction"]
        envelope, audit = fit(forecasts, 73, correction_config)
        scores, exports = {}, {}
        for blend, name in [(False, "envelope_boost"), (True, "envelope_sparse_mlp")]:
            surface = wrap(envelope, neural, fallback, forecasts, blend)
            state = copy.deepcopy(original)
            state["latency_surface"] = surface
            path = args.output / f"{name}.json"
            study.train.save(path, state)
            exports[name] = str(path)
            scores[name] = arch.metrics(
                arch.target(development), predict(surface, development), development
            )
        study.train.save(
            args.output / "selection.json",
            dict(
                selected=min(scores, key=lambda n: scores[n]["workload_macro_mae_ms"]),
                development=scores,
                exports=exports,
                frozen_before_final=True,
                audit=audit,
                cv=cv,
                seconds=time.monotonic() - started,
            ),
        )
        final_scores = {}
        for name, path in exports.items():
            surface = json.loads(Path(path).read_text())["latency_surface"]
            final_scores[name] = arch.metrics(
                arch.target(final), predict(surface, final), final
            )
        study.train.save(args.output / "final_scores.json", final_scores)


if __name__ == "__main__":
    main()
