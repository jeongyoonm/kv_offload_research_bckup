# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare frozen regression architectures using grouped CPU-only validation."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "experiments/predictor_two_stage_20261006"))
sys.path.insert(0, str(SOURCE / "experiments/predictor_corpus_20261006"))
DATA = Path("/mnt/localnvme/glm53_predictor_overnight_20261006")
PHASES = [
    "submission_progress",
    "transfer_progress",
    "mixed",
    "dense_validation",
    "online_live",
    "generalization",
    "final_ood",
]
MODELS = [
    "trees_fixed",
    "trees_tuned",
    "tree_base_only",
    "mlp_base_only",
    "tree_mlp",
    "mlp_tree",
    "mlp_mlp",
    "mlp_direct",
    "tree_direct",
]
LABELS = {
    "trees_fixed": "Current tree architecture, fixed recipe",
    "trees_tuned": "Tree base + tree correction, tuned",
    "tree_base_only": "Tree base only",
    "mlp_base_only": "MLP base only",
    "tree_mlp": "Tree base + MLP correction",
    "mlp_tree": "MLP base + tree correction",
    "mlp_mlp": "MLP base + MLP correction",
    "mlp_direct": "Single MLP",
    "tree_direct": "Single boosted-tree model",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


surface = load_module(
    "architecture_surface", SOURCE / "vllm/v1/kv_offload/latency_surface.py"
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def key(row):
    return row["episode"], row["step"]


def valid(row, forecast=False):
    ranks = row["execution_ranks"]
    return len(ranks) == 8 and all(
        0 < r[2] <= r[3]
        and r[3] - r[2] <= 2_000_000
        and (not forecast or r[3] <= row["forecast_time_ns"])
        for r in ranks
    )


def target(rows):
    return np.array(
        [
            (
                max(r[3] + r[6] for r in row["execution_ranks"])
                - max(r[3] for r in row["execution_ranks"])
            )
            / 1e6
            for row in rows
        ]
    )


def features(rows):
    return np.stack([surface.batch_features(row) for row in rows])


def runtime_features(rows):
    result = []
    for row in rows:
        start = max(r[3] for r in row["execution_ranks"])
        extra = list(row["forecast_features"][:10]) + [
            (row["forecast_time_ns"] - start) / 1e6,
            (start - row["dispatch_ns"]) / 1e6,
        ]
        assert np.allclose(extra, row["forecast_features"], atol=1e-9, rtol=0)
        result.append(extra)
    return np.column_stack((features(rows), result))


def weights(rows):
    counts = Counter(row["workload"] for row in rows)
    w = np.array([1 / max(50, counts[row["workload"]]) for row in rows])
    return w * len(w) / w.sum()


class ScaledMLP:
    """Scale using training rows and support NumPy-only frozen inference."""

    def __init__(self, config, seed):
        self.config, self.seed = config, seed

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.config["input"] == "log1p":
            assert np.all(x >= 0)
            x = np.log1p(x)
        return (x - self.x_mean) / self.x_scale

    def fit(self, x, y, sample_weight):
        begin = time.perf_counter()
        raw = np.log1p(x) if self.config["input"] == "log1p" else x
        self.x_mean = np.average(raw, axis=0, weights=sample_weight)
        self.x_scale = np.sqrt(
            np.average((raw - self.x_mean) ** 2, axis=0, weights=sample_weight)
        )
        self.x_scale[self.x_scale < 1e-12] = 1
        self.y_mean = float(np.average(y, weights=sample_weight))
        self.y_scale = max(
            1.0,
            float(np.sqrt(np.average((y - self.y_mean) ** 2, weights=sample_weight))),
        )
        self.estimator = MLPRegressor(
            hidden_layer_sizes=tuple(self.config["layers"]),
            solver=self.config["solver"],
            alpha=self.config["alpha"],
            random_state=self.seed,
            max_iter=1000,
            max_fun=30000,
            learning_rate_init=0.001,
            tol=1e-6,
            n_iter_no_change=40,
            early_stopping=False,
            batch_size=min(256, len(x)),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.estimator.fit(
                self.transform(x),
                (y - self.y_mean) / self.y_scale,
                sample_weight=sample_weight,
            )
        self.fit_info = {
            "seconds": time.perf_counter() - begin,
            "iterations": int(self.estimator.n_iter_),
            "loss": float(self.estimator.loss_),
            "convergence_warnings": [
                str(w.message)
                for w in caught
                if issubclass(w.category, ConvergenceWarning)
            ],
        }
        return self

    def predict(self, x):
        return self.estimator.predict(self.transform(x)) * self.y_scale + self.y_mean

    def frozen(self):
        return {
            "kind": "mlp",
            "config": self.config,
            "x_mean": self.x_mean.tolist(),
            "x_scale": self.x_scale.tolist(),
            "y_mean": self.y_mean,
            "y_scale": self.y_scale,
            "weights": [x.tolist() for x in self.estimator.coefs_],
            "biases": [x.tolist() for x in self.estimator.intercepts_],
        }


def configs(kind):
    if kind == "base_tree":
        return [
            dict(kind=kind, trees=n, leaf=leaf, depth=depth)
            for n in [80, 160]
            for leaf in [3, 5, 10]
            for depth in [12, 18]
        ]
    if kind == "boost":
        return [
            dict(kind=kind, iterations=n, leaves=leaves, leaf=leaf, l2=l2)
            for n in [120, 180]
            for leaves, leaf in [(7, 10), (15, 15), (31, 30)]
            for l2 in [1.0, 10.0]
        ]
    return [
        dict(kind="mlp", layers=layers, input=input_kind, solver=solver, alpha=alpha)
        for layers in [[32], [64], [64, 64], [128, 128]]
        for input_kind, solver, alpha in [
            ("standard", "adam", 0.001),
            ("log1p", "adam", 0.01),
            ("log1p", "lbfgs", 0.01),
        ]
    ]


FIXED_BASE = dict(kind="base_tree", trees=160, leaf=5, depth=18)
FIXED_CORRECTION = dict(kind="boost", iterations=180, leaves=15, leaf=15, l2=10.0)


def fit(config, x, y, rows, seed):
    begin = time.perf_counter()
    w = weights(rows)
    if config["kind"] == "mlp":
        model = ScaledMLP(config, seed).fit(x, y, w)
    elif config["kind"] == "base_tree":
        model = ExtraTreesRegressor(
            n_estimators=config["trees"],
            min_samples_leaf=config["leaf"],
            max_depth=config["depth"],
            random_state=seed,
            n_jobs=1,
        ).fit(x, y, sample_weight=w)
    else:
        model = HistGradientBoostingRegressor(
            max_iter=config["iterations"],
            max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["leaf"],
            l2_regularization=config["l2"],
            random_state=seed,
            early_stopping=False,
            categorical_features=None,
        ).fit(np.asarray(x, dtype=np.float32), y, sample_weight=w)
    return model, time.perf_counter() - begin


def predict(model, x):
    if isinstance(model, HistGradientBoostingRegressor):
        x = np.asarray(x, dtype=np.float32)
    return model.predict(x)


def metrics(y, prediction, rows):
    prediction = np.maximum(1, prediction)
    error = prediction - y
    per_workload = {
        name: float(
            np.mean(np.abs(error[np.array([r["workload"] == name for r in rows])]))
        )
        for name in sorted({r["workload"] for r in rows})
    }
    return {
        "samples": len(y),
        "episodes": len({r["episode"] for r in rows}),
        "mae_ms": float(np.mean(np.abs(error))),
        "mape_percent": float(np.mean(np.abs(error) / y) * 100),
        "bias_ms": float(np.mean(error)),
        "p95_absolute_ms": float(np.quantile(np.abs(error), 0.95)),
        "under_gt5_percent": float(np.mean(error < -5) * 100),
        "over_gt5_percent": float(np.mean(error > 5) * 100),
        "p95_under_ms": float(np.quantile(np.maximum(-error, 0), 0.95)),
        "p95_over_ms": float(np.quantile(np.maximum(error, 0), 0.95)),
        "workload_macro_mae_ms": float(np.mean(list(per_workload.values()))),
        "worst_workload_mae_ms": max(per_workload.values()),
        "per_workload_mae_ms": per_workload,
    }


def objective(y, prediction, rows):
    return metrics(y, prediction, rows)["workload_macro_mae_ms"]


def select(kind, x, y, rows, vx, vy, validation, seed, log, offset=None):
    results = []
    for config in configs(kind):
        model, seconds = fit(config, x, y, rows, seed)
        prediction = predict(model, vx)
        # Score corrections after adding the base. Residuals can be negative;
        # only complete duration predictions have the serving floor of 1 ms.
        actual, estimated = vy, prediction
        if offset is not None:
            actual, estimated = vy + offset, prediction + offset
        record = {
            "config": config,
            "seconds": seconds,
            "validation_macro_mae_ms": objective(actual, estimated, validation),
        }
        if isinstance(model, ScaledMLP):
            record.update(model.fit_info)
        log.append(record)
        results.append((record["validation_macro_mae_ms"], config, model))
    _, config, model = min(results, key=lambda item: item[0])
    return config, model


def cross_fit(config, batches, forecasts, seed):
    x, y, fx = features(batches), target(batches), features(forecasts)
    bg = np.array([r["episode"] for r in batches])
    fg = np.array([r["episode"] for r in forecasts])
    prediction = np.full(len(forecasts), np.nan)
    audit = []
    for _, test in GroupKFold(n_splits=3).split(fx, groups=fg):
        held = set(fg[test])
        train = np.flatnonzero(~np.isin(bg, list(held)))
        assert not held.intersection(bg[train])
        model, _ = fit(config, x[train], y[train], [batches[i] for i in train], seed)
        prediction[test] = predict(model, fx[test])
        audit.append(
            {
                "training_batches": len(train),
                "held_episodes": len(held),
                "held_forecasts": len(test),
                "episode_overlap": 0,
            }
        )
    assert np.all(np.isfinite(prediction))
    return prediction, audit


def export_tree(model, names):
    trainer = load_module(
        "architecture_export",
        SOURCE / "experiments/predictor_two_stage_20261006/train.py",
    )
    if isinstance(model, ExtraTreesRegressor):
        return {"kind": "tree", "state": trainer.export(model, names, {})}
    state, shift = trainer.export_boost(model, names, {})
    return {"kind": "tree", "state": state, "shift": shift}


def export_component(model, names):
    return model.frozen() if isinstance(model, ScaledMLP) else export_tree(model, names)


class FrozenComponent:
    """Run either exported tree arrays or an exported dense network on CPU."""

    def __init__(self, state, native=False):
        self.state = state
        if state["kind"] == "tree":
            self.tree = surface.LatencySurface(state["state"])
            self.native = native
        else:
            self.mean = np.array(state["x_mean"])
            self.scale = np.array(state["x_scale"])
            self.layers = [
                (np.array(w), np.array(b))
                for w, b in zip(state["weights"], state["biases"])
            ]

    def predict_one(self, x):
        s = self.state
        if s["kind"] == "tree":
            return self.tree._predict_features(
                x, {"native_predictor": self.native}
            ) / 1e6 - s.get("shift", 0)
        if s["config"]["input"] == "log1p":
            x = np.log1p(x)
        v = (x - self.mean) / self.scale
        for i, (weight, bias) in enumerate(self.layers):
            v = v @ weight + bias
            if i < len(self.layers) - 1:
                v = np.maximum(v, 0)
        return float(v[0] * s["y_scale"] + s["y_mean"])


class FrozenArchitecture:
    def __init__(self, state, native=False):
        self.state = state
        self.base = (
            FrozenComponent(state["base"], native) if state.get("base") else None
        )
        self.correction = (
            FrozenComponent(state["correction"], native)
            if state.get("correction")
            else None
        )
        self.direct = (
            FrozenComponent(state["direct"], native) if state.get("direct") else None
        )

    def predict_one(self, x):
        if self.direct:
            return max(1, self.direct.predict_one(x))
        base = self.base.predict_one(x[:11])
        if self.correction:
            base += self.correction.predict_one(np.r_[x, base])
        return max(1, base)


def full_fit(base_config, correction_config, batches, forecasts, seed):
    base, _ = fit(base_config, features(batches), target(batches), batches, seed)
    oof, audit = cross_fit(base_config, batches, forecasts, seed)
    correction, _ = fit(
        correction_config,
        np.column_stack((runtime_features(forecasts), oof)),
        target(forecasts) - oof,
        forecasts,
        seed,
    )
    return base, correction, audit


def run_fold(task):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    start = time.perf_counter()
    output = Path(task["output"]) / "folds" / task["name"]
    if (output / "completed.json").exists():
        return json.loads((output / "completed.json").read_text())
    pool = joblib.load(task["dataset"])
    frows, brows = pool["forecasts"], pool["batches"]
    train = [frows[i] for i in task["train"]]
    test = [frows[i] for i in task["test"]]
    banned_episodes = {r["episode"] for r in test}
    batches = [r for r in brows if r["episode"] not in banned_episodes]
    if task.get("historical"):
        batches = [r for r in batches if r["_phase"] == "historical"]
    if task.get("workload"):
        banned_workloads = {task["workload"]}
        if task["workload"] != "mixed_h100":
            banned_workloads.add("mixed_h100")
        batches = [r for r in batches if r["workload"] not in banned_workloads]
        assert not banned_workloads.intersection(r["workload"] for r in batches)
        assert not banned_workloads.intersection(r["workload"] for r in train)
    if task["regime"] == "matched":
        batches = train
    assert not banned_episodes.intersection(r["episode"] for r in batches + train)
    assert not {key(r) for r in test}.intersection(key(r) for r in batches + train)
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "test_forecasts": len(test),
        "training_forecasts": len(train),
        "training_batches": len(batches),
        "test_episodes": len(banned_episodes),
        "test_training_episode_overlap": 0,
        "test_training_batch_overlap": 0,
    }
    dump(output / "split_audit.json", audit)
    print("START", task["name"], audit, flush=True)
    groups = np.array([r["episode"] for r in train])
    it, iv = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=73).split(
            np.arange(len(train)), groups=groups
        )
    )
    inner, validation = [train[i] for i in it], [train[i] for i in iv]
    held = {r["episode"] for r in validation}
    ib = [r for r in batches if r["episode"] not in held]
    assert not held.intersection(r["episode"] for r in ib + inner)
    audit["inner_validation_episode_overlap"] = 0
    selected, logs, fitted_inner, inner_oof, cross_audit = {}, {}, {}, {}, {}
    with threadpool_limits(limits=1):
        for family, kind in [("tree", "base_tree"), ("mlp", "mlp")]:
            logs[f"base_{family}"] = []
            cfg, mdl = select(
                kind,
                features(ib),
                target(ib),
                ib,
                features(validation),
                target(validation),
                validation,
                task["seed"],
                logs[f"base_{family}"],
            )
            selected[f"base_{family}"], fitted_inner[family] = cfg, mdl
            inner_oof[family], cross_audit[f"inner_{family}"] = cross_fit(
                cfg, ib, inner, task["seed"]
            )
            oof = inner_oof[family]
            val_base = predict(mdl, features(validation))
            for correction, kind in [("tree", "boost"), ("mlp", "mlp")]:
                name = f"correction_{family}_{correction}"
                logs[name] = []
                cfg, _ = select(
                    kind,
                    np.column_stack((runtime_features(inner), oof)),
                    target(inner) - oof,
                    inner,
                    np.column_stack((runtime_features(validation), val_base)),
                    target(validation) - val_base,
                    validation,
                    task["seed"],
                    logs[name],
                    offset=val_base,
                )
                selected[name] = cfg
        for family, kind in [("mlp", "mlp"), ("tree", "boost")]:
            name = f"direct_{family}"
            logs[name] = []
            cfg, _ = select(
                kind,
                runtime_features(inner),
                target(inner),
                inner,
                runtime_features(validation),
                target(validation),
                validation,
                task["seed"],
                logs[name],
            )
            selected[name] = cfg
        dump(output / "tuning.json", {"selected": selected, "trials": logs})
        tx = runtime_features(test)
        predictions, exported = {}, {}
        for family in ["tree", "mlp"]:
            cfg = selected[f"base_{family}"]
            base, _ = fit(
                cfg, features(batches), target(batches), batches, task["seed"]
            )
            base_pred = predict(base, tx[:, :11])
            base_state = export_component(base, surface.FEATURE_NAMES)
            predictions[f"{family}_base_only"] = np.maximum(1, base_pred)
            exported[f"{family}_base_only"] = {"base": base_state}
            oof, cross_audit[f"full_{family}"] = cross_fit(
                cfg, batches, train, task["seed"]
            )
            for correction in ["tree", "mlp"]:
                name = (
                    "trees_tuned"
                    if family == correction == "tree"
                    else f"{family}_{correction}"
                )
                mdl, _ = fit(
                    selected[f"correction_{family}_{correction}"],
                    np.column_stack((runtime_features(train), oof)),
                    target(train) - oof,
                    train,
                    task["seed"],
                )
                predictions[name] = np.maximum(
                    1, base_pred + predict(mdl, np.column_stack((tx, base_pred)))
                )
                exported[name] = {
                    "base": base_state,
                    "correction": export_component(
                        mdl,
                        surface.FEATURE_NAMES
                        + surface.FORECAST_NAMES
                        + surface.BASE_PREDICTION_NAMES,
                    ),
                }
        base, correction, cross_audit["fixed"] = full_fit(
            FIXED_BASE, FIXED_CORRECTION, batches, train, task["seed"]
        )
        bp = predict(base, tx[:, :11])
        predictions["trees_fixed"] = np.maximum(
            1, bp + predict(correction, np.column_stack((tx, bp)))
        )
        exported["trees_fixed"] = {
            "base": export_component(base, surface.FEATURE_NAMES),
            "correction": export_component(
                correction,
                surface.FEATURE_NAMES
                + surface.FORECAST_NAMES
                + surface.BASE_PREDICTION_NAMES,
            ),
        }
        for family in ["tree", "mlp"]:
            mdl, _ = fit(
                selected[f"direct_{family}"],
                runtime_features(train),
                target(train),
                train,
                task["seed"],
            )
            predictions[f"{family}_direct"] = np.maximum(1, predict(mdl, tx))
            exported[f"{family}_direct"] = {
                "direct": export_component(
                    mdl, surface.FEATURE_NAMES + surface.FORECAST_NAMES
                )
            }
        equivalence = {}
        for name, state in exported.items():
            frozen = FrozenArchitecture(state)
            actual = np.array([frozen.predict_one(x) for x in tx])
            error = float(np.max(np.abs(actual - predictions[name])))
            assert error < 1e-4, (task["name"], name, error)
            equivalence[name] = error
        y = target(test)
        scores = {name: metrics(y, p, test) for name, p in predictions.items()}
        np.savez_compressed(
            output / "predictions.npz",
            y=y,
            episode=np.array([r["episode"] for r in test]),
            workload=np.array([r["workload"] for r in test]),
            phase=np.array([r["_phase"] for r in test]),
            features=tx,
            **predictions,
        )
        if task.get("historical"):
            dump(output / "checkpoints.json", exported)
        dump(output / "cross_fit_audit.json", cross_audit)
        result = {
            "name": task["name"],
            "seed": task["seed"],
            "regime": task["regime"],
            "audit": audit,
            "scores": scores,
            "export_equivalence_max_ms": equivalence,
            "selected": selected,
            "seconds": time.perf_counter() - start,
        }
        dump(output / "completed.json", result)
        print(
            "DONE",
            task["name"],
            "seconds",
            round(result["seconds"], 1),
            {n: round(v["mae_ms"], 2) for n, v in scores.items()},
            flush=True,
        )
        return result


def prepare(output):
    forecasts, batches, sources = [], [], {}
    for kind, filename in [
        ("batches", "batches.jsonl"),
        ("forecasts", "forecasts.jsonl"),
    ]:
        path = DATA / "refreshed/data" / filename
        rows = read(path)
        for row in rows:
            row["_phase"] = "historical"
            assert valid(row, kind == "forecasts")
        (batches if kind == "batches" else forecasts).extend(rows)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for phase in PHASES:
        path = DATA / phase / "frozen_evaluation/forecasts/forecasts.jsonl"
        rows = [r for r in read(path) if valid(r, True)]
        for row in rows:
            row["_phase"] = phase
        forecasts.extend(rows)
        # This control uses only additional batches with recorded causal forecasts.
        batches.extend(rows)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len({key(r) for r in forecasts}) == len(forecasts)
    assert len({key(r) for r in batches}) == len(batches)
    assert np.all(np.isfinite(runtime_features(forecasts)))
    assert np.all(target(batches) > 0)
    dataset = output / "dataset.joblib"
    joblib.dump({"forecasts": forecasts, "batches": batches}, dataset, compress=3)
    dump(
        output / "inventory.json",
        {
            "forecasts": len(forecasts),
            "batches": len(batches),
            "forecast_episodes": len({r["episode"] for r in forecasts}),
            "batch_episodes": len({r["episode"] for r in batches}),
            "workloads": dict(Counter(r["workload"] for r in forecasts)),
            "base_features": surface.FEATURE_NAMES,
            "runtime_features": surface.FORECAST_NAMES,
            "sources_sha256": sources,
            "cpu_only": True,
        },
    )
    tasks = []

    def add(name, train, test, regime="wide", **extra):
        tasks.append(
            dict(
                name=name,
                train=list(map(int, train)),
                test=list(map(int, test)),
                regime=regime,
                dataset=str(dataset),
                output=str(output),
                seed=17,
                **extra,
            )
        )

    historic = np.array([r["_phase"] == "historical" for r in forecasts])
    add(
        "historical_wide",
        np.flatnonzero(historic),
        np.flatnonzero(~historic),
        historical=True,
    )
    add(
        "historical_matched",
        np.flatnonzero(historic),
        np.flatnonzero(~historic),
        regime="matched",
        historical=True,
    )
    groups = np.array([r["episode"] for r in forecasts])
    for i, (train, test) in enumerate(
        GroupKFold(n_splits=3).split(np.arange(len(forecasts)), groups=groups)
    ):
        add(f"episode_{i}_wide", train, test)
        add(f"episode_{i}_matched", train, test, regime="matched")
    workloads = np.array([r["workload"] for r in forecasts])
    for workload, count in sorted(Counter(workloads).items()):
        if count < 60:
            continue
        banned = [workload]
        if workload != "mixed_h100":
            banned.append("mixed_h100")
        add(
            f"workload_{workload}",
            np.flatnonzero(~np.isin(workloads, banned)),
            np.flatnonzero(workloads == workload),
            workload=workload,
        )
    dump(output / "tasks.json", tasks)
    return tasks


def benchmark(output):
    library = DATA / "fast_forest/build/libpredictor_forest_walk.so"
    assert library.exists()
    os.environ["VLLM_DEFERRAL_FAST_FOREST"] = str(library)
    reports = []
    with threadpool_limits(limits=1):
        for regime in ["wide", "matched"]:
            folder = output / "folds" / f"historical_{regime}"
            state = json.loads((folder / "checkpoints.json").read_text())
            a = np.load(folder / "predictions.npz")
            x = a["features"]
            for name in MODELS:
                for backend in ["numpy", "native_tree"]:
                    if backend == "native_tree" and "tree" not in name:
                        continue
                    begin = time.perf_counter_ns()
                    frozen = FrozenArchitecture(state[name], backend == "native_tree")
                    init = (time.perf_counter_ns() - begin) / 1e6
                    got = np.array([frozen.predict_one(v) for v in x])
                    assert np.max(np.abs(got - a[name])) < 1e-4
                    for i in range(200):
                        frozen.predict_one(x[i % len(x)])
                    times = []
                    for repeat in range(5):
                        for i in np.random.default_rng(900 + repeat).permutation(
                            len(x)
                        ):
                            begin = time.perf_counter_ns()
                            frozen.predict_one(x[i])
                            times.append((time.perf_counter_ns() - begin) / 1000)
                    reports.append(
                        {
                            "regime": regime,
                            "model": name,
                            "backend": backend,
                            "calls": len(times),
                            "mean_us": float(np.mean(times)),
                            "median_us": float(np.median(times)),
                            "p95_us": float(np.quantile(times, 0.95)),
                            "p99_us": float(np.quantile(times, 0.99)),
                            "max_us": float(max(times)),
                            "init_ms": init,
                            "checkpoint_json_bytes": len(
                                json.dumps(state[name]).encode()
                            ),
                            "max_equivalence_error_ms": float(
                                np.max(np.abs(got - a[name]))
                            ),
                        }
                    )
    dump(output / "runtime.json", reports)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.benchmark_only:
        benchmark(args.output)
        return
    tasks = (
        json.loads((args.output / "tasks.json").read_text())
        if (args.output / "tasks.json").exists()
        else prepare(args.output)
    )
    if args.prepare_only:
        return
    if args.only:
        tasks = [t for t in tasks if t["name"] in args.only]
        assert tasks
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_fold, task): task["name"] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            dump(args.output / "progress.json", results)
    if not args.only:
        benchmark(args.output)
    dump(args.output / "completed.json", {"folds": len(results), "cpu_only": True})


if __name__ == "__main__":
    main()
