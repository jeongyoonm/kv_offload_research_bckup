# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Purged current-worker comparison of tree and later direct MLP recipes."""

import argparse
import copy
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from experiments.current_chunk_retrain_20261010 import train
from vllm.v1.kv_offload.frozen_seal_predictor import FrozenSealPredictor

arch = train.arch
MLP = dict(kind="mlp", layers=[64, 64], input="log1p", solver="lbfgs", alpha=0.01)
NAMES = ("trees_fixed", "trees_tuned", "tree_direct", "mlp_direct", "sparse_mlp")


def full_features(rows):
    extra, lead = [], []
    for row in rows:
        prefix = causal = sparse = 0
        for q, c in row["request_shapes"]:
            prefix += q * c
            causal += q * (q + 1) / 2
            n = min(q, max(0, 2048 - c))
            sparse += n * c + n * (n + 1) / 2 + (q - n) * 2048
        extra.append([prefix, causal, sparse])
        ranks = row["execution_ranks"]
        lead.append((max(r[3] for r in ranks) - max(r[1] for r in ranks)) / 1e6)
    return np.column_stack((train.matrix(rows), extra, lead))


def wrap(checkpoint):
    return dict(
        schema=4,
        architecture="guarded_sparse_mlp_offset",
        checkpoint=checkpoint,
        provenance=dict(general_prefill=True),
        target_schema="critical_end_after_all_rank_start.v1",
    )


def predict(checkpoint, rows):
    frozen = FrozenSealPredictor(wrap(checkpoint))
    return np.array(
        [frozen.predict_features_ms(x, native=False) for x in full_features(rows)]
    )


def fit(name, batches, forecasts, seed, fallback=None):
    if name == "trees_fixed":
        return train.fit(batches, forecasts, seed)
    if name == "trees_tuned":
        groups = np.array([r["episode"] for r in forecasts])
        it, iv = next(
            GroupShuffleSplit(1, test_size=0.2, random_state=73).split(
                forecasts, groups=groups
            )
        )
        tr, va = [forecasts[i] for i in it], [forecasts[i] for i in iv]
        banned = {r["episode"] for r in va}
        tb = [r for r in batches if r["episode"] not in banned]
        log = []
        base_config, base = arch.select(
            "base_tree",
            arch.features(tb),
            arch.target(tb),
            tb,
            arch.features(va),
            arch.target(va),
            va,
            seed,
            log,
        )
        oof, _ = arch.cross_fit(base_config, tb, tr, seed)
        vp = arch.predict(base, arch.features(va))
        correction_config, _ = arch.select(
            "boost",
            np.column_stack((train.matrix(tr), oof)),
            arch.target(tr) - oof,
            tr,
            np.column_stack((train.matrix(va), vp)),
            arch.target(va) - vp,
            va,
            seed,
            log,
            offset=vp,
        )
        checkpoint, folds = train.fit(
            batches, forecasts, seed, base_config, correction_config
        )
        return checkpoint, dict(
            base=base_config, correction=correction_config, tuning=log, folds=folds
        )
    x = train.matrix(forecasts)
    names = arch.surface.FEATURE_NAMES + arch.surface.FORECAST_NAMES
    indices = None
    if name == "sparse_mlp":
        x = full_features(forecasts)[:, :26]
        names += arch.surface.SPARSE_NAMES
        indices = list(range(23)) + [28, 29, 30]
    tuning = []
    config = MLP
    if name == "tree_direct":
        groups = np.array([r["episode"] for r in forecasts])
        it, iv = next(
            GroupShuffleSplit(1, test_size=0.2, random_state=73).split(
                forecasts, groups=groups
            )
        )
        config, _ = arch.select(
            "boost",
            x[it],
            arch.target(forecasts)[it],
            [forecasts[i] for i in it],
            x[iv],
            arch.target(forecasts)[iv],
            [forecasts[i] for i in iv],
            seed,
            tuning,
        )
    model, seconds = arch.fit(config, x, arch.target(forecasts), forecasts, seed)
    exported = dict(direct=arch.export_component(model, names))
    if indices is not None:
        exported["input_indices"] = indices
    if fallback is None:
        fallback, _ = train.fit(batches, forecasts, seed)
    checkpoint = copy.deepcopy(fallback)
    checkpoint["models"][0] = exported
    expected = np.maximum(1, arch.predict(model, x))
    actual = predict(checkpoint, forecasts)
    error = float(np.max(np.abs(expected - actual)))
    assert error < 2e-6, dict(name=name, export_error_ms=error)
    return checkpoint, dict(
        seconds=seconds,
        export_error_ms=error,
        fit_info=getattr(model, "fit_info", {}),
        tuning=tuning,
    )


def load_pool(live, controlled):
    pool = json.loads(gzip.decompress(live.read_bytes()))
    for role, filename in [
        ("batches", "batches.jsonl"),
        ("forecasts", "forecasts.jsonl"),
    ]:
        for row in pool[role]:
            row["_phase"] = "live_current"
        for line in (controlled / "data" / filename).read_text().splitlines():
            row = json.loads(line)
            if not arch.valid(row, role == "forecasts"):
                continue
            row["_phase"] = "controlled_current"
            pool[role].append(row)
    return pool


def weights(rows):
    values = np.zeros(len(rows))
    for phase in sorted({r["_phase"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["_phase"] == phase]
        w = train.weights([rows[i] for i in indices])
        values[indices] = w / w.sum()
    return values * len(rows) / values.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-state", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arch.weights = weights
    pool = load_pool(args.live, args.controlled)
    batches = [r for r in pool["batches"] if r.get("split", "train") == "train"]
    forecasts = [r for r in pool["forecasts"] if r.get("split", "train") == "train"]
    development = [r for r in pool["forecasts"] if r.get("split") == "development"]
    final = [r for r in pool["forecasts"] if r.get("split") == "final"]
    assert development and final, "Controlled independent holdouts required"
    data = json.dumps(pool, separators=(",", ":")).encode()
    (args.output / "pool.json.gz").write_bytes(gzip.compress(data, mtime=0))
    train.save(
        args.output / "protocol.json",
        dict(
            names=NAMES,
            base=arch.FIXED_BASE,
            correction=arch.FIXED_CORRECTION,
            mlp=MLP,
            batches=len(batches),
            forecasts=len(forecasts),
            development=len(development),
            final=len(final),
            pool_sha256=hashlib.sha256(data).hexdigest(),
            target="critical_end_after_all_rank_start.v1",
            elapsed_mask=21,
            weights="Equal total live-current and controlled-current phase mass; "
            "equal allocation mass within phase, historical workload weighting "
            "within allocation. Preserves the initial live-only refit weights.",
            validation="Three-fold episode purging, train-only OOF base correction; "
            "development for selection, final checked after selection freeze",
            external_shape_calibration=False,
        ),
    )
    groups = np.array([r["episode"] for r in forecasts])
    cv = {}
    with threadpool_limits(limits=1):
        for fold, (it, iv) in enumerate(GroupKFold(3).split(forecasts, groups=groups)):
            tr, va = [forecasts[i] for i in it], [forecasts[i] for i in iv]
            banned = {r["episode"] for r in va}
            tb = [r for r in batches if r["episode"] not in banned]
            assert not banned & {r["episode"] for r in tb + tr}
            fallback = None
            for name in NAMES:
                path = args.output / f"cv_{fold}_{name}.json"
                if path.exists():
                    result = json.loads(path.read_text())
                    if name == "trees_fixed":
                        fallback = json.loads(
                            (
                                args.output / f"cv_{fold}_{name}_checkpoint.json"
                            ).read_text()
                        )
                else:
                    started = time.monotonic()
                    checkpoint, audit = fit(name, tb, tr, 73, fallback)
                    train.save(
                        args.output / f"cv_{fold}_{name}_checkpoint.json", checkpoint
                    )
                    if name == "trees_fixed":
                        fallback = checkpoint
                    score = arch.metrics(arch.target(va), predict(checkpoint, va), va)
                    result = dict(
                        fold=fold,
                        name=name,
                        score=score,
                        audit=audit,
                        seconds=time.monotonic() - started,
                    )
                    train.save(path, result)
                    print(json.dumps(result), flush=True)
                cv.setdefault(name, []).append(result["score"])
        original = json.loads(args.old_state.read_text())
        scores, exports = {}, {}
        fallback = None
        for name in NAMES:
            checkpoint, audit = fit(name, batches, forecasts, 73, fallback)
            if name == "trees_fixed":
                fallback = checkpoint
            state = copy.deepcopy(original)
            state["latency_surface"] = wrap(checkpoint)
            state["latency_surface"]["provenance"].update(
                seed=73, fitting="current_chunk_controlled_and_live_refit"
            )
            path = args.output / f"{name}.json"
            train.save(path, state)
            exports[name] = str(path)
            scores[name] = arch.metrics(
                arch.target(development), predict(checkpoint, development), development
            )
            train.save(args.output / f"{name}_fit_audit.json", audit)
        selected = min(NAMES, key=lambda n: scores[n]["workload_macro_mae_ms"])
        train.save(
            args.output / "selection.json",
            dict(
                selected=selected,
                development=scores,
                cv=cv,
                exports=exports,
                frozen_before_final=True,
            ),
        )
        final_scores = {}
        for name, path in exports.items():
            checkpoint = json.loads(Path(path).read_text())["latency_surface"][
                "checkpoint"
            ]
            final_scores[name] = arch.metrics(
                arch.target(final), predict(checkpoint, final), final
            )
        train.save(args.output / "final_scores.json", final_scores)


if __name__ == "__main__":
    main()
