# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Geometry exclusion versus a source/count matched episode control."""

import argparse
import importlib
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "experiments/predictor_launch_envelope_20261007"))
e = importlib.import_module("envelope")
guard = importlib.import_module("envelope_guard")
ROOT = Path("/mnt/localnvme/glm53_geometry_control_20261007")
EXTRA = [
    "guarded_sparse_mlp",
    "envelope_boost",
    "suffix_guard_envelope_boost",
    "envelope_sparse_mlp",
]


def key(row):
    return json.dumps(
        [sorted(row["request_shapes"]), row["load_state"]], separators=(",", ":")
    )


def category(row):
    return row["_source"], row["_phase"]


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    pool = joblib.load(e.DATA / "pool.joblib")
    forecasts = pool["forecasts"]
    original = json.loads((e.DATA / "tasks.json").read_text())
    tasks, audits = [], {}
    for fold, original_task in enumerate(
        t
        for t in original
        if t["range_scope"] is None and t["seed"] == 73 and t["regime"] == "matched"
    ):
        test = original_task["test"]
        banned = {key(forecasts[i]) for i in test}
        strict = [i for i in original_task["train"] if key(forecasts[i]) not in banned]
        counts = Counter(category(forecasts[i]) for i in strict)
        rng = np.random.default_rng(73070 + fold)
        control = []
        for source, count in sorted(counts.items()):
            available = [
                i for i in original_task["train"] if category(forecasts[i]) == source
            ]
            control.extend(map(int, rng.choice(available, size=count, replace=False)))
        assert Counter(category(forecasts[i]) for i in control) == counts
        for mode, train in [
            ("geometry_excluded", strict),
            ("count_matched", sorted(control)),
        ]:
            assert not {forecasts[i]["episode"] for i in train} & {
                forecasts[i]["episode"] for i in test
            }
            overlap = {key(forecasts[i]) for i in train} & banned
            assert mode != "geometry_excluded" or not overlap
            audits[f"fold{fold}:{mode}"] = dict(
                train_rows=len(train),
                test_rows=len(test),
                geometry_overlap=len(overlap),
                episode_overlap=0,
                source_counts={str(k): v for k, v in counts.items()},
            )
            for seed in [73, 74, 75]:
                tasks.append(
                    dict(
                        name=f"geometry_{fold}_{mode}_s{seed}",
                        train=train,
                        test=test,
                        seed=seed,
                        regime="matched",
                        output=str(ROOT),
                        dataset=str(e.DATA / "pool.joblib"),
                        range_scope=None,
                        mode=mode,
                        save_checkpoints=True,
                    )
                )
    e.ARCH.dump(ROOT / "tasks.json", tasks)
    e.ARCH.dump(ROOT / "split_audit.json", audits)
    return tasks


def run(task):
    started = time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    directory = ROOT / "followups" / task["name"]
    if (directory / "completed.json").exists():
        return json.loads((directory / "completed.json").read_text())
    with threadpool_limits(limits=1):
        e.improve.configure()
        e.ARCH.run_fold(task)
        pool = joblib.load(task["dataset"])
        train, test = (
            [pool["forecasts"][i] for i in task[k]] for k in ["train", "test"]
        )
        fold = ROOT / "folds" / task["name"]
        selected = json.loads((fold / "tuning.json").read_text())["selected"]
        fixed = json.loads((fold / "checkpoints.json").read_text())["trees_fixed"]
        sparse_states, _ = e.sparse.fit_variants(train, train, selected, task["seed"])
        x = e.features(test)
        nx = np.column_stack((x[:, :23], np.zeros((len(x), 5)), x[:, 23:26]))
        old_guard = importlib.import_module("guard")
        neural = old_guard.states_for(train, sparse_states["sparse_mlp"], fixed)[
            "guarded_sparse_mlp"
        ]
        neural_frozen = e.improve.FrozenCandidate(neural)
        neural_prediction = np.array([neural_frozen.predict_one(v) for v in nx])
        states, audit = e.fit_variants(train, selected, task["seed"])
        envelope = states["envelope_boost"]
        guarded = dict(
            support=guard.bounds(train, [5]), envelope=envelope, fallback=fixed
        )
        guarded_model = guard.Frozen(guarded)
        guarded_prediction = np.array([guarded_model.predict_one(v) for v in x])
        predictions = dict(
            guarded_sparse_mlp=neural_prediction,
            envelope_boost=e.predict(envelope, test),
            suffix_guard_envelope_boost=guarded_prediction,
            envelope_sparse_mlp=(neural_prediction + guarded_prediction) / 2,
        )
    with np.load(fold / "predictions.npz") as f:
        metadata = {k: f[k] for k in ["y", "episode", "workload", "phase"]}
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "predictions.npz", **metadata, **predictions)
    e.ARCH.dump(
        directory / "checkpoints.json",
        dict(
            guarded_sparse_mlp=neural,
            envelope_boost=envelope,
            suffix_guard_envelope_boost=guarded,
            envelope_sparse_mlp=dict(
                envelope=guarded, neural=neural, weights=[0.5, 0.5]
            ),
        ),
    )
    result = dict(
        task=task["name"],
        seconds=time.perf_counter() - started,
        training_episode_overlap=0,
        envelope_audit=audit,
    )
    e.ARCH.dump(directory / "completed.json", result)
    print(json.dumps(dict(event="geometry_completed", **result)), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    tasks = (
        json.loads((ROOT / "tasks.json").read_text())
        if (ROOT / "tasks.json").exists()
        else prepare()
    )
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        for future in as_completed([executor.submit(run, t) for t in tasks]):
            future.result()
