# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expanded-data, nested CPU architecture comparisons with purged range tests."""

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_predictor_expanded_20261007")
COVERAGE = Path("/mnt/localnvme/glm53_dataset_coverage_20261007")
HIST = Path("/mnt/localnvme/glm53_predictor_overnight_20261006")
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


arch = module(
    "expanded_architecture",
    SOURCE / "experiments/predictor_architecture_cpu_20261007/study.py",
)
legacy_weights = arch.weights


def weights(rows):
    result = np.zeros(len(rows))
    fresh = np.array([r["_source"] == "fresh" for r in rows])
    for mask in [fresh, ~fresh]:
        if mask.any():
            result[mask] = legacy_weights(
                [r for r, include in zip(rows, mask) if include]
            )
    return result


arch.weights = weights


def held(row, name):
    shapes = row["request_shapes"]
    if name == "long_context":
        return max(c for q, c in shapes) >= 49152
    if name == "long_suffix":
        return max(q for q, c in shapes) >= 8192
    if name == "large_batch":
        return len(shapes) >= 8
    if name == "heavy_transfers":
        return row["load_state"][0] >= 4 or (
            "forecast_features" in row and row["forecast_features"][0] >= 4
        )
    raise ValueError(name)


def prepare():
    batches, forecasts, sources = [], [], {}

    def load(path, phase, source):
        rows = arch.read(path)
        for r in rows:
            r.update(_phase=phase, _source=source)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return rows

    batches += load(HIST / "refreshed/data/batches.jsonl", "historical", "historical")
    forecasts += load(
        HIST / "refreshed/data/forecasts.jsonl", "historical", "historical"
    )
    for phase in arch.PHASES:
        rows = load(
            HIST / phase / "frozen_evaluation/forecasts/forecasts.jsonl",
            "legacy_replay",
            "historical",
        )
        rows = [r for r in rows if arch.valid(r, True)]
        batches += rows
        forecasts += rows
    fb = load(COVERAGE / "data/batches.jsonl", "fresh", "fresh")
    ff = load(COVERAGE / "data/forecasts.jsonl", "fresh", "fresh")
    final = [r for r in ff if r["split"] == "final"]
    batches += [r for r in fb if r["split"] in ("train", "development")]
    forecasts += [r for r in ff if r["split"] in ("train", "development")]
    for r in batches + forecasts:
        assert arch.valid(r, "forecast_time_ns" in r)
    assert len({arch.key(r) for r in batches}) == len(batches)
    assert len({arch.key(r) for r in forecasts}) == len(forecasts)
    assert not {r["episode"] for r in final} & {
        r["episode"] for r in batches + forecasts
    }
    assert np.isfinite(arch.runtime_features(forecasts)).all()
    assert np.isfinite(arch.features(batches)).all()
    assert final
    ROOT.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / "pool.joblib"
    joblib.dump(dict(batches=batches, forecasts=forecasts), dataset, compress=3)
    joblib.dump(final, ROOT / "locked_benchmark.joblib", compress=3)
    tasks = []

    def add(name, train, test, scope=None):
        pool = dataset
        if scope:
            pool = ROOT / f"pool-{scope}.joblib"
            kept = [r for r in batches if not held(r, scope)]
            assert not any(held(r, scope) for r in kept)
            joblib.dump(dict(batches=kept, forecasts=forecasts), pool, compress=3)
        for regime in ["wide", "matched"]:
            for seed in [73, 74, 75]:
                tasks.append(
                    dict(
                        name=f"{name}_{regime}_s{seed}",
                        train=list(map(int, train)),
                        test=list(map(int, test)),
                        regime=regime,
                        seed=seed,
                        dataset=str(pool),
                        output=str(ROOT),
                        save_checkpoints=True,
                        range_scope=scope,
                    )
                )

    groups = np.array([r["episode"] for r in forecasts])
    for i, (train, test) in enumerate(
        GroupKFold(n_splits=3).split(np.arange(len(forecasts)), groups=groups)
    ):
        add(f"episode_{i}", train, test)
    for scope in ["long_context", "long_suffix", "large_batch", "heavy_transfers"]:
        mask = np.array([held(r, scope) for r in forecasts])
        assert mask.any() and (~mask).any()
        test_episodes = {r["episode"] for r, include in zip(forecasts, mask) if include}
        train_mask = np.array(
            [
                not include and r["episode"] not in test_episodes
                for r, include in zip(forecasts, mask)
            ]
        )
        assert train_mask.any()
        add("range_" + scope, np.flatnonzero(train_mask), np.flatnonzero(mask), scope)
    arch.dump(ROOT / "tasks.json", tasks)
    arch.dump(
        ROOT / "inventory.json",
        dict(
            batches=len(batches),
            forecasts=len(forecasts),
            benchmark=len(final),
            benchmark_training_episode_overlap=0,
            source_counts=dict(Counter(r["_phase"] for r in forecasts)),
            hashes=sources,
            feature_names=arch.surface.FEATURE_NAMES + arch.surface.FORECAST_NAMES,
            weighting="source_mass_matched",
            previous_final_is_reused_benchmark=True,
            cpu_only=True,
        ),
    )
    return tasks


def run_task(task):
    with threadpool_limits(limits=1):
        return arch.run_fold(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = (
        json.loads((ROOT / "tasks.json").read_text())
        if (ROOT / "tasks.json").exists()
        else prepare()
    )
    if args.prepare_only:
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {executor.submit(run_task, task): task["name"] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            print(
                json.dumps(dict(event="task_completed", task=result["name"])),
                flush=True,
            )


if __name__ == "__main__":
    main()
