# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Open the reused final benchmark only after CPU model choices are frozen."""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import improve
import joblib
import study


def run_task(task):
    improve.configure()
    study.run_task(task)
    return improve.run_task(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selection_path = study.ROOT / "final_selection.json"
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    pool = joblib.load(study.ROOT / "pool.joblib")
    final = joblib.load(study.ROOT / "locked_benchmark.joblib")
    training_size = len(pool["forecasts"])
    training_episodes = {r["episode"] for r in pool["forecasts"] + pool["batches"]}
    assert not training_episodes.intersection(r["episode"] for r in final)
    pool["forecasts"] += final
    path = study.ROOT / "benchmark_pool.joblib"
    joblib.dump(pool, path, compress=3)
    tasks = [
        dict(
            name=f"benchmark_{regime}_s{seed}",
            dataset=str(path),
            output=str(study.ROOT),
            train=list(range(training_size)),
            test=list(range(training_size, training_size + len(final))),
            regime=regime,
            seed=seed,
            save_checkpoints=True,
            range_scope=None,
        )
        for regime in ["wide", "matched"]
        for seed in [73, 74, 75]
    ]
    study.arch.dump(study.ROOT / "benchmark_tasks.json", tasks)
    study.arch.dump(
        study.ROOT / "benchmark_opened.json",
        dict(
            selection_sha256=selection_hash,
            selection=json.loads(selection_path.read_text()),
            reused_final_benchmark=True,
            benchmark_training_episode_overlap=0,
            samples=len(final),
        ),
    )
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            print(
                json.dumps(dict(event="benchmark_complete", task=result["task"])),
                flush=True,
            )
    assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == selection_hash


if __name__ == "__main__":
    main()
