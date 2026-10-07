# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze on episode validation, then evaluate a reused benchmark."""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import envelope as e
import numpy as np
import report

BASELINES = ["trees_fixed", "selected", "guarded_sparse_mlp", "guarded_sparse_blend"]


def arrays(tasks, benchmark=False):
    pieces = []
    for task in tasks:
        with np.load(e.ROOT / "folds" / task["name"] / "predictions.npz") as f:
            data = {k: f[k] for k in f.files}
        with np.load(e.DATA / "improvements" / task["name"] / "predictions.npz") as f:
            np.testing.assert_array_equal(data["y"], f["y"])
            np.testing.assert_array_equal(data["episode"], f["episode"])
            data.update({n: f[n].copy() for n in BASELINES})
        pieces.append(data)
    return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}


def scores(data):
    result = {}
    for phase, mask in [
        ("overall", np.ones(len(data["y"]), dtype=bool)),
        ("fresh", data["phase"] == "fresh"),
        ("historical", data["phase"] != "fresh"),
    ]:
        rows = [
            dict(episode=a, workload=b)
            for a, b in zip(data["episode"][mask], data["workload"][mask])
        ]
        result[phase] = (
            {
                n: e.ARCH.metrics(data["y"][mask], data[n][mask], rows)
                for n in e.NAMES + BASELINES
            }
            if rows
            else {}
        )
    return result


def summarize():
    tasks = json.loads((e.DATA / "tasks.json").read_text())
    summary = {}
    for group in sorted(
        {f"{t['range_scope'] or 'episode'}:{t['regime']}" for t in tasks}
    ):
        members = [
            t
            for t in tasks
            if f"{t['range_scope'] or 'episode'}:{t['regime']}" == group
        ]
        data = arrays(members)
        summary[group] = dict(
            scores=scores(data),
            paired={
                n: report.paired_interval(data, n, data["phase"] == "fresh", macro=True)
                for n in e.NAMES
            },
            seeds={
                str(s): scores(arrays([t for t in members if t["seed"] == s]))
                for s in [73, 74, 75]
            },
        )
    cv = summary["episode:wide"]["scores"]
    limit = cv["historical"]["trees_fixed"]["mae_ms"] * 1.1
    eligible = [n for n in e.NAMES if cv["historical"][n]["mae_ms"] <= limit]
    winner = (
        min(eligible, key=lambda n: cv["fresh"][n]["workload_macro_mae_ms"])
        if eligible
        else None
    )
    e.ARCH.dump(e.ROOT / "validation_summary.json", summary)
    e.ARCH.dump(
        e.ROOT / "selection.json",
        dict(
            winner=winner,
            criterion=(
                "fresh episode macro MAE with <=10% historical "
                "regression against fixed trees"
            ),
            benchmark_used_for_selection=False,
            benchmark_is_reused=True,
            historical_limit_ms=limit,
        ),
    )
    for n in e.NAMES + BASELINES:
        print(
            n,
            {p: round(cv[p][n]["mae_ms"], 3) for p in ["fresh", "historical"]},
            "macro",
            round(cv["fresh"][n]["workload_macro_mae_ms"], 3),
            flush=True,
        )
    print("selected", winner, flush=True)
    return winner


def benchmark():
    selection = e.ROOT / "selection.json"
    checksum = hashlib.sha256(selection.read_bytes()).hexdigest()
    tasks = json.loads((e.DATA / "benchmark_tasks.json").read_text())
    # run_task fits only task[train]; the final rows remain exclusively task[test].
    e.ARCH.dump(
        e.ROOT / "benchmark_opened.json",
        dict(
            selection_sha256=checksum,
            benchmark_is_reused=True,
            training_episode_overlap=0,
        ),
    )
    with ProcessPoolExecutor(
        max_workers=6, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        for future in as_completed(
            [executor.submit(e.run_task, task) for task in tasks]
        ):
            future.result()
    assert hashlib.sha256(selection.read_bytes()).hexdigest() == checksum
    result = {
        r: scores(arrays([t for t in tasks if t["regime"] == r]))
        for r in ["wide", "matched"]
    }
    e.ARCH.dump(e.ROOT / "benchmark_summary.json", result)
    for n in e.NAMES + BASELINES:
        print(
            "benchmark", n, round(result["wide"]["overall"][n]["mae_ms"], 3), flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.benchmark:
        benchmark()
    else:
        summarize()
