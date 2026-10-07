# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize nested architecture validation before opening a reused benchmark."""

import json
import os
import time
from collections import defaultdict

import numpy as np
import study
from threadpoolctl import threadpool_limits


def arrays(tasks):
    values = []
    for task in tasks:
        path = study.ROOT / "folds" / task["name"] / "predictions.npz"
        with np.load(path) as data:
            values.append({k: data[k] for k in data.files})
    return {k: np.concatenate([a[k] for a in values]) for k in values[0]}


def scores(data, mask=None):
    if mask is None:
        mask = np.ones(len(data["y"]), dtype=bool)
    if not mask.any():
        return {}
    rows = [
        dict(episode=e, workload=w)
        for e, w in zip(data["episode"][mask], data["workload"][mask])
    ]
    return {
        name: study.arch.metrics(data["y"][mask], data[name][mask], rows)
        for name in study.arch.MODELS
    }


def paired_interval(data, name, mask=None, macro=False):
    if mask is None:
        mask = np.ones(len(data["y"]), dtype=bool)
    if not mask.any():
        return {}
    episodes, inverse = np.unique(data["episode"][mask], return_inverse=True)
    delta = np.abs(data[name][mask] - data["y"][mask]) - np.abs(
        data["trees_fixed"][mask] - data["y"][mask]
    )
    weight = np.ones(len(delta))
    if macro:
        _, wi, wc = np.unique(
            data["workload"][mask], return_inverse=True, return_counts=True
        )
        weight = 1 / wc[wi]
    count = np.bincount(inverse, weights=weight)
    total = np.bincount(inverse, weights=delta * weight)
    rng = np.random.default_rng(7307)
    draws = rng.integers(len(episodes), size=(2000, len(episodes)))
    values = total[draws].sum(1) / count[draws].sum(1)
    return dict(
        difference_ms=float(np.average(delta, weights=weight)),
        ci95_ms=np.quantile(values, [0.025, 0.975]).tolist(),
        episodes=len(episodes),
        estimand="fixed empirical workload weights" if macro else "sample MAE",
    )


def main():
    tasks = json.loads((study.ROOT / "tasks.json").read_text())
    groups = defaultdict(list)
    for task in tasks:
        p = study.ROOT / "folds" / task["name"] / "completed.json"
        result = json.loads(p.read_text())
        assert result["audit"]["test_training_episode_overlap"] == 0
        assert result["audit"]["inner_validation_episode_overlap"] == 0
        assert max(result["export_equivalence_max_ms"].values()) < 1e-4
        group = task["range_scope"] or "episode"
        groups[(group, task["regime"])].append(task)
    summary, paired, seed_scores = {}, {}, {}
    for (group, regime), ts in groups.items():
        data = arrays(ts)
        key = f"{group}:{regime}"
        summary[key] = dict(
            overall=scores(data),
            fresh=scores(data, data["phase"] == "fresh"),
            historical=scores(data, data["phase"] != "fresh"),
        )
        paired[key] = {
            name: dict(
                overall=paired_interval(data, name),
                fresh=paired_interval(data, name, data["phase"] == "fresh"),
                fresh_macro=paired_interval(
                    data, name, data["phase"] == "fresh", macro=True
                ),
            )
            for name in study.arch.MODELS
        }
        seed_scores[key] = {
            str(seed): scores(arrays([task for task in ts if task["seed"] == seed]))
            for seed in [73, 74, 75]
        }
    cv = summary["episode:wide"]
    limit = cv["historical"]["trees_fixed"]["mae_ms"] * 1.1
    eligible = [
        name for name in study.arch.MODELS if cv["historical"][name]["mae_ms"] <= limit
    ]
    winner = min(eligible, key=lambda n: cv["fresh"][n]["workload_macro_mae_ms"])
    selection = dict(
        winner=winner,
        criterion=(
            "fresh episode-CV workload-macro MAE, historical MAE within 10% "
            "of fixed trees"
        ),
        benchmark_used_for_selection=False,
        historical_limit_ms=limit,
    )
    study.arch.dump(study.ROOT / "architecture_selection.json", selection)
    study.arch.dump(
        study.ROOT / "architecture_summary.json",
        dict(
            scores=summary,
            paired=paired,
            seed_scores=seed_scores,
            seed_aggregation="mean error over three trained models, no new ensemble",
        ),
    )
    os.environ["VLLM_DEFERRAL_FAST_FOREST"] = str(
        study.HIST / "fast_forest/build/libpredictor_forest_walk.so"
    )
    representative = study.ROOT / "folds/episode_0_wide_s73"
    states = json.loads((representative / "checkpoints.json").read_text())
    with np.load(representative / "predictions.npz") as data:
        x, expected = (
            data["features"][:200],
            {n: data[n][:200] for n in study.arch.MODELS},
        )
    runtime = {}
    with threadpool_limits(limits=1):
        for name, state in states.items():
            native = any(v.get("kind") == "tree" for v in state.values())
            model = study.arch.FrozenArchitecture(state, native)
            got = np.array([model.predict_one(v) for v in x])
            assert np.max(np.abs(got - expected[name])) < 1e-4
            for i in range(200):
                model.predict_one(x[i % len(x)])
            durations = []
            for repeat in range(5):
                for i in np.random.default_rng(73 + repeat).permutation(len(x)):
                    begin = time.perf_counter_ns()
                    model.predict_one(x[i])
                    durations.append((time.perf_counter_ns() - begin) / 1000)
            runtime[name] = dict(
                mean_us=float(np.mean(durations)),
                median_us=float(np.median(durations)),
                p95_us=float(np.quantile(durations, 0.95)),
                calls=len(durations),
                json_bytes=len(json.dumps(state).encode()),
                native_tree=native,
                numpy_mlp=True,
            )
    study.arch.dump(study.ROOT / "runtime.json", runtime)
    lines = [
        "# Expanded GLM 5.3 TP8 architecture study",
        "",
        f"Selected using nested episode validation: **{winner}**.",
        "",
        (
            "The current coverage final-test set has not entered selection. "
            "It is a reused benchmark, not a newly collected blind test."
        ),
        "",
        (
            "| Architecture | Fresh CV MAE | Fresh workload-macro MAE "
            "| Historical CV MAE | Median evaluation |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for name in study.arch.MODELS:
        lines.append(
            f"| {name} | {cv['fresh'][name]['mae_ms']:.2f} ms | "
            f"{cv['fresh'][name]['workload_macro_mae_ms']:.2f} ms | "
            f"{cv['historical'][name]['mae_ms']:.2f} ms | "
            f"{runtime[name]['median_us']:.1f} us |"
        )
    lines += [
        "",
        (
            "JSON files include both wide and matched regimes, all range tests, "
            "three seeds, export checks, and paired episode-bootstrap intervals. "
            "Timings exclude feature extraction and engine contention. CPU replay "
            "does not establish throughput or TTFT gains."
        ),
    ]
    (study.ROOT / "ARCHITECTURE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    main()
