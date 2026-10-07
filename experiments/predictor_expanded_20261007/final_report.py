# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish accuracy, robustness, export latency, and explicit study limits."""

import json
import os
import time
from collections import defaultdict

import improve
import joblib
import matplotlib
import numpy as np
import report
import study
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def variation():
    pool = joblib.load(study.ROOT / "pool.joblib")
    groups = defaultdict(list)
    for row in pool["batches"]:
        key = (tuple(map(tuple, row["request_shapes"])), tuple(row["load_state"]))
        groups[key].append(row)
    values = []
    for (shapes, loads), rows in groups.items():
        if len({r["episode"] for r in rows}) < 3:
            continue
        y = study.arch.target(rows)
        values.append(
            dict(
                requests=len(shapes),
                uncached_tokens=sum(q for q, c in shapes),
                loads=list(loads),
                samples=len(rows),
                episodes=len({r["episode"] for r in rows}),
                median_duration_ms=float(np.median(y)),
                mean_abs_deviation_from_median_ms=float(
                    np.mean(np.abs(y - np.median(y)))
                ),
                p10_p90_ms=np.quantile(y, [0.1, 0.9]).tolist(),
            )
        )
    result = dict(
        groups=sorted(values, key=lambda v: -v["mean_abs_deviation_from_median_ms"]),
        diagnostic_only=True,
        note=(
            "Same execution geometry and dispatch load state can differ in transfer "
            "progress, routing and machine state. This is not an irreducible "
            "error bound."
        ),
        benchmark_used=False,
    )
    study.arch.dump(study.ROOT / "repeated_geometry_variation.json", result)
    return result


def timings(states, x):
    os.environ["VLLM_DEFERRAL_FAST_FOREST"] = str(
        study.HIST / "fast_forest/build/libpredictor_forest_walk.so"
    )
    results = {}
    with threadpool_limits(limits=1):
        for name, state in states.items():
            model = improve.FrozenCandidate(state, native=True)
            reference = improve.FrozenCandidate(state)
            expected = np.array([reference.predict_one(v) for v in x])
            actual = np.array([model.predict_one(v) for v in x])
            discrepancy = float(np.max(np.abs(expected - actual)))
            assert discrepancy < 1e-4
            for i in range(200):
                model.predict_one(x[i % len(x)])
            durations = []
            for seed in range(5):
                for i in np.random.default_rng(73 + seed).permutation(len(x)):
                    start = time.perf_counter_ns()
                    model.predict_one(x[i])
                    durations.append((time.perf_counter_ns() - start) / 1000)
            results[name] = dict(
                median_us=float(np.median(durations)),
                mean_us=float(np.mean(durations)),
                p95_us=float(np.quantile(durations, 0.95)),
                max_equivalence_error_ms=discrepancy,
                calls=len(durations),
                checkpoint_bytes=len(json.dumps(state).encode()),
                native_trees=True,
                numpy_mlp=True,
            )
    return results


def convergence(tasks):
    output = {}
    for task in tasks:
        tuning = json.loads(
            (study.ROOT / "folds" / task["name"] / "tuning.json").read_text()
        )
        output[task["name"]] = dict(
            total_trial_seconds=sum(
                r["seconds"] for trials in tuning["trials"].values() for r in trials
            ),
            mlp_trials=[
                r
                for trials in tuning["trials"].values()
                for r in trials
                if r["config"]["kind"] == "mlp"
            ],
        )
    study.arch.dump(study.ROOT / "convergence.json", output)


def main():
    tasks = json.loads((study.ROOT / "benchmark_tasks.json").read_text())
    selection = json.loads((study.ROOT / "final_selection.json").read_text())
    architecture = json.loads((study.ROOT / "architecture_summary.json").read_text())
    improvements = json.loads((study.ROOT / "improvement_summary.json").read_text())
    proposal = json.loads((study.ROOT / "improvement_plan.json").read_text())
    benchmark = {}
    for regime in ["wide", "matched"]:
        members = [t for t in tasks if t["regime"] == regime]
        data = report.arrays(members)
        scores = report.scores(data)
        paired = {n: report.paired_interval(data, n) for n in study.arch.MODELS}
        extra = []
        for task in members:
            with np.load(
                study.ROOT / "improvements" / task["name"] / "predictions.npz"
            ) as f:
                extra.append({k: f[k] for k in f.files})
        added = {k: np.concatenate([p[k] for p in extra]) for k in extra[0]}
        rows = [
            dict(episode=e, workload=w)
            for e, w in zip(added["episode"], added["workload"])
        ]
        benchmark[regime] = dict(
            architectures=scores,
            architecture_paired=paired,
            improvements={
                n: study.arch.metrics(added["y"], added[n], rows)
                for n in proposal["candidates"]
            },
            improvement_paired={
                n: report.paired_interval(added, n) for n in proposal["candidates"]
            },
        )
    study.arch.dump(study.ROOT / "benchmark_summary.json", benchmark)
    folder = study.ROOT / "improvements/benchmark_wide_s73"
    states = json.loads((folder / "checkpoints.json").read_text())
    with np.load(folder / "predictions.npz") as f:
        x = f["features"][:200]
    runtime = timings(states, x)
    study.arch.dump(study.ROOT / "improvement_runtime.json", runtime)
    architecture_states = json.loads(
        (study.ROOT / "folds/benchmark_wide_s73/checkpoints.json").read_text()
    )
    architecture_runtime = timings(
        {name: improve.wrapped(state) for name, state in architecture_states.items()},
        x,
    )
    study.arch.dump(
        study.ROOT / "architecture_runtime_final.json", architecture_runtime
    )
    diagnostic = variation()
    convergence(json.loads((study.ROOT / "tasks.json").read_text()) + tasks)
    cv = architecture["scores"]["episode:wide"]["fresh"]
    names = study.arch.MODELS
    fig, ax = plt.subplots(figsize=(12, 4.8))
    positions = np.arange(len(names))
    ax.bar(
        positions - 0.19,
        [cv[n]["mae_ms"] for n in names],
        0.38,
        label="Fresh episode validation",
    )
    ax.bar(
        positions + 0.19,
        [benchmark["wide"]["architectures"][n]["mae_ms"] for n in names],
        0.38,
        label="Reused final benchmark",
    )
    ax.set_xticks(positions, names, rotation=35, ha="right")
    ax.set_ylabel("Mean absolute error (ms)")
    ax.set_title("GLM 5.3 TP8: architectures on expanded data")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(study.ROOT / "architecture_comparison.png", dpi=180)
    fig.savefig(study.ROOT / "architecture_comparison.pdf")
    plt.close(fig)
    chosen, model = selection["improvement"], selection["effective_architecture"]
    scopes = ["long_context", "long_suffix", "large_batch", "heavy_transfers"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    positions = np.arange(len(scopes))
    fixed = [
        improvements[f"{s}:wide"]["scores"]["fresh"]["trees_fixed"]["mae_ms"]
        for s in scopes
    ]
    improved = [
        improvements[f"{s}:wide"]["scores"]["fresh"][chosen]["mae_ms"] for s in scopes
    ]
    ax.bar(positions - 0.19, fixed, 0.38, label="Fixed tree recipe")
    ax.bar(positions + 0.19, improved, 0.38, label="Validation-selected predictor")
    ax.set_xticks(
        positions,
        ["Context >=48K", "Suffix >=8K", "Batch >=8", "Transfers >=4"],
    )
    ax.set_ylabel("Mean absolute error (ms)")
    ax.set_title("GLM 5.3 TP8: withheld workload ranges")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(study.ROOT / "range_generalization.png", dpi=180)
    fig.savefig(study.ROOT / "range_generalization.pdf")
    plt.close(fig)
    best = benchmark["wide"]["improvements"][chosen]
    reference = benchmark["wide"]["architectures"]["trees_fixed"]
    lines = [
        "# Expanded predictor study: GLM 5.3 TP8",
        "",
        f"Validation-selected architecture: **{model}**; follow-up: **{chosen}**.",
        "",
        (
            f"Reused benchmark MAE: **{best['mae_ms']:.2f} ms**, versus "
            f"**{reference['mae_ms']:.2f} ms** for the same-data fixed tree recipe. "
            f"MAPE: **{best['mape_percent']:.2f}%**; p95 absolute error: "
            f"**{best['p95_absolute_ms']:.2f} ms**. Median exported evaluation: "
            f"**{runtime[chosen]['median_us']:.1f} us**."
        ),
        "",
        "## Architecture comparison",
        "",
        (
            "| Architecture | Fresh CV MAE | Fresh CV macro MAE | "
            "Reused benchmark MAE | Reused benchmark p95 error | Evaluation (us) |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for n in names:
        score = benchmark["wide"]["architectures"][n]
        lines.append(
            f"| {n} | {cv[n]['mae_ms']:.2f} | "
            f"{cv[n]['workload_macro_mae_ms']:.2f} | {score['mae_ms']:.2f} | "
            f"{score['p95_absolute_ms']:.2f} | "
            f"{architecture_runtime[n]['median_us']:.1f} |"
        )
    lines += [
        "",
        "## Follow-up improvements",
        "",
        (
            "| Candidate | Fresh CV MAE | Fresh CV macro MAE | "
            "Reused benchmark MAE | Median evaluation (us) |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    fresh = improvements["episode:wide"]["scores"]["fresh"]
    for n in proposal["candidates"]:
        b = benchmark["wide"]["improvements"][n]
        lines.append(
            f"| {n} | {fresh[n]['mae_ms']:.2f} | "
            f"{fresh[n]['workload_macro_mae_ms']:.2f} | {b['mae_ms']:.2f} | "
            f"{runtime[n]['median_us']:.1f} |"
        )
    lines += [
        "",
        "## Range exclusion tests",
        "",
        (
            "Each range removes matching geometry and entire held episodes from "
            "training. These tests measure extrapolation, beyond interpolation "
            "among broader training workloads."
        ),
        "",
        "| Withheld range | Fixed-tree MAE | Selected follow-up MAE |",
        "|---|---:|---:|",
    ]
    for scope in ["long_context", "long_suffix", "large_batch", "heavy_transfers"]:
        s = improvements[f"{scope}:wide"]["scores"]["fresh"]
        lines.append(
            f"| {scope} | {s['trees_fixed']['mae_ms']:.2f} | "
            f"{s[chosen]['mae_ms']:.2f} |"
        )
    lines += [
        "",
        "## Controls and limits",
        "",
        (
            "Three seeds, outer episode splits, inner episode tuning, source-weight "
            "controls, matched base-data controls, paired episode-bootstrap "
            "intervals and export equivalence checks are included in the JSON "
            "artifacts. Calibration labels are predictions from complete "
            "architectures trained on other episodes. Reported errors average "
            "three trained seeds; they do not create a new prediction ensemble."
        ),
        "",
        (
            "The benchmark is the previous coverage study's final set, reused "
            "after that study. It did not select architectures or follow-up "
            "changes, but it is not a new blind test. Model choices were written "
            "and hashed before it was opened."
        ),
        "",
        (
            "All follow-up runs are CPU only. No additional GPU allocation was "
            "used. Exported latency excludes feature extraction and scheduler "
            "contention. Replay accuracy does not establish throughput or TTFT "
            "gains. The private CPU cache optimization and the serving preparation "
            "optimization remain part of the original collection path."
        ),
        (
            "The research exports support architecture comparisons. MLP and "
            "linear-trend candidates require a serving integration before GPU "
            "deployment; the production scheduler was not changed by CPU replay."
        ),
        "",
        (
            f"Repeated-geometry diagnostic: {len(diagnostic['groups'])} groups "
            "with at least three episodes. This timing variation is not a proven "
            "irreducible noise floor."
        ),
        "",
        (
            "Each architecture and each follow-up has a separate Git branch with "
            "dataset and checkpoint hashes; see branches.json. Current "
            "dataset-study report: "
            "/mnt/localnvme/glm53_dataset_coverage_20261007/REPORT.md."
        ),
    ]
    (study.ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(dict(selection=selection, benchmark=best, runtime=runtime[chosen])),
        flush=True,
    )


if __name__ == "__main__":
    main()
