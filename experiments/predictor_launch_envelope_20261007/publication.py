# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify exports and publish accuracy, uncertainty, runtime and branches."""

import hashlib
import json
import os
import subprocess
import time

import envelope as e
import envelope_blend as blend
import envelope_evaluate as evaluate
import envelope_guard as guard
import joblib
import matplotlib
import numpy as np
import report
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = e.NAMES + guard.NAMES + list(blend.SOURCES)


def git(*args, cwd=e.SOURCE):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def data(tasks):
    values = blend.arrays(tasks)
    pieces = []
    for task in tasks:
        with np.load(guard.ROOT / "folds" / task["name"] / "predictions.npz") as f:
            pieces.append({n: f[n].copy() for n in guard.NAMES})
    values.update({n: np.concatenate([p[n] for p in pieces]) for n in guard.NAMES})
    return values


def scores(values):
    old = e.NAMES
    try:
        e.NAMES = NAMES
        return evaluate.scores(values)
    finally:
        e.NAMES = old


def model(state, native=True):
    return (
        blend.Frozen
        if "neural" in state
        else guard.Frozen
        if "support" in state
        else e.Frozen
    )(state, native)


def state_set(task):
    directory = e.DATA / "folds" / task["name"]
    states = json.loads(
        (e.ROOT / "folds" / task["name"] / "checkpoints.json").read_text()
    )
    fallback = json.loads((directory / "checkpoints.json").read_text())["trees_fixed"]
    audits = json.loads(
        (guard.ROOT / "folds" / task["name"] / "audit.json").read_text()
    )
    for policy in guard.POLICIES:
        for name in e.NAMES:
            states[f"{policy}_{name}"] = dict(
                support=audits[policy]["support"],
                envelope=states[name],
                fallback=fallback,
            )
    neural = json.loads(
        (e.DATA / "improvements" / task["name"] / "checkpoints.json").read_text()
    )
    for name, source in blend.SOURCES.items():
        states[name] = dict(
            envelope=states["suffix_guard_envelope_boost"],
            neural=neural[source],
            weights=[0.5, 0.5],
        )
    return states


def exports(tasks):
    exported = {n: {} for n in NAMES}
    runtime = {}
    os.environ["VLLM_DEFERRAL_FAST_FOREST"] = str(
        e.expanded.HIST / "fast_forest/build/libpredictor_forest_walk.so"
    )
    pool = joblib.load(e.DATA / "benchmark_pool.joblib")
    for task in [t for t in tasks if t["regime"] == "wide"]:
        rows = [pool["forecasts"][i] for i in task["test"]]
        x = e.features(rows)
        expected = data([task])
        states = state_set(task)
        for name, state in states.items():
            path = blend.ROOT / "exports" / name / f"seed{task['seed']}.json"
            e.ARCH.dump(path, state)
            exported[name][str(task["seed"])] = dict(
                path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            frozen = model(state)
            predicted = np.array([frozen.predict_one(v) for v in x])
            error = float(np.max(np.abs(predicted - expected[name])))
            assert error < 1e-4, (name, error)
            if task["seed"] != 73:
                continue
            for v in x:
                frozen.predict_one(v)
            calls = []
            with threadpool_limits(limits=1):
                for repeat in range(5):
                    for i in np.random.default_rng(7307 + repeat).permutation(len(x)):
                        begin = time.perf_counter_ns()
                        frozen.predict_one(x[i])
                        calls.append((time.perf_counter_ns() - begin) / 1000)
            runtime[name] = dict(
                median_us=float(np.median(calls)),
                p95_us=float(np.quantile(calls, 0.95)),
                export_max_error_ms=error,
                calls=len(calls),
                feature_extraction_included=False,
            )
    extraction = []
    for _ in range(5):
        for row in rows:
            begin = time.perf_counter_ns()
            e.features([row])
            extraction.append((time.perf_counter_ns() - begin) / 1000)
    e.ARCH.dump(blend.ROOT / "exports.json", exported)
    e.ARCH.dump(blend.ROOT / "runtime.json", runtime)
    e.ARCH.dump(
        blend.ROOT / "feature_runtime.json",
        dict(
            median_us=float(np.median(extraction)),
            p95_us=float(np.quantile(extraction, 0.95)),
            full_geometry_and_runtime_features=True,
        ),
    )
    return exported, runtime


def branches(exported):
    base = git("rev-parse", "HEAD")
    staging = blend.ROOT / "branch_staging"
    if not staging.exists():
        git("worktree", "add", "--detach", str(staging), base)
        (staging / ".venv").symlink_to(e.SOURCE / ".venv", target_is_directory=True)
    recorded = {}
    for name in NAMES:
        branch = f"experiment/glm53-envelope-{name.replace('_', '-')}-20261007"
        exists = (
            subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
                cwd=e.SOURCE,
            ).returncode
            == 0
        )
        if not exists:
            git("switch", "-c", branch, base, cwd=staging)
            manifest = dict(
                candidate=name,
                checkpoints=exported[name],
                implementation=base,
                cpu_only=True,
                no_inference_fitting=True,
                benchmark_is_reused=True,
                benchmark_used_for_selection=False,
                serving_integration_required=True,
            )
            path = (
                staging / "experiments/predictor_launch_envelope_20261007/variant.json"
            )
            e.ARCH.dump(path, manifest)
            git("add", str(path.relative_to(staging)), cwd=staging)
            env = dict(
                os.environ, PATH=str(staging / ".venv/bin") + ":" + os.environ["PATH"]
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Codex",
                    "-c",
                    "user.email=noreply@openai.com",
                    "commit",
                    "-m",
                    f"Record envelope predictor {name}",
                    "-m",
                    "Co-authored-by: Codex <noreply@openai.com>",
                    "-m",
                    "Signed-off-by: Codex <noreply@openai.com>",
                ],
                cwd=staging,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        recorded[name] = dict(branch=branch, commit=git("rev-parse", branch))
    e.ARCH.dump(blend.ROOT / "branches.json", recorded)
    git("switch", "--detach", base, cwd=staging)
    git("worktree", "remove", "--force", str(staging))


def main():
    tasks = json.loads((e.DATA / "tasks.json").read_text())
    benchmark_tasks = json.loads((e.DATA / "benchmark_tasks.json").read_text())
    validation = data(
        [t for t in tasks if t["range_scope"] is None and t["regime"] == "wide"]
    )
    final = data([t for t in benchmark_tasks if t["regime"] == "wide"])
    cv, benchmark = scores(validation), scores(final)
    winner = json.loads((blend.ROOT / "selection.json").read_text())["winner"]
    paired = {}
    for reference in ["trees_fixed", "guarded_sparse_mlp", "guarded_sparse_blend"]:
        paired[reference] = dict(
            validation_macro=report.paired_interval(
                dict(validation, trees_fixed=validation[reference]),
                winner,
                validation["phase"] == "fresh",
                macro=True,
            ),
            reused_benchmark=report.paired_interval(
                dict(final, trees_fixed=final[reference]), winner
            ),
        )
    exported, runtime = exports(benchmark_tasks)
    e.ARCH.dump(
        blend.ROOT / "combined_summary.json",
        dict(validation=cv, reused_benchmark=benchmark, paired=paired),
    )
    lines = [
        "# Expanded predictor exploration: launch envelope and fixed blends",
        "",
        f"Frozen blend selection: **{winner}**.",
        "",
        (
            "All accuracy figures below average errors over three "
            "independently fitted seeds; they do not average predictions over "
            "seeds."
        ),
        "",
        (
            "| Candidate | Fresh CV MAE | Fresh macro MAE | Historical MAE | "
            "Reused replay MAE | Replay MAPE | Replay p95 error | Eval median "
            "|"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in evaluate.BASELINES + NAMES:
        f, h, b = cv["fresh"][name], cv["historical"][name], benchmark["overall"][name]
        latency = (
            f"{runtime[name]['median_us']:.1f} us"
            if name in runtime
            else "see expanded study"
        )
        lines.append(
            f"| {name} | "
            f"{f['mae_ms']:.2f} | "
            f"{f['workload_macro_mae_ms']:.2f} | "
            f"{h['mae_ms']:.2f} | "
            f"{b['mae_ms']:.2f} | "
            f"{b['mape_percent']:.2f}% | "
            f"{b['p95_absolute_ms']:.2f} | "
            f"{latency} |"
        )
    lines += [
        "",
        (
            "MAE/error units are milliseconds. Evaluation excludes feature "
            "extraction; its separate cost is in feature_runtime.json."
        ),
        "",
        "## Excluded workload ranges",
        "",
        (
            "| Withheld range | Fixed trees | Guarded sparse MLP | Guarded "
            "envelope | Selected fixed blend |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    range_values = {}
    for scope in ["long_context", "long_suffix", "large_batch", "heavy_transfers"]:
        values = data(
            [t for t in tasks if t["range_scope"] == scope and t["regime"] == "wide"]
        )
        s = scores(values)["fresh"]
        names = [
            "trees_fixed",
            "guarded_sparse_mlp",
            "suffix_guard_envelope_boost",
            winner,
        ]
        range_values[scope] = {n: s[n]["mae_ms"] for n in names}
        lines.append(
            "| "
            + scope
            + " | "
            + " | ".join(f"{s[n]['mae_ms']:.2f} ms" for n in names)
            + " |"
        )
    e.ARCH.dump(blend.ROOT / "range_summary.json", range_values)
    lines += [
        "",
        "## Interpretation and limits",
        "",
        (
            "The workload estimate and submission envelope are latent fitted "
            "components. Matching-shape traces motivated this structure; they "
            "do not prove a CPU/GPU causal decomposition. The uncorrected "
            "envelope performs poorly, and residual boosting supplies much of "
            "the accuracy."
        ),
        "",
        (
            "A fixed equal blend improves validation because envelope and "
            "sparse-MLP errors differ. There is no weight search. The suffix "
            "guard uses training bounds only and same-fold fixed trees outside "
            "support. It retains improvements for context and batch-size "
            "ranges but does not solve suffix extrapolation. Bounds also do "
            "not detect unsupported joint feature combinations."
        ),
        "",
        (
            "The final 192-row, 141-episode set was excluded from fitting and "
            "candidate selection, but has been used by earlier experiments: it "
            "is a reused replay benchmark, not an independent new blind test. "
            "Pairwise episode bootstrap intervals are in "
            "combined_summary.json. Repeated candidate exploration on the same "
            "CV split can itself overfit selection."
        ),
        "",
        (
            "No new GPU runs were performed after the four-hour collection "
            "cap. CPU prediction gains do not establish serving throughput, "
            "TTFT, or admission gains. Serving integration and a new blind "
            "workload evaluation are required before deployment."
        ),
        "",
        "## Artifacts",
        "",
        f"Implementation: `{e.SOURCE}`.",
        "",
        (
            "Exports, hashes, branch inventory, validation and benchmark "
            "summaries, and runtime measurements are beside this report. All "
            "fourteen candidates have three exported checkpoints and separate "
            "Git branches."
        ),
    ]
    (blend.ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    selected = [
        "trees_fixed",
        "guarded_sparse_mlp",
        "suffix_guard_envelope_boost",
        winner,
    ]
    labels = ["Fixed trees", "Sparse MLP + guard", "Envelope + guard", "Equal blend"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].bar(
        labels,
        [cv["fresh"][n]["mae_ms"] for n in selected],
        color=["#777777", "#4477aa", "#ee9944", "#228833"],
    )
    axes[1].bar(
        labels,
        [benchmark["overall"][n]["mae_ms"] for n in selected],
        color=["#777777", "#4477aa", "#ee9944", "#228833"],
    )
    for ax, title in zip(axes, ["Fresh episode CV", "Reused final replay"]):
        ax.set_title(title)
        ax.set_ylabel("Mean absolute error (ms)")
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(blend.ROOT / "predictor_comparison.png", dpi=180)
    fig.savefig(blend.ROOT / "predictor_comparison.pdf")
    plt.close(fig)
    branches(exported)
    e.ARCH.dump(
        blend.ROOT / "status.json",
        dict(phase="complete", finished=time.time(), serving_integration_required=True),
    )
    print("publication_complete", flush=True)


if __name__ == "__main__":
    main()
