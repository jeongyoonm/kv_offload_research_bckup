# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit completed folds and write paired architecture comparisons."""

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

import matplotlib
import numpy as np
import sklearn
from study import LABELS, MODELS, dump, metrics
from threadpoolctl import threadpool_info, threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_predictions(output, names):
    arrays = []
    for name in names:
        with np.load(output / "folds" / name / "predictions.npz") as archive:
            arrays.append({key: archive[key] for key in archive.files})
    return {key: np.concatenate([array[key] for array in arrays]) for key in arrays[0]}


def scores(array):
    rows = [
        {"episode": episode, "workload": workload}
        for episode, workload in zip(array["episode"], array["workload"])
    ]
    return {name: metrics(array["y"], array[name], rows) for name in MODELS}


def paired_bootstrap(array, reference, repeats=2000):
    """Resample episodes, preserving paired forecasts and their sample counts."""
    episodes, inverse = np.unique(array["episode"], return_inverse=True)
    count = np.bincount(inverse)
    reference_error = np.abs(array[reference] - array["y"])
    rng = np.random.default_rng(20261007)
    draws = rng.integers(0, len(episodes), (repeats, len(episodes)))
    denominator = count[draws].sum(axis=1)
    result = {}
    for name in MODELS:
        delta = np.abs(array[name] - array["y"]) - reference_error
        numerator = np.bincount(inverse, weights=delta)
        boot = numerator[draws].sum(axis=1) / denominator
        result[name] = {
            "difference_ms": float(np.mean(delta)),
            "lower95_ms": float(np.quantile(boot, 0.025)),
            "upper95_ms": float(np.quantile(boot, 0.975)),
            "resampled_episodes": len(episodes),
        }
    return result


def audit(output, tasks):
    """Check split coverage, persisted invariants, and numerical agreement."""
    maximum_error, cross_fit_splits, fits_seconds = 0.0, 0, 0.0
    tuning_seconds, trials, warnings, selected_warnings = 0.0, 0, 0, 0
    folds = {}
    for task in tasks:
        folder = output / "folds" / task["name"]
        result = json.loads((folder / "completed.json").read_text())
        split = result["audit"]
        assert split["test_training_episode_overlap"] == 0
        assert split["test_training_batch_overlap"] == 0
        assert split["inner_validation_episode_overlap"] == 0
        assert split["test_forecasts"] == len(task["test"])
        assert not set(task["train"]).intersection(task["test"])
        fits_seconds += result["seconds"]
        maximum_error = max(
            maximum_error, max(result["export_equivalence_max_ms"].values())
        )
        cross = json.loads((folder / "cross_fit_audit.json").read_text())
        for values in cross.values():
            for entry in values:
                assert entry["episode_overlap"] == 0
                cross_fit_splits += 1
        tuning = json.loads((folder / "tuning.json").read_text())
        by_component = {}
        for component, records in tuning["trials"].items():
            assert len(records) == 12
            selected = tuning["selected"][component]
            chosen = next(r for r in records if r["config"] == selected)
            selected_warnings += bool(chosen.get("convergence_warnings"))
            by_component[component] = {
                "selected": selected,
                "selected_validation_macro_mae_ms": chosen["validation_macro_mae_ms"],
                "selected_tuning_fit_seconds": chosen["seconds"],
                "selected_convergence_warnings": chosen.get("convergence_warnings", []),
                "selected_iterations": chosen.get("iterations"),
                "tuning_seconds": sum(r["seconds"] for r in records),
                "trials_with_convergence_warning": sum(
                    bool(r.get("convergence_warnings")) for r in records
                ),
            }
            tuning_seconds += sum(r["seconds"] for r in records)
            trials += len(records)
            warnings += sum(bool(r.get("convergence_warnings")) for r in records)
        folds[task["name"]] = {
            "seconds": result["seconds"],
            "split": split,
            "components": by_component,
        }
        array = load_predictions(output, [task["name"]])
        assert len(array["y"]) == len(task["test"])
        for name in MODELS:
            assert np.all(np.isfinite(array[name]))
            assert np.all(array[name] >= 1)
            assert (
                abs(
                    np.mean(np.abs(array[name] - array["y"]))
                    - result["scores"][name]["mae_ms"]
                )
                < 1e-9
            )
    for regime in ["wide", "matched"]:
        episode = [
            t
            for t in tasks
            if t["name"].startswith("episode_") and t["regime"] == regime
        ]
        indices = [i for task in episode for i in task["test"]]
        assert len(indices) == len(set(indices)) == 3759
        episode_sets = []
        for task in episode:
            a = load_predictions(output, [task["name"]])
            held = set(a["episode"])
            assert all(not held.intersection(other) for other in episode_sets)
            episode_sets.append(held)
    assert maximum_error < 1e-4
    return {
        "folds_completed": len(tasks),
        "cross_fit_splits_without_episode_overlap": cross_fit_splits,
        "max_export_error_ms": maximum_error,
        "summed_fold_wall_seconds": fits_seconds,
        "summed_tuning_fit_seconds": tuning_seconds,
        "tuning_trials": trials,
        "trials_with_convergence_warning": warnings,
        "selected_trials_with_convergence_warning": selected_warnings,
        "fold_details": folds,
    }


def write_csv(output, groups, runtime):
    fields = [
        "evaluation",
        "model",
        "samples",
        "episodes",
        "mae_ms",
        "mape_percent",
        "p95_absolute_ms",
        "workload_macro_mae_ms",
        "worst_workload_mae_ms",
        "bias_ms",
        "under_gt5_percent",
        "over_gt5_percent",
    ]
    with (output / "accuracy.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for group, result in groups.items():
            for name, values in result["scores"].items():
                writer.writerow(
                    {
                        "evaluation": group,
                        "model": name,
                        **{key: values[key] for key in fields[2:]},
                    }
                )
    with (output / "runtime.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(runtime[0]))
        writer.writeheader()
        writer.writerows(runtime)


def accuracy_table(values):
    lines = [
        "| Design | MAE (ms) | MAPE | p95 error (ms) | Macro MAE (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        value = values[name]
        lines.append(
            f"| {LABELS[name]} | {value['mae_ms']:.2f} | "
            f"{value['mape_percent']:.2f}% | {value['p95_absolute_ms']:.2f} | "
            f"{value['workload_macro_mae_ms']:.2f} |"
        )
    return lines


def figure(output, groups, runtime):
    names = [
        "trees_fixed",
        "trees_tuned",
        "tree_base_only",
        "tree_mlp",
        "mlp_tree",
        "mlp_mlp",
        "mlp_direct",
        "tree_direct",
    ]
    short = [
        "Current trees",
        "Tuned trees",
        "Tree base",
        "Tree + MLP",
        "MLP + tree",
        "MLP + MLP",
        "Direct MLP",
        "Direct tree",
    ]
    colors = [
        "#4477AA",
        "#4477AA",
        "#66CCEE",
        "#228833",
        "#228833",
        "#EE6677",
        "#EE6677",
        "#66CCEE",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, group, title in zip(
        axes[:2],
        ["episode_wide", "workload_holdout"],
        ["Unseen episodes (3,759 forecasts)", "Unseen workloads (8 workloads)"],
    ):
        values = groups[group]["scores"]
        y = [values[name]["workload_macro_mae_ms"] for name in names]
        ax.barh(short, y, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("Workload-macro MAE (ms), lower is better")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        for i, value in enumerate(y):
            ax.text(value, i, f" {value:.1f}", va="center", fontsize=9)
        ax.set_xlim(0, max(y) * 1.2)
    timings = {
        r["model"]: r
        for r in runtime
        if r["regime"] == "wide"
        and (r["backend"] == "native_tree" or "tree" not in r["model"])
    }
    y = [timings[name]["mean_us"] for name in names]
    axes[2].barh(short, y, color=colors)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Mean CPU inference (µs), lower is better")
    axes[2].set_title("Frozen inference, one CPU thread")
    axes[2].grid(axis="x", alpha=0.2)
    axes[2].set_axisbelow(True)
    for i, value in enumerate(y):
        axes[2].text(value, i, f" {value:.1f}", va="center", fontsize=9)
    axes[2].set_xlim(0, max(y) * 1.2)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("GLM5.3 TP8 predictor architecture comparison — existing traces only")
    fig.savefig(output / "architecture_comparison.png", dpi=200)
    fig.savefig(output / "architecture_comparison.pdf")
    plt.close(fig)


def markdown(output, report):
    groups, runtime = report["evaluations"], report["runtime"]
    checks = report["audit"]
    archived = report["archived_reproduction"]
    old_scores = archived["archived_scores"]
    lines = [
        "# CPU predictor model and architecture comparison",
        "",
        "GLM5.3 TP8; existing cached-prefill traces only. All fitting and inference",
        "ran on CPU. No serving predictor was changed and no GPU cells were run.",
        "",
        "## Episode-held-out comparison",
        "",
        "Three disjoint outer episode folds predict each of 3,759 forecasts once.",
        "Entire test episodes are absent from base and correction training. Each",
        "fold tunes on its training episodes only; corrections use cross-fitted",
        "base predictions. The wide base can use additional compute-only batches.",
        "",
        *accuracy_table(groups["episode_wide"]["scores"]),
        "",
        "## Matched-data architecture control",
        "",
        "The base receives exactly the forecast training rows available to the",
        "direct models. This removes the extra-base-data advantage.",
        "",
        *accuracy_table(groups["episode_matched"]["scores"]),
        "",
        "## Workload-held-out comparison",
        "",
        "Eight workloads with at least 60 forecasts are each excluded from both",
        "training stages. For homogeneous tests, mixed-workload training batches",
        "are excluded too. Macro MAE treats the eight test workloads equally.",
        "",
        *accuracy_table(groups["workload_holdout"]["scores"]),
        "",
        "### Per-workload MAE (ms)",
        "",
        (
            "| Held-out workload | Samples | Current trees | Tuned trees | "
            "Tree + MLP | MLP + tree | MLP + MLP | Direct MLP | Direct tree |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["workload_details"].items():
        values = result["scores"]
        cells = [
            f"{values[n]['mae_ms']:.2f}"
            for n in [
                "trees_fixed",
                "trees_tuned",
                "tree_mlp",
                "mlp_tree",
                "mlp_mlp",
                "mlp_direct",
                "tree_direct",
            ]
        ]
        lines.append(
            f"| {name} | {values['trees_fixed']['samples']} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## Replay of the previous evaluation",
        "",
        "These 670 forecasts were previously inspected during predictor",
        "development. This section reproduces the earlier comparison, but",
        "is not a new blind test.",
        "",
        *accuracy_table(groups["historical_wide"]["scores"]),
        "",
        "### Archived checkpoint reproduction",
        "",
        f"The unchanged archived serving checkpoint gives {old_scores['mae_ms']:.5f}",
        f"ms MAE and {old_scores['mape_percent']:.5f}% MAPE, reproducing the previous",
        "report. Refitting with the legacy arithmetic matches that checkpoint",
        f"within {archived['legacy_refit_max_difference_ms']:.8f} ms.",
        "",
        "The comparison runner uses NumPy summation for weight normalization;",
        "the old trainer used Python summation. The normalized weights differ",
        f"by at most {archived['weight_max_difference']:.2e}, but this changes some",
        "tied Extra Trees splits and the freshly refitted historical-replay MAE.",
        "Features and labels are identical. Every newly fitted architecture in",
        "this comparison uses the same NumPy-normalized weights. The archived",
        "checkpoint is kept separate; it would be contaminated as a reference",
        "for pooled episode/workload holdouts because it saw their training rows.",
        "This sensitivity also cautions against interpreting tiny ranking gaps.",
        "",
        "## Paired uncertainty",
        "",
        "2,000 paired bootstrap draws resample whole episodes. Differences",
        "below zero favor the alternative over the current fixed tree recipe.",
        "These intervals describe episode sampling, conditional on these",
        "fitted models; they do not include model refitting or tuning variance.",
        "",
        (
            "| Design | Episode MAE difference (ms), 95% interval | "
            "Workload-held-out MAE difference (ms), 95% interval |"
        ),
        "|---|---:|---:|",
    ]
    for name in MODELS[1:]:
        cells = []
        for group in ["episode_wide", "workload_holdout"]:
            v = groups[group]["paired_vs_current"][name]
            cells.append(
                f"{v['difference_ms']:+.2f} "
                f"[{v['lower95_ms']:+.2f}, {v['upper95_ms']:+.2f}]"
            )
        lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Architecture contrasts on matched data",
        "",
        "Each pair uses the same available training rows. Values are the MAE",
        "of the first design minus the second, with episode-bootstrap intervals.",
        "A negative difference favors the first design.",
        "",
        "| Designs | MAE difference (ms), 95% interval |",
        "|---|---:|",
    ]
    contrasts = [
        ("trees_tuned", "tree_direct"),
        ("trees_tuned", "tree_base_only"),
        ("tree_mlp", "tree_base_only"),
        ("mlp_mlp", "mlp_direct"),
        ("mlp_mlp", "mlp_base_only"),
        ("mlp_tree", "mlp_base_only"),
    ]
    for alternative, reference in contrasts:
        v = groups["episode_matched"]["paired_by_reference"][reference][alternative]
        lines.append(
            f"| {LABELS[alternative]} vs {LABELS[reference]} | "
            f"{v['difference_ms']:+.2f} [{v['lower95_ms']:+.2f}, "
            f"{v['upper95_ms']:+.2f}] |"
        )
    lines += [
        "",
        "## Frozen CPU inference",
        "",
        "Historical-wide checkpoints; 3,350 randomized single-row calls per",
        "entry, 200 warmups, one BLAS/OpenMP thread, after training finishes.",
        "Native entries use the existing CPU tree walker. MLP uses frozen",
        "NumPy dense arrays. Mixed models combine those paths. Feature",
        "extraction, JSON loading, and live-engine contention are excluded.",
        "Initialization measures object construction from decoded state.",
        "",
        "| Design | Backend | Mean (µs) | p95 (µs) | Init (ms) | JSON (KiB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for value in runtime:
        if value["regime"] == "wide":
            lines.append(
                f"| {LABELS[value['model']]} | {value['backend']} | "
                f"{value['mean_us']:.2f} | {value['p95_us']:.2f} | "
                f"{value['init_ms']:.2f} | "
                f"{value['checkpoint_json_bytes'] / 1024:.1f} |"
            )
    lines += [
        "",
        "## Error direction",
        "",
        "Episode-held-out predictions. Early predictions can miss admissions;",
        "late predictions can expose waits. These are prediction errors, not",
        "measured scheduling effects.",
        "",
        (
            "| Design | Mean bias (ms) | Under by >5 ms | Over by >5 ms | "
            "p95 under (ms) | p95 over (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        value = groups["episode_wide"]["scores"][name]
        lines.append(
            f"| {LABELS[name]} | {value['bias_ms']:+.2f} | "
            f"{value['under_gt5_percent']:.1f}% | "
            f"{value['over_gt5_percent']:.1f}% | "
            f"{value['p95_under_ms']:.2f} | {value['p95_over_ms']:.2f} |"
        )
    lines += [
        "",
        "## Training and verification",
        "",
        (
            f"- {checks['folds_completed']} completed outer tasks; "
            f"{checks['tuning_trials']} tuning fits (12 candidates per component)."
        ),
        (
            f"- {checks['cross_fit_splits_without_episode_overlap']} inner/full "
            "cross-fit splits have zero episode overlap."
        ),
        (
            "- All exported predictions match fitted predictions: maximum "
            f"difference {checks['max_export_error_ms']:.8f} ms; gate 0.0001 ms."
        ),
        (
            f"- Summed fold wall time: {checks['summed_fold_wall_seconds']:.1f} s; "
            "tasks overlap on four CPU workers, so this is not elapsed study time."
        ),
        (
            f"- Sum of recorded tuning-fit time: "
            f"{checks['summed_tuning_fit_seconds']:.1f} s."
        ),
        (
            f"- Optimizer convergence warnings in "
            f"{checks['trials_with_convergence_warning']} tuning trials; "
            f"{checks['selected_trials_with_convergence_warning']} selected trials."
        ),
        "- Selected configurations, iterations, warnings, and component fit",
        "  times are in `summary.json` and each fold's `tuning.json`.",
        "",
        "## Interpretation limits",
        "",
        "The study compares nine model/architecture combinations, including",
        "the full base/correction family crossing and base-only ablations.",
        "Base-only models receive 11 batch features; direct models receive",
        "all 23 features; corrections also receive the base prediction.",
        "Inputs and target scaling are fitted on training rows only.",
        "",
        "Each family gets 12 configurations, with stagewise tuning. This is",
        "equal candidate count rather than equal training time, parameter",
        "count, or an exhaustive neural architecture search. A single inner",
        "validation split and one initialization seed limit tuning stability.",
        "Two-stage designs have separate base and correction searches; a direct",
        "model has only one search. The current fixed recipe is not retuned.",
        "Convergence warnings indicate a fixed optimizer budget may leave",
        "some MLP candidates short of their optimum.",
        "",
        "All underlying data were available during earlier engineering.",
        "Grouped retraining prevents split leakage, but this is retrospective",
        "evidence on one model and TP setting. It cannot establish generality",
        "to other LLMs or predict throughput, TTFT, or recovered admissions",
        "without new serving runs. No architecture is promoted here.",
        "",
        "## Artifacts",
        "",
        f"Data and checkpoints: `{output}`.",
        "Plot: `architecture_comparison.png` and `.pdf`.",
        "Machine-readable results: `summary.json`, `accuracy.csv`, `runtime.csv`.",
        "Code and protocol: `experiments/predictor_architecture_cpu_20261007`.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    tasks = json.loads((output / "tasks.json").read_text())
    checks = audit(output, tasks)
    definition = {
        "historical_wide": ["historical_wide"],
        "historical_matched": ["historical_matched"],
        "episode_wide": [f"episode_{i}_wide" for i in range(3)],
        "episode_matched": [f"episode_{i}_matched" for i in range(3)],
        "workload_holdout": [t["name"] for t in tasks if t.get("workload")],
    }
    groups = {}
    for name, folders in definition.items():
        array = load_predictions(output, folders)
        groups[name] = {
            "scores": scores(array),
            "paired_vs_current": paired_bootstrap(array, "trees_fixed"),
            "paired_by_reference": {
                reference: paired_bootstrap(array, reference)
                for reference in [
                    "tree_direct",
                    "tree_base_only",
                    "mlp_direct",
                    "mlp_base_only",
                ]
            },
            "folds": folders,
        }
    details = {}
    for task in tasks:
        if task.get("workload"):
            array = load_predictions(output, [task["name"]])
            details[task["workload"]] = {"scores": scores(array)}
    reproduction = groups["historical_wide"]["scores"]["trees_fixed"]
    archived = json.loads((output / "reproduction.json").read_text())
    assert abs(reproduction["mae_ms"] - archived["new_refit_scores"]["mae_ms"]) < 1e-9
    checks["archived_checkpoint_reproduced"] = (
        archived["legacy_refit_max_difference_ms"] < 3e-6
    )
    runtime = json.loads((output / "runtime.json").read_text())
    assert len(runtime) == 30
    assert all(r["max_equivalence_error_ms"] < 1e-4 for r in runtime)
    report = {
        "evaluations": groups,
        "workload_details": details,
        "runtime": runtime,
        "audit": checks,
        "archived_reproduction": archived,
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "cpu_only": True,
            "platform": platform.platform(),
            "cpu_affinity_count": len(os.sched_getaffinity(0)),
            "cpu_model": next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "native_tree_sha256": hashlib.sha256(
                Path(
                    "/mnt/localnvme/glm53_predictor_overnight_20261006/"
                    "fast_forest/build/libpredictor_forest_walk.so"
                ).read_bytes()
            ).hexdigest(),
        },
        "code_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ["study.py", "report.py", "verify.py", "reproduce.py"]
        },
        "inventory": json.loads((output / "inventory.json").read_text()),
    }
    with threadpool_limits(limits=1):
        report["environment"]["single_thread_libraries"] = threadpool_info()
    dump(output / "summary.json", report)
    write_csv(output, groups, runtime)
    figure(output, groups, runtime)
    markdown(output, report)
    print(output / "REPORT.md")


if __name__ == "__main__":
    main()
