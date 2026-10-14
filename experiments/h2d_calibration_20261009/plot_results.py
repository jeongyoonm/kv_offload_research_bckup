# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the recorded H2D sweep and strict confirmation without GPU work."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROFILES = ("4k", "16k", "60k")
TITLES = (
    "4K prefix · 4,096 tokens\n195 MB per load per GPU",
    "16K prefix · 15,552 tokens\n742 MB per load per GPU",
    "60K prefix · 58,368 tokens\n2.784 GB per load per GPU",
)
COLORS = ("#707070", "#0072B2", "#D55E00", "#009E73")
BASELINE = (128, 2)


def save(fig, output: Path, name: str) -> None:
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=220)
    plt.close(fig)


def load(root: Path, stage: str, cells: int, trials: int) -> dict:
    data = json.loads((root / stage / "comparison.json").read_text())
    assert data["complete"]
    assert data["measured_all_rank_cells"] == cells
    assert all(row["trials"] == trials for row in data["aggregate"])
    assert (data["baseline"]["chunk_mib"], data["baseline"]["depth"]) == BASELINE
    return data


def grid(axis) -> None:
    axis.grid(axis="y", color="#dedede", linewidth=0.6)
    axis.set_axisbelow(True)


def plot_sweep(data: dict, output: Path) -> None:
    rows = {(r["profile"], r["chunk_mib"], r["depth"]): r for r in data["aggregate"]}
    chunks = sorted({r["chunk_mib"] for r in data["aggregate"]})
    depths = sorted({r["depth"] for r in data["aggregate"]})
    assert len(rows) == 84 and len(chunks) == 7 and len(depths) == 4
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    metrics = ("effective_per_gpu_gbps", "mean_first_job_all_rank_ms")
    for col, (profile, title) in enumerate(zip(PROFILES, TITLES)):
        axes[0, col].set_title(title, fontsize=11, pad=12)
        for row, metric in enumerate(metrics):
            axis = axes[row, col]
            for depth, color in zip(depths, COLORS):
                values = [rows[profile, chunk, depth][metric] for chunk in chunks]
                axis.plot(chunks, values, marker="o", ms=4, color=color, lw=1.8)
            base = rows[(profile, *BASELINE)][metric]
            axis.scatter([128], [base], marker="*", s=170, c="black", zorder=5)
            axis.set_xscale("log", base=2)
            axis.set_xticks(chunks, chunks)
            axis.set_xlabel("Nominal chunk size (MiB)")
            axis.set_ylabel(
                "Effective bandwidth per GPU (GB/s) ↑"
                if row == 0
                else "First-load all-rank readiness (ms) ↓"
            )
            grid(axis)
            if row == 0:
                axis.set_ylim(23, 49)
            else:
                axis.set_ylim(bottom=0)
    handles = [
        Line2D([], [], color=color, marker="o", label=f"Depth {depth}")
        for depth, color in zip(depths, COLORS)
    ] + [
        Line2D(
            [],
            [],
            color="black",
            marker="*",
            linestyle="none",
            ms=12,
            label="Current: 128 MiB / depth 2",
        )
    ]
    fig.suptitle(
        "Full transfer-only sweep · GLM 5.3 KV geometry · eight B300 GPUs", y=0.98
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=5)
    fig.subplots_adjust(
        left=0.065, right=0.985, top=0.81, bottom=0.115, hspace=0.37, wspace=0.3
    )
    fig.text(
        0.5,
        0.025,
        "28 configurations × 3 prefix sizes × 5 repeats; four FIFO loads per rank. "
        "No model compute.\n"
        "Screening validation starts after rank-local completion; "
        "use the separate strict confirmation for final comparisons.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save(fig, output, "full_sweep")


def plot_confirmation(data: dict, output: Path) -> None:
    configs = (BASELINE, (64, 2), (16, 2), (16, 8), (8, 4))
    labels = ("128 / 2\ncurrent", "64 / 2", "16 / 2", "16 / 8", "8 / 4")
    colors = ("#707070", "#56B4E9", "#0072B2", "#009E73", "#D55E00")
    rows = {(r["profile"], r["chunk_mib"], r["depth"]): r for r in data["aggregate"]}
    contrasts = {
        (r["profile"], r["chunk_mib"], r["depth"]): r for r in data["paired_contrasts"]
    }
    assert len(rows) == 15
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.1))
    x = np.arange(len(configs))
    for col, (profile, title) in enumerate(zip(PROFILES, TITLES)):
        axes[0, col].set_title(title, fontsize=11, pad=12)
        for row, metric in enumerate(
            ("effective_per_gpu_gbps", "mean_first_job_all_rank_ms")
        ):
            axis = axes[row, col]
            values = [rows[(profile, *config)][metric] for config in configs]
            bars = axis.bar(x, values, color=colors, width=0.67)
            axis.bar_label(
                bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=9
            )
            axis.set_ylim(0, max(values) * 1.18)
            axis.set_ylabel(
                "Effective bandwidth per GPU (GB/s) ↑"
                if row == 0
                else "First-load all-rank readiness (ms) ↓"
            )
        axis = axes[2, col]
        gains, lower, upper = [], [], []
        for config in configs:
            if config == BASELINE:
                value, bounds = 0.0, (0.0, 0.0)
            else:
                contrast = contrasts[(profile, *config)]
                value = contrast["paired_gain_percent"]
                bounds = contrast["paired_95pct_bounds"]
            gains.append(value)
            lower.append(value - bounds[0])
            upper.append(bounds[1] - value)
        axis.bar(x, gains, color=colors, width=0.67)
        axis.errorbar(
            x, gains, yerr=[lower, upper], fmt="none", color="black", capsize=3, lw=1
        )
        for i, (value, lo, hi) in enumerate(zip(gains, lower, upper)):
            axis.text(
                i,
                value + (hi + 1.0 if value >= 0 else -lo - 0.7),
                f"{value:+.2f}%",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
        axis.axhline(0, color="black", lw=0.8)
        axis.set_ylim((-5, 37) if col == 0 else (-7.5, 4))
        axis.set_ylabel("Paired bandwidth gain vs. current (%) ↑")
        for axis in axes[:, col]:
            axis.set_xticks(x, labels)
            axis.set_xlabel("Chunk (MiB) / depth")
            grid(axis)
    fig.suptitle(
        "Final strict confirmation · eight B300 GPUs · no model compute", y=0.98
    )
    fig.subplots_adjust(
        left=0.065, right=0.985, top=0.89, bottom=0.12, hspace=0.5, wspace=0.3
    )
    fig.text(
        0.5,
        0.025,
        "Five preselected configurations × 3 sizes × 8 repeats; "
        "validation waits for all ranks to finish timed copies.\n"
        "Error bars: nominal paired 95% intervals, "
        "unadjusted for multiple comparisons. "
        "Transfer bandwidth is not serving throughput.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save(fig, output, "confirmation_results")


def main() -> None:
    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "svg.hashsalt": "h2dcal20261009",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    sweep = load(args.root, "sweep", cells=420, trials=5)
    confirmation = load(args.root, "confirmation_strict", cells=120, trials=8)
    plot_sweep(sweep, output)
    plot_confirmation(confirmation, output)
    for stage, data in (("sweep", sweep), ("confirmation_strict", confirmation)):
        with (output / f"{stage}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(data["aggregate"][0]))
            writer.writeheader()
            writer.writerows(data["aggregate"])
    print(f"Plots and numerical tables: {output}")


if __name__ == "__main__":
    main()
