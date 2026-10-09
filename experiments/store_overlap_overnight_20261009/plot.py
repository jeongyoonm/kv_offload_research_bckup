# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot measured serving rates and gaps in separate workload panels."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.store_overlap_overnight_20261009.summarize import BASE


def draw(ax, names, labels, metric, ylabel, color):
    means, observations = [], []
    for name in names:
        path = BASE / name / "comparison.json"
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        means.append(next(iter(data["means"].values()))[metric])
        observations.append([row[metric] for row in data["episodes"]])
    ax.bar(range(len(names)), means, color=color, alpha=0.8, width=0.62)
    for x, (mean, points) in enumerate(zip(means, observations)):
        offsets = [(i - (len(points) - 1) / 2) * 0.06 for i in range(len(points))]
        ax.scatter([x + offset for offset in offsets], points, s=18, color="black")
        ax.annotate(
            f"{mean:.2f}" if metric == "throughput_rps" else f"{mean:.0f}",
            (x, max(mean, max(points))),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(range(len(names)), labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(max(row) for row in observations) * 1.18)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 220})
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    groups = [
        (["sn0", "snf", "sn8"], ["Sync", "Async full", "Async 8 MiB"]),
        (
            ["st0", "st1", "stm", "stc"],
            ["Block major", "Tensor major", "Merge", "Merge +\nCPU order"],
        ),
    ]
    for column, (names, labels) in enumerate(groups):
        assert draw(
            axes[0, column],
            names,
            labels,
            "throughput_rps",
            "Throughput (req/s)",
            "#2c7fb8",
        ), "Wait for all planned measurements"
        assert draw(
            axes[1, column],
            names,
            labels,
            "inter_compute_gap_ms",
            "Inter-compute gaps (ms/episode)",
            "#f0a44b",
        )
    axes[0, 0].set_title("75-request episodes: store submission")
    axes[0, 1].set_title("Continuous 300 requests: cache layout")
    figure.suptitle("GLM 5.3 TP8, 60K prompts, 95% CPU prefix restoration")
    figure.text(
        0.5,
        0.012,
        "Bars: episode means. Dots: individual repeats. Panels use distinct "
        "episode sizes; no statistical significance claim.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    figure.savefig(args.output)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
