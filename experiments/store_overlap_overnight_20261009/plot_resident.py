# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot capacity-fitting cache comparisons, grouped by workload."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.store_overlap_overnight_20261009.summarize import BASE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    modes = (
        ("sg2", "CPU cache, sync stores", "#9e9e9e"),
        ("sg3", "CPU cache, async stores", "#2c7fb8"),
        ("sg4", "Resident GPU cache", "#f0a44b"),
    )
    workloads = ("familiar60k", "short16k")
    data = {}
    for root, _, _ in modes:
        summary = json.loads((BASE / root / "summary.json").read_text())
        assert summary["complete"] and summary["stopped"], root
        rows = json.loads((BASE / root / "ceiling_summary.json").read_text())["rows"]
        data[root] = {r["workload"]: r for r in rows}
        assert set(data[root]) == set(workloads), root
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 220, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    metrics = (
        ("throughput_rps", "Throughput (req/s)", True),
        ("mean_ttft_ms", "Mean TTFT (ms)", True),
        ("p95_ttft_ms", "Pooled p95 TTFT (ms)", False),
    )
    width = 0.23
    for axis, (metric, ylabel, repeats) in zip(axes, metrics):
        top = 0
        for mode, (root, label, color) in enumerate(modes):
            positions = [i + (mode - 1) * width for i in range(len(workloads))]
            values = [data[root][w][metric] for w in workloads]
            axis.bar(positions, values, width=width * 0.94, label=label, color=color)
            for x, workload, value in zip(positions, workloads, values):
                points = (
                    [r[metric] for r in data[root][workload]["per_repeat"]]
                    if repeats
                    else [value]
                )
                if repeats:
                    offsets = [
                        (i - (len(points) - 1) / 2) * 0.008 for i in range(len(points))
                    ]
                    axis.scatter(
                        [x + o for o in offsets], points, s=9, color="black", alpha=0.65
                    )
                highest = max(points + [value])
                top = max(top, highest)
                axis.annotate(
                    f"{value:.2f}" if metric == "throughput_rps" else f"{value:.0f}",
                    (x, highest),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        axis.set_xticks(
            range(len(workloads)),
            ["60K prompt\n32 prefixes", "16K prompt\n75 prefixes"],
        )
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, top * 1.2)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.92)
    )
    figure.suptitle("GLM 5.3 TP8: matched working sets that fit in GPU cache")
    figure.text(
        0.5,
        0.015,
        "Dots: episode repeats. Warm priming excluded. Fixed FP8 KV and cache "
        "capacity. GPU mode uses ordinary scheduling; a measured reference.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.84))
    figure.savefig(args.output)
    from pathlib import Path

    figure.savefig(Path(args.output).with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
