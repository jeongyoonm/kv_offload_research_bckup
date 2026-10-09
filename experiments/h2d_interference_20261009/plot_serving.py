# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot small serving validation; show episode spread without confidence claims."""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {"svg.hashsalt": "h2dserving20261009", "pdf.fonttype": 42, "font.size": 10}
    )
    root = args.root
    data = json.loads((root / "results.json").read_text())["serving"]
    workloads = ("short16k", "familiar60k", "heterogeneous")
    chunks = (128, 8, 16)
    colors = ("#777777", "#0072B2", "#E69F00")
    metrics = (
        ("throughput_rps", "Throughput (requests/s) ↑"),
        ("mean_ttft_ms", "Mean TTFT (ms) ↓"),
        ("p95_ttft_ms", "Mean episode p95 TTFT (ms) ↓"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for axis, (metric, label) in zip(axes, metrics):
        for i, (chunk, color) in enumerate(zip(chunks, colors)):
            selected = {r["workload"]: r for r in data if r["chunk_mib"] == chunk}
            x = [j + (i - 1) * 0.25 for j in range(len(workloads))]
            axis.bar(
                x,
                [selected[w][metric] for w in workloads],
                width=0.23,
                color=color,
                label=f"{chunk} MiB / depth 2",
            )
            raw = json.loads((root / f"serving_{chunk}mib/summary.json").read_text())
            for j, workload in enumerate(workloads):
                ys = [r[metric] for r in raw["results"] if r["workload"] == workload]
                axis.scatter([x[j]] * len(ys), ys, color="black", s=12, zorder=5)
        axis.set_xticks(range(3), ["16K cached", "60K cached", "Mixed"])
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3)
    fig.suptitle("GLM 5.3 TP8 · unchanged latest scheduler / preparation policy")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.23, top=0.79, wspace=0.35)
    fig.text(
        0.5,
        0.05,
        "Two matched 32-request episodes per workload/config; "
        "concurrency 32; one output token. Bars: episode means; dots: episodes.\n"
        "Separate allocations; small validation, not a continuous 300-request "
        "study or a statistical equivalence test.",
        ha="center",
        fontsize=9,
    )
    for extension in ("png", "pdf", "svg"):
        fig.savefig(root / f"serving_comparison.{extension}", dpi=200)


if __name__ == "__main__":
    main()
