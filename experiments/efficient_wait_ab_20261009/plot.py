# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot grouped waiting ablations from audited serving measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("/mnt/localnvme/glm53_efficient_wait_ab_20261009")


def main():
    report = json.loads((ROOT / "comparison.json").read_text())
    workloads = ["short16k", "familiar60k", "heterogeneous"]
    policies = ["poll", "bridge", "full"]
    labels = ["Plain 2 ms polling", "Hybrid + polling bridge", "Hybrid + coalescing"]
    colors = ["#909090", "#e59538", "#3274a1"]
    groups = {(g["workload"], g["policy"]): g for g in report["aggregate"]}
    x = np.arange(len(workloads))
    width = 0.24
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), layout="constrained")
    for index, (policy, label, color) in enumerate(zip(policies, labels, colors)):
        values = [[], [], []]
        intervals = [[], []]
        for workload in workloads:
            base = groups[workload, "poll"]
            cell = groups[workload, policy]
            if policy == "poll":
                gain, bounds = 0, [0, 0]
            else:
                contrast = next(
                    c
                    for c in report["contrasts"]
                    if c["workload"] == workload
                    and c["reference"] == "poll"
                    and c["treatment"] == policy
                )
                gain, bounds = (
                    contrast["paired_gain_percent"],
                    contrast["paired_95pct_bounds"],
                )
            values[0].append(gain)
            intervals[0].append(gain - bounds[0])
            intervals[1].append(bounds[1] - gain)
            values[1].append(100 * (1 - cell["mean_ttft_ms"] / base["mean_ttft_ms"]))
            values[2].append(cell["waiting"]["total_wait_cpu_ms"] / 75)
        for axis, data in zip(axes, values):
            bars = axis.bar(
                x + (index - 1) * width, data, width, color=color, label=label
            )
            axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        axes[0].errorbar(
            x + (index - 1) * width,
            values[0],
            yerr=intervals,
            fmt="none",
            color="black",
            capsize=3,
            linewidth=0.8,
        )
    titles = [
        "Paired throughput gain vs polling (%)",
        "Mean TTFT reduction vs plain polling (%)",
        "Measured waiting CPU (ms/request)",
    ]
    for axis, title in zip(axes, titles):
        axis.set_title(title, fontsize=10)
        axis.set_xticks(x, ["16K", "60K", "Mixed"])
        axis.axhline(0, color="black", linewidth=0.7)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.margins(y=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[2].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="outside lower center", ncol=3, frameon=False
    )
    fig.suptitle("GLM 5.3 TP8: same sealing rules, different waiting mechanisms")
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(ROOT / f"waiting_ablation.{extension}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
