# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot joint completion, compute submission and KV readiness by fixed workload."""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "svg.hashsalt": "h2dinterference20261009",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    root = args.root
    plan = json.loads((root / "plan.json").read_text())
    data = json.loads((root / "comparison.json").read_text())
    rows = {
        (r["shape"], r["profile"], r["chunk_mib"], r["depth"]): r
        for r in data["aggregate"]
        if r["mode"] == "concurrent"
    }
    chunks = sorted({r[2] for r in rows})
    depths = sorted({r[3] for r in rows})
    shapes = plan["shapes"]
    for metric, label, title in (
        (
            "joint_finish_ms",
            "Both compute and copy cohort ready (ms) ↓",
            "Joint compute / copy completion",
        ),
        (
            "forward_enqueue_ms",
            "Maximum rank forward enqueue time (ms) ↓",
            "Host compute submission including copy enqueue",
        ),
        (
            "first_ready_ms",
            "First-load all-rank readiness (ms) ↓",
            "KV readiness while compute runs",
        ),
    ):
        fig, axes = plt.subplots(
            len(shapes), 3, figsize=(13, 3.3 * len(shapes) + 1.4), squeeze=False
        )
        for i, shape in enumerate(shapes):
            geom = plan["geometry"][shape]
            for j, profile in enumerate(("4k", "16k", "60k")):
                axis = axes[i, j]
                for depth in depths:
                    color = "#707070" if depth == 1 else "#0072B2"
                    ys = [
                        rows[shape, profile, c, depth][metric]
                        if (shape, profile, c, depth) in rows
                        else float("nan")
                        for c in chunks
                    ]
                    axis.plot(
                        chunks, ys, color=color, marker="o", label=f"Depth {depth}"
                    )
                baseline = rows.get((shape, profile, 128, 2))
                if baseline is not None:
                    axis.scatter(
                        [128],
                        [baseline[metric]],
                        marker="*",
                        s=130,
                        c="black",
                        zorder=5,
                    )
                axis.set_xscale("log", base=2)
                axis.set_xticks(chunks, chunks)
                axis.set_xlabel("Nominal chunk size (MiB)")
                axis.set_ylabel(label)
                axis.grid(axis="y", color="#dddddd", lw=0.6)
                axis.set_title(
                    f"{profile.upper()} copy cohort · {shape} compute\n"
                    f"n={geom['batch']}, context={geom['context']:,}, "
                    f"suffix={geom['suffix']:,}",
                    fontsize=10,
                )
        handles = [
            Line2D([], [], color="#707070", marker="o", label="Depth 1"),
            Line2D([], [], color="#0072B2", marker="o", label="Depth 2"),
            Line2D(
                [],
                [],
                color="black",
                marker="*",
                ls="none",
                ms=12,
                label="Current: 128 MiB / depth 2",
            ),
        ]
        fig.suptitle(f"{title} · GLM 5.3 TP8 · eight B300 GPUs", y=0.98)
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3
        )
        fig.subplots_adjust(
            left=0.07, right=0.99, top=0.84, bottom=0.09, hspace=0.65, wspace=0.32
        )
        fig.text(
            0.5,
            0.02,
            f"{plan['repeats']} randomized repeats; four FIFO loads per rank. "
            "Fixed GPU-resident compute cohorts; not serving throughput.\n"
            "Joint completion uses native end-interval upper bounds. "
            "Unique suffix-leading tokens prevent suffix-cache hits.",
            ha="center",
            fontsize=9,
            color="#444444",
        )
        for extension in ("png", "pdf", "svg"):
            fig.savefig(root / f"{metric}.{extension}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
