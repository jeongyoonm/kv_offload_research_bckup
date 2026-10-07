# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot exact-checkpoint offline and common-snapshot live prediction errors."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("/mnt/localnvme/glm53_offset_predictor_20261008")


def main():
    data = json.loads((ROOT / "audit_live/audit.json").read_text())
    labels = ["Reused offline\n(n=192)"]
    columns = {
        "trees": [
            next(
                r["metrics"]["mae_ms"]
                for r in data["offline_benchmark"]
                if r["predictor"] == "trees"
            )
        ]
    }
    for name in ["matched_elapsed", "matched_offset"]:
        columns[name] = [
            next(
                r["metrics"]["mae_ms"]
                for r in data["offline_benchmark"]
                if r["predictor"] == name
            )
        ]
    for name, title in [
        ("familiar60k", "Live 60K"),
        ("short16k", "Live 16K"),
        ("heterogeneous", "Live mixed"),
    ]:
        subset = next(r for r in data["live"] if r["workload"] == name)
        labels.append(f"{title}\n(n={subset['count']})")
        for model in columns:
            columns[model].append(subset["models"][model]["accuracy"]["mae_ms"])
    fig, ax = plt.subplots(figsize=(8.6, 3.9), layout="constrained")
    titles = ["Previous trees", "Matched elapsed input", "Matched offset only"]
    colors = ["#8f98a3", "#3478a5", "#df8439"]
    x = np.arange(len(labels))
    for i, (model, title, color) in enumerate(zip(columns, titles, colors)):
        bars = ax.bar(
            x + (i - 1) * 0.24, columns[model], 0.24, label=title, color=color
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean absolute error (ms)")
    ax.set_ylim(0, max(max(v) for v in columns.values()) * 1.28)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=9)
    ax.set_title("GLM 5.3 TP8: offline versus live decision snapshots", pad=14)
    for extension in ["png", "pdf"]:
        fig.savefig(ROOT / f"prediction_accuracy.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
