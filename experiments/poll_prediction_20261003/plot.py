"""Plot polling curves without mixing confirmation and screening cohorts."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CASES = [
    ("p8192_f95_h100", "8K prompt, 95% cached"),
    ("p61440_f95_h100", "60K prompt, 95% cached"),
    ("p61440_f99_h100", "60K prompt, 99% cached"),
]


def save(fig, directory, name):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot(root):
    out = root / "summary"
    rows = json.loads((out / "cells.json").read_text())
    lookup = {(r["phase"], r["workload"], r["policy"]): r for r in rows}
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    intervals = [1, 2, 5, 10]
    for ax, (case, title) in zip(axes, CASES):
        for family, color in (("readiness", "#2266aa"), ("prediction", "#cc6633")):
            values = [
                lookup.get(("screen", case, f"{family}_{ms}ms")) for ms in intervals
            ]
            if all(values):
                ax.plot(
                    intervals,
                    [r["throughput"] for r in values],
                    marker="o",
                    color=color,
                    label=family.capitalize(),
                )
        for policy, color, style, label in (
            ("fixed", "#222222", "--", "Fixed 200 + 100 ms"),
            ("worker", "#999999", ":", "Worker only"),
        ):
            row = lookup.get(("screen", case, policy))
            if row:
                ax.axhline(row["throughput"], color=color, ls=style, label=label)
        ax.set(title=title, xlabel="Polling interval (ms)", xticks=intervals)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Throughput (requests/s)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("GLM 5.3 TP8 — screening cells, concurrency 32, one output token")
    save(fig, out, "polling_screen")

    policies = [
        "fixed",
        "readiness_10ms",
        "prediction_1ms",
        "prediction_5ms",
        "recent_1ms",
        "hybrid_50ms",
    ]
    labels = [
        "Fixed",
        "Ready 10 ms",
        "Predict 1 ms",
        "Predict 5 ms",
        "Recent reserve",
        "Hybrid 50 ms",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, (case, title) in zip(axes, CASES):
        means, low, high = [], [], []
        for policy in policies:
            values = [
                lookup[key]["throughput"]
                for key in (("screen", case, policy), ("confirmation", case, policy))
                if key in lookup
            ]
            value = float(np.mean(values)) if values else 0
            means.append(value)
            low.append(value - min(values) if values else 0)
            high.append(max(values) - value if values else 0)
        ax.bar(
            np.arange(len(policies)),
            means,
            color=["#777777", "#2266aa", "#cc6633", "#e69a6b", "#339977", "#8866aa"],
            yerr=[low, high],
            capsize=3,
        )
        ax.set(title=title, xticks=np.arange(len(labels)), xticklabels=labels)
        ax.tick_params(axis="x", labelrotation=40)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean throughput (requests/s)")
    fig.suptitle("Matched screening and confirmation seeds; whiskers show range")
    save(fig, out, "strategy_comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    plot(parser.parse_args().root)
