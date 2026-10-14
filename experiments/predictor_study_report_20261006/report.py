# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build reproducible initial-forecast accuracy tables and a standalone figure."""

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PHASES = [
    "submission_progress",
    "transfer_progress",
    "mixed",
    "dense_validation",
    "online_live",
    "generalization",
]
LABELS = {
    "glm53_seal_calibration_20261006/trained_model": "Original predictor",
    "expanded/model": "Expanded forest",
    "refreshed/model": "Refreshed forest",
    "boosted_residual/model": "Boosted residual",
    "boosted_base/model": "Boosted base + residual",
    "dense_augmentation/model": "Dense cached coverage",
    "realized_augmentation/model": "Realized cached coverage forest",
    "boosted/model": "Realized cached coverage residual",
}


def gather(root):
    phases = {}
    for phase in PHASES:
        completed = root / phase / "frozen_evaluation/completed.json"
        if not completed.is_file():
            continue
        score_paths = [
            completed.parent / "common_accuracy.json",
            completed.parent / "refreshed_accuracy.json",
            completed.parent / "refreshed_boosted_accuracy.json",
            root / "boosted_residual" / f"{phase}.json",
            root / "dense_augmentation" / f"{phase}.json",
            root / "boosted_base" / f"{phase}.json",
            root / "realized_augmentation" / f"{phase}.json",
            root / phase / "analysis/frozen_accuracy.json",
            root / phase / "frozen_evaluation/final_models_accuracy.json",
        ]
        models, sources, count, episodes, digest = {}, [], None, None, None
        for path in score_paths:
            if not path.is_file():
                continue
            report = json.loads(path.read_text())
            if count is not None and (
                report["samples"] != count or report["corpus_sha256"] != digest
            ):
                raise ValueError(f"model scores do not share the same frames: {phase}")
            count, episodes = report["samples"], report["episodes"]
            digest = report["corpus_sha256"]
            for name, values in report["models"].items():
                if name in LABELS:
                    models[name] = values
            sources.append(
                dict(
                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
            )
        serving = root / phase / "serving_comparison.json"
        phases[phase] = dict(
            samples=count,
            serving_episodes=episodes,
            corpus_sha256=digest,
            initial_models=models,
            score_sources=sources,
            serving=(
                json.loads(serving.read_text())["grouped"] if serving.is_file() else []
            ),
            issued_accuracy=(
                json.loads((root / phase / "issued_accuracy.json").read_text())
                if (root / phase / "issued_accuracy.json").is_file()
                else None
            ),
        )
    return phases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phases = gather(args.root)
    if not phases:
        raise ValueError("need a complete independent initial-forecast phase")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        model="GLM5.3 NVFP4; TP8; B300; FP8 KV",
        target="All-rank physical end after all-rank GPU start (ms)",
        phases=phases,
        limits=[
            "Only initial forecasts appear in the accuracy table and figure.",
            "GPU profiles and later progress accuracy never enter this comparison.",
            "Cached-prefill training and supported-shape evaluation only.",
            "Original predictor retains worker and preparation optimizations.",
            "Newer phases absent from frozen training; some used for model selection.",
            "A lower MAE does not establish a throughput gain.",
        ],
    )
    decision = args.root / "realized_augmentation/serving_decision.json"
    results = args.root / "realized_augmentation/serving_results.json"
    if decision.is_file():
        summary["fresh_serving_decision"] = json.loads(decision.read_text())
    if results.is_file():
        summary["fresh_serving_results"] = json.loads(results.read_text())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    names = [
        name
        for name in LABELS
        if all(name in p["initial_models"] for p in phases.values())
    ]
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(max(8, 2.4 * len(phases)), 4.6))
    centers = np.arange(len(phases))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        values = [
            p["initial_models"][name]["physical_end"]["mae_ms"] for p in phases.values()
        ]
        bars = ax.bar(
            centers + (i - (len(names) - 1) / 2) * width,
            values,
            width * 0.94,
            label=LABELS[name],
            color=colors[i],
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
    ax.axhline(5, color="#a50f15", linestyle="--", linewidth=1, label="5 ms target")
    ax.set_xticks(
        centers,
        [f"{name.replace('_', ' ')}\n(n={p['samples']})" for name, p in phases.items()],
    )
    ax.set_ylabel("Initial physical-end forecast MAE (ms)")
    ax.set_title("GLM5.3 TP8 — identical causal forecast frames")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, frameon=False)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(
            args.output / f"initial_accuracy.{extension}", dpi=200, bbox_inches="tight"
        )
    plt.close(fig)
    table = []
    for phase, values in phases.items():
        for name, scores in values["initial_models"].items():
            metrics = scores["physical_end"]
            table.append(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(value))}</td>"
                    for value in [
                        phase,
                        LABELS[name],
                        values["samples"],
                        f"{metrics['mae_ms']:.3f}",
                        f"{metrics['bias_ms']:.3f}",
                        f"{metrics['p95_absolute_ms']:.3f}",
                    ]
                )
                + "</tr>"
            )
    issued_table = []
    for phase, values in phases.items():
        issued = values["issued_accuracy"]
        if not issued:
            continue
        for group in issued["groups"]:
            if group["workload"] != "ALL":
                continue
            issued_table.append(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(v))}</td>"
                    for v in [
                        phase,
                        group["policy"],
                        group["samples"],
                        f"{group['issued']['mae_ms']:.3f}",
                        f"{group['frozen_same_frames']['mae_ms']:.3f}",
                        group["nonzero_corrections"],
                        group["max_history"],
                    ]
                )
                + "</tr>"
            )
    issued_html = ""
    if issued_table:
        issued_html = (
            "<h2>Actual issued predictions</h2>"
            "<p>Online history is transient; static checkpoint scores exclude it. "
            "Each row compares issuance with the frozen model on those same frames.</p>"
            "<table><tr><th>Phase</th><th>Policy</th><th>Frames</th>"
            "<th>Issued MAE</th><th>Frozen MAE</th><th>Corrections</th>"
            "<th>Max history</th></tr>" + "".join(issued_table) + "</table>"
        )
    body = (
        "<!doctype html><meta charset='utf-8'><title>GLM5.3 predictor study</title>"
        "<style>body{font:16px sans-serif;max-width:1100px;margin:40px auto}"
        "td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}"
        "img{max-width:100%}</style><h1>GLM5.3 TP8 predictor study</h1>"
        "<p>Frozen cached-workload models, identical causal initial forecasts.</p>"
        "<img src='initial_accuracy.png' alt='Initial forecast mean absolute error'>"
        "<table><thead><tr><th>Phase</th><th>Predictor</th><th>Frames</th>"
        "<th>MAE (ms)</th><th>Bias (ms)</th><th>p95 absolute (ms)</th>"
        "</tr></thead><tbody>"
        + "".join(table)
        + "</tbody></table>"
        + issued_html
        + "<h2>Scope</h2><ul>"
        + "".join(f"<li>{html.escape(limit)}</li>" for limit in summary["limits"])
        + "</ul><p>See summary.json for full scores and source hashes.</p>"
    )
    (args.output / "index.html").write_text(body + "\n")
    print(json.dumps({name: p["samples"] for name, p in phases.items()}))


if __name__ == "__main__":
    main()
