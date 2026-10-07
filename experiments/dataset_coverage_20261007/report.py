# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize independent dataset additions, uncertainty, and live results."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/mnt/localnvme/glm53_dataset_coverage_20261007")


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    batches = read(ROOT / "data/batches.jsonl")
    forecasts = read(ROOT / "data/forecasts.jsonl")
    final = json.loads((ROOT / "study/final.json").read_text())
    selection = final["selection"]
    allocation = json.loads((ROOT / "allocation.json").read_text())
    inventory = {
        "batches": dict(Counter(f"{r['split']}:{r['block']}" for r in batches)),
        "forecasts": dict(Counter(f"{r['split']}:{r['block']}" for r in forecasts)),
        "context_range": [
            min(c for r in batches for s, c in r["request_shapes"]),
            max(c for r in batches for s, c in r["request_shapes"]),
        ],
        "suffix_range": [
            min(s for r in batches for s, c in r["request_shapes"]),
            max(s for r in batches for s, c in r["request_shapes"]),
        ],
        "allocation_seconds": allocation["total_seconds"],
    }
    lines = [
        "# GLM 5.3 TP8 dataset coverage evaluation",
        "",
        f"Selected on development data: **{selection['winner']}**, seed 73.",
        "",
        (
            "The tree architecture, features, and hyperparameters are unchanged. "
            "Original-recipe variants retain the existing weighting; mass_matched "
            "controls equalize each addition's total training weight. "
            "GPU compute geometry uses actual scheduled prompt tokens and context; "
            "CPU cache-hit percentage only generates request mixtures. "
            "All new GPU cells use the same private CPU-cache layout."
        ),
        "",
        "## Fresh final accuracy",
        "",
        "| Dataset addition | MAE (ms, seed 73) | MAPE | p95 error (ms) |",
        "|---|---:|---:|---:|",
    ]
    names, values = [], []
    for name, result in final["results"].items():
        metric = result.get("serving_seed73", result)
        lines.append(
            f"| {name} | {metric['mae_ms']:.2f} | {metric['mape_percent']:.2f}% "
            f"| {metric['p95_absolute_ms']:.2f} |"
        )
        names.append(name)
        values.append(metric["mae_ms"])
    lines += [
        "",
        (
            "The live checkpoint uses seed 73. Final JSON reports all three seeds "
            "and paired episode-bootstrap intervals for their mean error."
        ),
        "",
        "## Paired live confirmation",
        "",
    ]
    cells = []
    for path in (ROOT / "live").rglob("result.json"):
        r = json.loads(path.read_text())
        if r["accepted"]:
            cells.append(r)
    pairs = defaultdict(dict)
    for r in cells:
        pairs[(r["workload"], r["repeat"])][r["label"]] = r
    complete = [p for p in pairs.values() if set(p) == {"historical", "selected"}]
    live = {}
    lines += [
        (
            "| Workload | Pairs | Historical req/s | Selected req/s "
            "| Throughput gain | Mean TTFT gain | p95 TTFT gain |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for workload in sorted({r["workload"] for r in cells}):
        ps = [p for p in complete if p["historical"]["workload"] == workload]
        if not ps:
            continue
        aggregates = {}
        for label in ["historical", "selected"]:
            rs = [p[label] for p in ps]
            ttft = [s["ttft_ms"] for r in rs for s in r["responses"]]
            aggregates[label] = dict(
                rps=sum(r["requests"] for r in rs) / sum(r["seconds"] for r in rs),
                mean_ttft=float(np.mean(ttft)),
                p95_ttft=float(np.quantile(ttft, 0.95)),
                compute_rounds=sum(r["compute_rounds"] for r in rs),
            )
        a, b = aggregates["historical"], aggregates["selected"]
        gains = dict(
            throughput=100 * (b["rps"] / a["rps"] - 1),
            mean_ttft=100 * (1 - b["mean_ttft"] / a["mean_ttft"]),
            p95_ttft=100 * (1 - b["p95_ttft"] / a["p95_ttft"]),
        )
        live[workload] = dict(pairs=len(ps), aggregates=aggregates, gains_percent=gains)
        lines.append(
            f"| {workload} | {len(ps)} | {a['rps']:.2f} | {b['rps']:.2f} "
            f"| {gains['throughput']:+.1f}% | {gains['mean_ttft']:+.1f}% "
            f"| {gains['p95_ttft']:+.1f}% |"
        )
    lines += [
        "",
        "## Coverage and limits",
        "",
        (
            f"Normalized batches: {len(batches)}; causal forecasts: {len(forecasts)}. "
            f"Total TP8 allocation: {allocation['total_seconds'] / 3600:.2f} hours. "
            f"Live pairs completed: {len(complete)}/8."
        ),
        "",
        (
            "First training occurrences and the first development seed are warmup. "
            "Cells with reported inference JIT or unconfirmed own-KV readiness "
            "are excluded. Transfer overlap uses submission/completion intervals; "
            "it does not measure SM contention. Original-recipe and mass-matched "
            "weighting comparisons are reported separately; the dataset audit "
            "records weight masses. The generalization scope is "
            "GLM 5.3 TP8 within the tested 64K range."
        ),
        "",
    ]
    summary = dict(
        inventory=inventory,
        selected=selection["winner"],
        live=live,
        final_samples=final["samples"],
        complete_live_pairs=len(complete),
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for mass_matched, ax in zip([False, True], axes):
        indices = [
            i
            for i, n in enumerate(names)
            if n.endswith("_mass_matched") == mass_matched
        ]
        labels = [names[i].removesuffix("_mass_matched") for i in indices]
        ax.bar(
            np.arange(len(indices)),
            [values[i] for i in indices],
            color=[
                "#f58518" if names[i] == selection["winner"] else "#4c78a8"
                for i in indices
            ],
        )
        ax.set_xticks(np.arange(len(indices)), labels, rotation=35, ha="right")
        ax.set_ylabel("Final MAE (ms)")
        ax.set_title("Equal source weight" if mass_matched else "Original weights")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "accuracy.png", dpi=180)
    fig.savefig(ROOT / "accuracy.pdf")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
