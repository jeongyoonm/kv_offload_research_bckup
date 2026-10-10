# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize completed setup-deadline serving allocations and their caveats."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

LABELS = dict(
    short16k="16K",
    mid32k="32K",
    unseen48k="49.5K",
    familiar60k="60K",
    mixed_supported="Mixed",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    allocations, total = [], 0
    text = [
        "Learned B setup-time sealing allowance: GLM5.3 NVFP4 TP8",
        "Updated " + datetime.now(timezone.utc).isoformat(),
        "",
        "A compute predictor weights, worker configuration and numerical "
        "execution remain frozen. Deadline = predicted A finish - B setup "
        "allowance. Unsupported inputs retain fixed50. Safety and readiness "
        "early exits are preserved.",
        "Worker: H2D8MiB/depth2, native D2H1GiB/depth2; private CPU cache "
        "2,130,300,354,560 server-wide bytes. Concurrency32; output1 token. "
        "CPU-restored cached prefixes, zero GPU-prefix hits and no preemption "
        "are acceptance guards.",
        "Two repeats per cell; reverse arm order on repeat1. Levels are "
        "arithmetic means; gains are paired geometric means. p95 TTFT is "
        "the mean of episode p95 values, not a pooled p95. These are "
        "measured cells, not confidence intervals.",
        "",
    ]
    for root in args.root:
        summary = json.loads((root / "summary.json").read_text())
        assert summary["complete"] and summary["stopped"]
        assert all(r["accepted"] and r["frozen_unchanged"] for r in summary["results"])
        audit = json.loads((root / "setup_audit.json").read_text())
        plan = json.loads((root / "plan.json").read_text())
        count = len(summary["results"])
        total += count * plan["requests"]
        allocations.append(
            dict(
                root=str(root),
                plan=plan,
                audit=audit,
                replicates=[
                    {
                        k: r[k]
                        for k in (
                            "workload",
                            "label",
                            "repeat",
                            "seed",
                            "throughput_rps",
                            "mean_ttft_ms",
                            "p95_ttft_ms",
                        )
                    }
                    for r in summary["results"]
                ],
            )
        )
        text += [
            root.name + f": {count} accepted {plan['requests']}-request cells",
            "Seeds: " + str(plan["seeds"]),
            "Workload  Arm             req/s   gain%  meanTTFT  p95TTFT  "
            "rounds  recovery%  gaps_ms",
        ]
        changes = {(r["workload"], r["arm"]): r for r in audit["comparisons"]}
        for r in audit["levels"]:
            change = changes[r["workload"], r["arm"]]
            text.append(
                f"{LABELS[r['workload']]:<9} {r['arm']:<16} "
                f"{r['throughput_rps']:6.3f} "
                f"{change['throughput_rps_change_percent']:+6.2f} "
                f"{r['mean_ttft_ms']:8.1f} {r['p95_ttft_ms']:8.1f} "
                f"{r['compute_rounds']:6.1f} {r['recovered_percent']:9.2f} "
                f"{r['inter_compute_gap_ms']:8.1f}"
            )
        text += [
            "",
            "Governing setup allowance errors; FIFO removed from labels:",
            "Workload  Arm             MAE_ms   bias_ms  coverage%  "
            "actual_setup_ms  allowance_ms  live_cost_median_us",
        ]
        for r in audit["levels"]:
            s = r["setup_accuracy"]
            cost = r["live_forecasts"]["cost_us"].get("median")
            text.append(
                f"{LABELS[r['workload']]:<9} {r['arm']:<16} "
                f"{s['mae_ms']:7.2f} {s['bias_ms']:+8.2f} "
                f"{s['covered_percent']:9.2f} "
                f"{r['setup']['target_ready_ms']['mean']:15.2f} "
                f"{r['setup']['governing_allowance_ms']['mean']:13.2f} "
                + (f"{cost:19.2f}" if cost is not None else "                  -")
            )
        text += ["Replay audit: " + json.dumps(audit["audit"]), ""]
    text += [
        f"Total accepted measured requests: {total}",
        "",
        "Interpretation limits:",
        "- GPU timings are conservative execution-event envelopes and "
        "readiness proxies, not isolated first-GEMM or pure kernel time.",
        "- Setup begins per rank at max(seal, next-RPC-ready); FIFO wait "
        "is separate. A prediction error still affects the deadline.",
        "- More accurate setup does not guarantee throughput improvement: "
        "batch composition, arrival/load timing and load-only pipeline "
        "regimes also matter. Inspect per-repeat gaps and initial load "
        "coverage; no causal throughput claim follows from MAE alone.",
        "- A conditional q90 learned with a different model is not "
        "pointwise above the ExtraTrees mean, nor guaranteed90% coverage.",
        "- Historical unseen48k means49.5K total prompt/48K cached prefix. "
        "It is present in setup training. Mixed includes two new C/Q "
        "combinations within the existing per-request maxima.",
        "- Scope is the current model/TP8/hardware/worker configuration, "
        "concurrency32 and slots16; arbitrary parallelism, decoding and "
        "hardware generalization are not established.",
        "",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "serving_report.txt").write_text("\n".join(text))
    (args.output / "serving_report.json").write_text(
        json.dumps(dict(allocations=allocations, measured_requests=total), indent=2)
        + "\n"
    )
    if args.plot:
        plot(allocations, args.output)


def plot(allocations, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(
        len(allocations),
        2,
        figsize=(12, 3.3 * len(allocations)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = dict(
        fixed50="#666666",
        fixed5="#D89028",
        learned_mean="#2E779D",
        learned_q90="#8C63AB",
        shadow_mean50="#B77540",
    )
    for row, allocation in enumerate(allocations):
        plan, audit = allocation["plan"], allocation["audit"]
        levels = {(r["workload"], r["arm"]): r for r in audit["levels"]}
        x = np.arange(len(plan["workloads"]))
        width = 0.24
        for col, (metric, label) in enumerate(
            [
                ("throughput_rps", "Throughput (requests/s)"),
                ("p95_ttft_ms", "p95 TTFT (ms)"),
            ]
        ):
            ax = axes[row, col]
            for i, arm in enumerate(plan["arms"]):
                ax.bar(
                    x + (i - 1) * width,
                    [levels[w, arm][metric] for w in plan["workloads"]],
                    width,
                    label=arm,
                    color=colors[arm],
                    alpha=0.85,
                )
                for j, workload in enumerate(plan["workloads"]):
                    values = [
                        r[metric]
                        for r in allocation["replicates"]
                        if r["workload"] == workload and r["label"] == arm
                    ]
                    ax.scatter(
                        j + (i - 1) * width + np.linspace(-0.035, 0.035, len(values)),
                        values,
                        color="black",
                        s=14,
                        zorder=3,
                    )
            ax.set_xticks(x, [LABELS[w] for w in plan["workloads"]])
            ax.set_ylabel(label)
            ax.set_title(Path(allocation["root"]).name)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8)
    fig.suptitle("Frozen A predictor and workers; dots show two300-request repeats")
    fig.savefig(output / "serving_comparison.png", dpi=180)
    fig.savefig(output / "serving_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
