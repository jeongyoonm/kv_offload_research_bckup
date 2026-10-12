# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched finite-episode results; separate offered-load and saturation evidence."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CAMPAIGN = Path("/mnt/localnvme/jeongyoon/real_traces_20261012")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    out = CAMPAIGN / "report"
    out.mkdir(exist_ok=True)
    rows = []
    pairs = {}
    rejected = []
    for root in args.roots:
        for path in (root / "cells").glob("*/result.json"):
            result = json.loads(path.read_text())
            if not result["accepted"]:
                rejected.append(str(path))
                continue
            row = dict(
                root=str(root),
                path=str(path),
                id=result["spec"]["id"],
                dataset=result["spec"]["dataset"],
                arm=result["arm"],
                requests=result["requests"],
                seconds=result["seconds"],
                throughput_rps=result["throughput_rps"],
                output_tps=result["output_tokens_per_second"],
                mean_ttft_ms=result["ttft_ms"]["mean"],
                p50_ttft_ms=result["ttft_ms"]["p50"],
                p95_ttft_ms=result["ttft_ms"]["p95"],
                p99_ttft_ms=result["ttft_ms"]["p99"],
                scheduled_mean_ttft_ms=result["scheduled_ttft_ms"]["mean"],
                scheduled_p95_ttft_ms=result["scheduled_ttft_ms"]["p95"],
                driver_mean_queue_ms=result["driver_queue_ms"]["mean"],
                cached_input_fraction=result["cached_input_fraction"],
                cpu_hit_requests=result["realized_outcomes"].get("cpu_hit", 0),
                gpu_hit_requests=result["realized_outcomes"].get("gpu_hit", 0),
                miss_requests=result["realized_outcomes"].get("miss", 0),
                h2d_gib_all_ranks=result["h2d_bytes_all_ranks"] / 1024**3,
                d2h_gib_all_ranks=result["d2h_bytes_all_ranks"] / 1024**3,
                late_admissions=result["protocol_counts"].get(
                    "membership_sealed_activated", 0
                ),
                preemptions=result["metric_deltas"].get(
                    "vllm:num_preemptions_total", 0
                ),
                offered_rps=result.get("offered_rps"),
                drain_seconds=result.get("drain_after_last_arrival_seconds"),
            )
            rows.append(row)
            pairs.setdefault(row["id"], {})[row["arm"]] = (row, result)
    with (out / "absolute_results.csv").open("w") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    comparisons = []
    for name, arms in pairs.items():
        if set(arms) != {"baseline", "full"}:
            continue
        b, br = arms["baseline"]
        p, pr = arms["full"]
        assert br["spec"] == pr["spec"], name
        assert br["dataset_sha256"] == pr["dataset_sha256"], name
        assert (
            br["input_tokens"] == pr["input_tokens"]
            and br["output_tokens"] == pr["output_tokens"]
        ), name
        row = dict(
            id=name,
            dataset=b["dataset"],
            requests=b["requests"],
            baseline_root=b["root"],
            full_root=p["root"],
            baseline_rps=b["throughput_rps"],
            full_rps=p["throughput_rps"],
            throughput_gain_percent=(p["throughput_rps"] / b["throughput_rps"] - 1)
            * 100,
            baseline_mean_ttft_ms=b["mean_ttft_ms"],
            full_mean_ttft_ms=p["mean_ttft_ms"],
            mean_ttft_reduction_percent=(1 - p["mean_ttft_ms"] / b["mean_ttft_ms"])
            * 100,
            baseline_p95_ttft_ms=b["p95_ttft_ms"],
            full_p95_ttft_ms=p["p95_ttft_ms"],
            p95_ttft_reduction_percent=(1 - p["p95_ttft_ms"] / b["p95_ttft_ms"]) * 100,
            baseline_scheduled_ttft_ms=b["scheduled_mean_ttft_ms"],
            full_scheduled_ttft_ms=p["scheduled_mean_ttft_ms"],
            baseline_cached_fraction=b["cached_input_fraction"],
            full_cached_fraction=p["cached_input_fraction"],
            baseline_cpu_requests=b["cpu_hit_requests"],
            full_cpu_requests=p["cpu_hit_requests"],
            full_late_admissions=p["late_admissions"],
            baseline_h2d_gib=b["h2d_gib_all_ranks"],
            full_h2d_gib=p["h2d_gib_all_ranks"],
            offered_rps=b["offered_rps"],
            baseline_drain_seconds=b["drain_seconds"],
            full_drain_seconds=p["drain_seconds"],
        )
        comparisons.append(row)
    (out / "comparisons.json").write_text(
        json.dumps(
            dict(
                comparisons=comparisons,
                accepted_cells=len(rows),
                rejected=rejected,
                unmatched=[
                    name
                    for name, arms in pairs.items()
                    if set(arms) != {"baseline", "full"}
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    with (out / "comparisons.csv").open("w") as f:
        if comparisons:
            writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    if comparisons:
        main = [c for c in comparisons if not c["id"].startswith("cal_")]
        if main:
            fig, axes = plt.subplots(
                1, 3, figsize=(max(12, len(main) * 2.5), 4), constrained_layout=True
            )
            x = np.arange(len(main))
            width = 0.36
            for ax, field, label in zip(
                axes,
                ["rps", "mean_ttft_ms", "p95_ttft_ms"],
                ["Completed requests/s", "Mean TTFT (s)", "p95 TTFT (s)"],
            ):
                scale = 1 if field == "rps" else 1000
                for arm, offset, color in [
                    ("baseline", -width / 2, "#526677"),
                    ("full", width / 2, "#ce702d"),
                ]:
                    bars = ax.bar(
                        x + offset,
                        [c[arm + "_" + field] / scale for c in main],
                        width,
                        label="Baseline" if arm == "baseline" else "Full prototype",
                        color=color,
                    )
                    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
                ax.set_xticks(
                    x,
                    [
                        c["id"]
                        .replace("moon_", "")
                        .replace("_r0", "")
                        .replace("_", "\n")
                        for c in main
                    ],
                    fontsize=8,
                )
                ax.set_ylabel(label)
                ax.grid(axis="y", alpha=0.2)
                ax.set_axisbelow(True)
            axes[0].legend(loc="upper left")
            fig.suptitle("GLM 5.3 NVFP4 · TP8 · Natural evolving-cache trace replay")
            fig.savefig(out / "comparison.png", dpi=180)
            fig.savefig(out / "comparison.pdf")
            plt.close(fig)
    lines = [
        "GLM 5.3 NVFP4 TP8 — Inferact and Mooncake baseline/full comparison",
        "",
        f"Accepted cells: {len(rows)}; matched pairs: {len(comparisons)}.",
        "",
        (
            "Episode throughput runs from first scheduled arrival to last completion. "
            "Fixed offered-load cells can have similar throughput even when capacity "
            "differs. Read TTFT, scheduled TTFT, driver queueing and drain with it."
        ),
        (
            "Inferact uses completion-paced sessions, zero extra tool wait, recorded "
            "assistant history and GLM-tokenized response output targets. This is a "
            "performance replay, not original GPT cache behavior or quality evaluation."
        ),
        (
            "Mooncake synthetic token IDs preserve cumulative 512-token prefix hashes, "
            "exact ISL/OSL, and scaled irregular arrivals. Source text semantics and "
            "original generated-output prefix sharing cannot be reconstructed."
        ),
        (
            "Requests above 81920 input+output are excluded; Inferact trajectories are "
            "censored at first violation. Original selector/numerics retained. Shared "
            "private CPU initialization; 2.13TB CPU/96GiB per-rank GPU KV budgets."
        ),
        "",
        (
            "ID | Base rps | Full rps | Gain% | Mean TTFT base/full ms | "
            "p95 base/full ms | CPU-hit requests base/full"
        ),
    ]
    for c in comparisons:
        lines.append(
            f"{c['id']} | {c['baseline_rps']:.4f} | {c['full_rps']:.4f} | "
            f"{c['throughput_gain_percent']:+.2f} | "
            f"{c['baseline_mean_ttft_ms']:.1f}/{c['full_mean_ttft_ms']:.1f} | "
            f"{c['baseline_p95_ttft_ms']:.1f}/{c['full_p95_ttft_ms']:.1f} | "
            f"{c['baseline_cpu_requests']}/{c['full_cpu_requests']}"
        )
    (out / "REPORT.txt").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                accepted=len(rows),
                matched=len(comparisons),
                report=str(out / "REPORT.txt"),
            )
        )
    )


if __name__ == "__main__":
    main()
