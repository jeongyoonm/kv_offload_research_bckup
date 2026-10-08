# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit transfer-only measurements and select a startup configuration."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    root = args.directory
    plan = json.loads((root / "plan.json").read_text())
    assert json.loads((root / "summary.json").read_text())["complete"]
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    assert not any(e["event"] == "error" for e in events)
    ready = [e for e in events if e["event"] == "ready"]
    assert {e["rank"] for e in ready} == set(range(plan["ranks"]))
    assert all(e["binding"]["thp_disable"] == 1 for e in ready)
    assert {e["rank"] for e in events if e["event"] == "done"} == set(
        range(plan["ranks"])
    )
    ranks = defaultdict(list)
    for event in events:
        if event["event"] == "cell":
            ranks[event["cell"]].append(event)
    assert len(ranks) == len(plan["cells"])
    cells = []
    for cell in plan["cells"]:
        group = ranks[cell["cell"]]
        assert {r["rank"] for r in group} == set(range(plan["ranks"]))
        assert len(group) == plan["ranks"]
        for row in group:
            assert all(row[k] == cell[k] for k in cell)
            assert (
                row["bytes_per_job"]
                == plan["row_bytes"] * plan["profiles"][cell["profile"]]
            )
            assert (
                row["descriptors_per_job"]
                == len(plan["page_sizes"]) * plan["profiles"][cell["profile"]]
            )
            assert sum(row["chunk_bytes"]) == row["bytes_per_job"]
            assert row["validated_bytes"] == 99 * 3 * 64
            assert (
                len(row["completion_ns"])
                == len(row["cuda_duration_ms"])
                == plan["jobs"]
            )
        assert len({tuple(r["chunk_bytes"]) for r in group}) == 1
        start = min(r["begin_ns"] for r in group)
        finish = max(r["completion_ns"][-1] for r in group)
        ms = (finish - start) / 1e6
        total_bytes = sum(r["bytes_per_job"] for r in group) * plan["jobs"]
        cells.append(
            dict(
                **cell,
                all_rank_ms=ms,
                first_job_all_rank_ms=(
                    max(r["completion_ns"][0] for r in group) - start
                )
                / 1e6,
                start_spread_ms=(max(r["begin_ns"] for r in group) - start) / 1e6,
                aggregate_gbps=total_bytes / (ms * 1e6),
                effective_per_gpu_gbps=total_bytes / (ms * 1e6) / plan["ranks"],
                rank_mean_submit_ms=statistics.mean(
                    r["submit_ns"] / 1e6 for r in group
                ),
                rank_mean_cpu_ms=statistics.mean(r["cpu_ns"] / 1e6 for r in group),
                chunks_per_job=group[0]["chunks_per_job"],
                max_actual_chunk_mib=max(group[0]["chunk_bytes"]) / 1024**2,
                mean_actual_chunk_mib=statistics.mean(group[0]["chunk_bytes"])
                / 1024**2,
                transfer_bytes=total_bytes,
            )
        )
    groups = defaultdict(list)
    for cell in cells:
        groups[cell["chunk_mib"], cell["depth"], cell["profile"]].append(cell)
    aggregates = []
    for (chunk, depth, profile), group in sorted(groups.items()):
        aggregates.append(
            dict(
                chunk_mib=chunk,
                depth=depth,
                profile=profile,
                trials=len(group),
                mean_all_rank_ms=statistics.mean(c["all_rank_ms"] for c in group),
                median_all_rank_ms=statistics.median(c["all_rank_ms"] for c in group),
                mean_first_job_all_rank_ms=statistics.mean(
                    c["first_job_all_rank_ms"] for c in group
                ),
                effective_per_gpu_gbps=sum(c["transfer_bytes"] for c in group)
                / sum(c["all_rank_ms"] for c in group)
                / 1e6
                / plan["ranks"],
                rank_mean_submit_ms=statistics.mean(
                    c["rank_mean_submit_ms"] for c in group
                ),
                rank_mean_cpu_ms=statistics.mean(c["rank_mean_cpu_ms"] for c in group),
                max_start_spread_ms=max(c["start_spread_ms"] for c in group),
                chunks_per_job=group[0]["chunks_per_job"],
                max_actual_chunk_mib=group[0]["max_actual_chunk_mib"],
                mean_actual_chunk_mib=group[0]["mean_actual_chunk_mib"],
            )
        )
    by_config = defaultdict(dict)
    for aggregate in aggregates:
        by_config[aggregate["chunk_mib"], aggregate["depth"]][aggregate["profile"]] = (
            aggregate
        )
    baseline = by_config[128, 2]
    scores = []
    for (chunk, depth), values in by_config.items():
        ratios = {
            name: values[name]["effective_per_gpu_gbps"]
            / baseline[name]["effective_per_gpu_gbps"]
            for name in plan["profiles"]
        }
        scores.append(
            dict(
                chunk_mib=chunk,
                depth=depth,
                ratios_to_baseline=ratios,
                geometric_speedup=math.exp(
                    statistics.mean(math.log(r) for r in ratios.values())
                ),
                minimum_speedup=min(ratios.values()),
            )
        )
    best = max(scores, key=lambda row: row["geometric_speedup"])
    eligible = [
        r
        for r in scores
        if r["geometric_speedup"] >= 0.95 * best["geometric_speedup"]
        and r["minimum_speedup"] >= 0.97
    ]
    recommended = (
        min(eligible, key=lambda row: (row["depth"], row["chunk_mib"]))
        if eligible
        else best
    )
    from scipy.stats import t

    contrasts = []
    for (chunk, depth, profile), group in groups.items():
        if (chunk, depth) == (128, 2):
            continue
        reference = {c["repeat"]: c for c in groups[128, 2, profile]}
        assert {c["repeat"] for c in group} == set(reference)
        ratios = [
            math.log(reference[c["repeat"]]["all_rank_ms"] / c["all_rank_ms"])
            for c in group
        ]
        mean = statistics.mean(ratios)
        margin = (
            (
                t.ppf(0.975, len(ratios) - 1)
                * statistics.stdev(ratios)
                / math.sqrt(len(ratios))
            )
            if len(ratios) > 1
            else 0
        )
        contrasts.append(
            dict(
                chunk_mib=chunk,
                depth=depth,
                profile=profile,
                paired_gain_percent=100 * math.expm1(mean),
                paired_95pct_bounds=[
                    100 * math.expm1(mean - margin),
                    100 * math.expm1(mean + margin),
                ],
                repeats=len(ratios),
            )
        )
    report = dict(
        complete=True,
        scope=plan["scope"],
        measured_all_rank_cells=len(cells),
        measured_rank_cells=len(cells) * plan["ranks"],
        measured_rank_jobs=len(cells) * plan["ranks"] * plan["jobs"],
        aggregate_transferred_bytes=sum(c["transfer_bytes"] for c in cells),
        sampled_validation_bytes=sum(
            r["validated_bytes"] for group in ranks.values() for r in group
        ),
        selection_rule=(
            "Equal-profile geometric bandwidth; shallowest then smallest within "
            "95% of best score, with no profile below 97% of baseline"
        ),
        baseline=dict(chunk_mib=128, depth=2),
        best=best,
        recommended=recommended,
        scores=sorted(scores, key=lambda r: r["geometric_speedup"], reverse=True),
        aggregate=aggregates,
        cells=cells,
        paired_contrasts=contrasts,
    )
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Transfer-only chunk and depth calibration",
        "",
        "<!-- markdownlint-disable MD013 -->",
        "",
        "| Chunk MiB | Depth | 4K GB/s/GPU | 16K GB/s/GPU | 60K GB/s/GPU | "
        "Geometric speedup vs 128/2 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["scores"]:
        values = by_config[row["chunk_mib"], row["depth"]]
        speeds = [values[p]["effective_per_gpu_gbps"] for p in plan["profiles"]]
        lines.append(
            f"| {row['chunk_mib']} | {row['depth']} | {speeds[0]:.2f} | "
            f"{speeds[1]:.2f} | {speeds[2]:.2f} | {row['geometric_speedup']:.4f} |"
        )
    lines += [
        "",
        "<!-- markdownlint-enable MD013 -->",
        "",
        "GB/s is decimal effective bandwidth per GPU using completion across all eight",
        "ranks, including native submission and copy gaps. Each trial queues "
        f"{plan['jobs']}",
        "identical FIFO loads per rank. Descriptor construction, allocation,",
        "population, registration, warmups, and sampled byte checks are excluded.",
        "No model compute",
        "is active. These are reconstructed GLM page geometry and randomized source",
        "and destination block mappings, not recorded serving transfer descriptors.",
        "",
        f"Best score: {best['chunk_mib']} MiB / depth {best['depth']}.",
        f"Plateau selection: {recommended['chunk_mib']} MiB / "
        f"depth {recommended['depth']}.",
        "",
        "The 4,096-descriptor cap is unchanged. Nominal chunk sizes can therefore",
        "exceed the actual bytes per submission. Per-job chunk sizes and submission",
        "counts are recorded in the raw measurements.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["svg.hashsalt"] = "h2dcal20261009"
        import matplotlib.pyplot as plt
        import numpy as np

        chunks = sorted({key[0] for key in by_config})
        depths = sorted({key[1] for key in by_config})
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
        for axis, profile in zip(axes, plan["profiles"]):
            data = np.array(
                [
                    [
                        by_config.get((c, d), {})
                        .get(profile, {})
                        .get("effective_per_gpu_gbps", float("nan"))
                        for c in chunks
                    ]
                    for d in depths
                ]
            )
            plot = axis.imshow(data, aspect="auto", cmap="viridis")
            axis.set_xticks(range(len(chunks)), chunks)
            axis.set_yticks(range(len(depths)), depths)
            axis.set_xlabel("Nominal chunk size (MiB)")
            axis.set_ylabel("Pipeline depth")
            axis.set_title(profile + " prefix restoration")
            for i in range(len(depths)):
                for j in range(len(chunks)):
                    if not math.isnan(data[i, j]):
                        axis.text(
                            j,
                            i,
                            f"{data[i, j]:.1f}",
                            ha="center",
                            va="center",
                            color="black" if plot.norm(data[i, j]) > 0.6 else "white",
                            fontsize=8,
                        )
            fig.colorbar(plot, ax=axis, label="Effective GB/s per GPU")
        fig.suptitle("Eight-rank H2D calibration, no model compute")
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(root / f"calibration.{extension}", dpi=180)
        plt.close(fig)
    print(
        json.dumps(dict(best=best, recommended=recommended, cells=len(cells)), indent=2)
    )


if __name__ == "__main__":
    main()
