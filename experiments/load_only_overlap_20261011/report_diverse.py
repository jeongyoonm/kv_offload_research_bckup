# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize matched serving arms with cache accounting and scientific plots."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.load_only_overlap_20261011.analyze_serving import uncovered
from experiments.load_only_overlap_20261011.opportunities import (
    execution_bounds,
)
from experiments.load_only_overlap_20261011.opportunities import (
    summarize as readiness_opportunities,
)

ARMS = ("baseline", "worker", "full")
COLORS = dict(baseline="#444444", worker="#3976b8", full="#d87d25")
METRICS = ("throughput_rps", "mean_ttft_ms", "p95_ttft_ms", "p99_ttft_ms")
ROOTS = (
    "dc11bh",
    "dc11oi",
    "dc11bx",
    "dc11ox",
    "dc11be",
    "dc11oe",
    "dc11bc",
    "dc11oc",
    "dc11bs",
    "dc11os",
)


@lru_cache(maxsize=16)
def worker_entries(root):
    entries = defaultdict(list)
    for path in (root / "server/timing/late_admission").glob("*-worker-*.json"):
        exported = json.loads(path.read_text())
        assert not exported.get("overflow"), path
        for entry in exported["records"]:
            if entry["event"] == "worker_entry":
                entries[exported["rank"], entry["step"]].append(entry)
    return entries


def timing_details(data, root, complete):
    directory = Path(data["source_result"]).parent
    engine = [
        json.loads(line)
        for line in (directory / "engine.jsonl").read_text().splitlines()
    ]
    records = json.loads((directory / "protocol.json").read_text())
    computed = {
        e["step"]: e
        for e in engine
        if e["event"] == "step_completion" and e.get("selected") and e.get("valid")
    }
    intervals = sorted(
        (
            max(r[3] for r in e["execution_ranks"]),
            max(r[3] + r[6] for r in e["execution_ranks"]),
        )
        for e in computed.values()
    )
    holds, errors, seal_slacks, seal_to_gpu = [], [], [], []
    for event in records:
        if event["event"] != "seal_eligibility":
            continue
        transport = event["transport"]
        if transport.get("base_ready_ns") and transport.get("seal_ns"):
            holds.append(transport)
        prediction = transport.get("prediction") or {}
        predecessor = computed.get(prediction.get("predecessor_step"))
        upcoming = computed.get(event.get("step"))
        if predecessor and prediction.get("predicted_duration_ns"):
            end = max(r[3] + r[6] for r in predecessor["execution_ranks"])
            estimated = (
                prediction["gpu_start_upper_ns"] + prediction["predicted_duration_ns"]
            )
            errors.append((estimated - end) / 1e6)
            seal_slacks.append((end - transport["seal_ns"]) / 1e6)
        if upcoming and transport.get("seal_ns"):
            start = max(r[3] for r in upcoming["execution_ranks"])
            seal_to_gpu.append((start - transport["seal_ns"]) / 1e6)
    load_only = [t for t in holds if not t.get("has_compute", False)]
    leading = [
        (min(x[0] for x in measured) - start) / 1e6
        for start, end in data["windows"]
        for measured in [[x for x in intervals if start <= x[0] <= end]]
        if measured
    ]
    coverage = None
    if complete and data["arm"] != "baseline":
        indexed = worker_entries(root)
        coverage = {}
        for rank in range(8):
            entries = [
                entry
                for step in computed
                for entry in indexed.get((rank, step), [])
                if any(a <= entry["time_ns"] <= b for a, b in data["windows"])
            ]
            assert len(entries) <= len(computed), (directory, rank, len(entries))
            observed_steps = {e["step"] for e in entries}
            assert len(observed_steps) == len(entries), (directory, rank)
            coverage[rank] = dict(
                total_compute_steps=len(computed),
                compute_entries=len(entries),
                unobserved_steps=sorted(set(computed) - observed_steps),
                compact_entries=sum(bool(e.get("compact_tokens")) for e in entries),
                pinned_entries=sum(bool(e.get("pinned_tokens")) for e in entries),
                prepared_entries=sum(bool(e.get("prepared")) for e in entries),
            )
    return dict(
        execution_uncertainty=execution_bounds(
            list(computed.values()), data["windows"]
        ),
        readiness_opportunities=readiness_opportunities(
            [
                json.loads(line)
                for line in (directory / "load_timeline.jsonl").read_text().splitlines()
            ],
            records,
            list(computed.values()),
            data["windows"],
        ),
        worker_preparation_coverage=coverage,
        prediction_count=len(errors),
        prediction_end_mae_ms=statistics.mean(map(abs, errors)) if errors else None,
        prediction_end_bias_ms=statistics.mean(errors) if errors else None,
        prediction_late_seals=sum(x < 0 for x in seal_slacks),
        prediction_seal_slack_ms=seal_slacks,
        seal_to_B_gpu_start_ms=seal_to_gpu,
        release_to_first_gpu_start_ms=leading,
        load_only_seals=len(load_only),
        load_only_hold_ms=sum(t.get("seal_hold_actual_ns", 0) for t in load_only) / 1e6,
        all_exposed_hold_ms=sum(
            uncovered(t["base_ready_ns"], t["seal_ns"], intervals) for t in holds
        )
        / 1e6,
        load_only_exposed_hold_ms=sum(
            uncovered(t["base_ready_ns"], t["seal_ns"], intervals) for t in load_only
        )
        / 1e6,
    )


def read(base):
    cells, provenance = [], {}
    for name in ROOTS:
        root = base / name
        if not root.exists():
            continue
        plan = json.loads((root / "plan.json").read_text())
        summary = root / "summary.json"
        status = json.loads(summary.read_text()) if summary.exists() else {}
        provenance[name] = dict(
            plan=plan,
            complete=status.get("complete", False),
            stopped=status.get("stopped", False),
        )
        for path in sorted((root / "cells").glob("*/*/result.json")):
            data = json.loads(path.read_text())
            if data["repeat"] == "warmup":
                continue
            assert data["accepted"], path
            assert sum(data["realized_outcomes"].values()) == data["requests"], path
            for direction in ("h2d", "d2h"):
                assert (
                    sum(data[f"{direction}_bytes_per_rank"].values())
                    == data[f"{direction}_bytes_all_ranks"]
                ), path
                verb = "load" if direction == "h2d" else "store"
                metric = data["measured_metric_deltas"].get(
                    f"vllm:kv_offload_{verb}_bytes_total"
                )
                assert metric == data[f"{direction}_bytes_all_ranks"], (path, metric)
            assert all(n == 8 for n in data["store_rank_coverage"]), path
            if data["workload"]["family"] != "evolving":
                assert data["realized_outcomes"] == {"cpu_hit": data["requests"]}, path
            data["source_result"] = str(path)
            data["timing_details"] = timing_details(data, root, status.get("complete"))
            cells.append(data)
    return cells, provenance


def aggregate(cells):
    groups = defaultdict(list)
    for d in cells:
        groups[(d["workload"]["id"], d["concurrency"], d["arm"])].append(d)
    rows = []
    for (spec, concurrency, arm), members in sorted(groups.items()):
        row = dict(
            workload=spec,
            family=members[0]["workload"]["family"],
            concurrency=concurrency,
            arm=arm,
            repeats=len(members),
            requests=sum(d["requests"] for d in members),
        )
        for metric in METRICS:
            values = [d[metric] for d in members]
            row[metric] = statistics.mean(values)
            row[metric + "_min"] = min(values)
            row[metric + "_max"] = max(values)
        for outcome in ("cpu_hit", "gpu_hit", "miss"):
            row[outcome + "_percent"] = (
                100
                * sum(d["realized_outcomes"].get(outcome, 0) for d in members)
                / row["requests"]
            )
        for direction in ("h2d", "d2h"):
            row[direction + "_gb_per_request_all_ranks"] = (
                sum(d[direction + "_bytes_all_ranks"] for d in members)
                / row["requests"]
                / 1e9
            )
        row["preemptions"] = sum(
            d["measured_metric_deltas"].get("vllm:num_preemptions_total", 0)
            for d in members
        )
        row["late_admissions_mean"] = statistics.mean(
            d["late_admissions"] for d in members
        )
        opportunities = [
            d["timing_details"]["readiness_opportunities"] for d in members
        ]
        covered = [
            o
            for o in opportunities
            if o.get("native_ready_coverage_complete")
            and o.get("admission_predecessor_classification_complete")
        ]
        row["complete_readiness_coverage_cells"] = len(covered)
        denominator = sum(o["ready_during_compute_definite"] for o in covered)
        row["late_admission_ratio_percent"] = (
            100
            * sum(o["late_admitted_during_compute_definite"] for o in covered)
            / denominator
            if len(covered) == len(members) and denominator
            else None
        )
        for metric in ("compute_rounds", "inter_compute_gaps_ms"):
            values = [d[metric] for d in members if d[metric] is not None]
            row[metric] = statistics.mean(values) if values else None
        rows.append(row)
    for row in rows:
        references = [
            r
            for r in rows
            if r["workload"] == row["workload"]
            and r["concurrency"] == row["concurrency"]
            and r["arm"] == "baseline"
        ]
        if not references:
            continue
        reference = references[0]
        for metric in METRICS:
            row[metric + "_gain_percent"] = 100 * (
                row[metric] / reference[metric] - 1
                if metric == "throughput_rps"
                else 1 - row[metric] / reference[metric]
            )
    return rows


def matched(cells):
    groups = defaultdict(dict)
    for d in cells:
        key = (d["workload"]["id"], d["concurrency"], d["seed"])
        assert d["arm"] not in groups[key], key
        groups[key][d["arm"]] = d
    pairs = []
    for key, members in sorted(groups.items()):
        if len(members) != 3:
            continue
        assert len({d["dataset_sha256"] for d in members.values()}) == 1, key
        assert (
            len(
                {
                    json.dumps(d["cache_budgets"], sort_keys=True)
                    for d in members.values()
                }
            )
            == 1
        ), key
        baseline = members["baseline"]
        for arm in ("worker", "full"):
            pairs.append(
                dict(
                    workload=key[0],
                    concurrency=key[1],
                    seed=key[2],
                    arm=arm,
                    throughput_gain_percent=100
                    * (members[arm]["throughput_rps"] / baseline["throughput_rps"] - 1),
                    mean_ttft_reduction_percent=100
                    * (1 - members[arm]["mean_ttft_ms"] / baseline["mean_ttft_ms"]),
                    p95_ttft_reduction_percent=100
                    * (1 - members[arm]["p95_ttft_ms"] / baseline["p95_ttft_ms"]),
                )
            )
    return pairs


def savefig(fig, output, name):
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(output / (name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plots(rows, output):
    bykey = {(r["workload"], r["concurrency"], r["arm"]): r for r in rows}
    fig, axes = plt.subplots(4, 4, figsize=(15, 11), sharex=True)
    for i, context in enumerate((4096, 16384, 32768, 65536)):
        for j, suffix in enumerate((128, 512, 2048, 8192)):
            ax = axes[i, j]
            spec = f"c{context}_q{suffix}"
            for arm in ARMS:
                points = sorted(
                    (n, d["throughput_rps"])
                    for (s, n, a), d in bykey.items()
                    if s == spec and a == arm
                )
                if points:
                    ax.plot(
                        *zip(*points), marker="o", ms=3, label=arm, color=COLORS[arm]
                    )
            ax.set_xscale("log", base=2)
            ax.set_xticks((1, 2, 4, 8, 16, 32, 64), labels=(1, 2, 4, 8, 16, 32, 64))
            ax.grid(alpha=0.2)
            ax.set_title(f"Cached {context // 1024}K / suffix {suffix}", fontsize=10)
            if j == 0:
                ax.set_ylabel("Requests / s")
            if i == 3:
                ax.set_xlabel("Concurrency")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("GLM5.3 TP8: bounded independent CPU-hit episodes (64 requests)")
    fig.tight_layout()
    savefig(fig, output, "homogeneous_concurrency")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, arm, reference in zip(axes, ("worker", "full"), ("baseline", "worker")):
        array = np.full((4, 4), np.nan)
        for i, context in enumerate((4096, 16384, 32768, 65536)):
            for j, suffix in enumerate((128, 512, 2048, 8192)):
                s = f"c{context}_q{suffix}"
                if (s, 32, arm) in bykey and (s, 32, reference) in bykey:
                    array[i, j] = 100 * (
                        bykey[s, 32, arm]["throughput_rps"]
                        / bykey[s, 32, reference]["throughput_rps"]
                        - 1
                    )
        image = ax.imshow(array, cmap="RdBu", vmin=-50, vmax=50)
        for (i, j), value in np.ndenumerate(array):
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.1f}%", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(4), labels=("128", "512", "2K", "8K"))
        ax.set_yticks(range(4), labels=("4K", "16K", "32K", "64K"))
        ax.set_xlabel("Uncached prompt suffix tokens")
        ax.set_ylabel("Cached prefix tokens")
        ax.set_title(f"{arm} vs {reference} (concurrency32)")
    fig.colorbar(image, ax=axes, label="Throughput gain (%)", shrink=0.8)
    savefig(fig, output, "homogeneous_gain_grid")
    mixed = sorted({r["workload"] for r in rows if r["family"] != "homogeneous"})
    if mixed:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        positions = np.arange(len(mixed))
        width = 0.24
        for ax, metric, label in zip(
            axes,
            ("throughput_rps", "mean_ttft_ms", "p95_ttft_ms"),
            ("Requests / s", "Mean TTFT (ms)", "p95 TTFT (ms)"),
        ):
            for offset, arm in enumerate(ARMS):
                data = [bykey.get((spec, 32, arm)) for spec in mixed]
                means = np.array([d[metric] if d else np.nan for d in data])
                ranges = np.array(
                    [
                        [
                            d[metric] - d[metric + "_min"],
                            d[metric + "_max"] - d[metric],
                        ]
                        if d
                        else [0, 0]
                        for d in data
                    ]
                ).T
                ax.bar(
                    positions + (offset - 1) * width,
                    means,
                    width,
                    label=arm,
                    color=COLORS[arm],
                    yerr=ranges,
                    capsize=2,
                )
            ax.set_xticks(
                positions,
                labels=[
                    s.replace("heterogeneous_", "mix ").replace("_", " ") for s in mixed
                ],
                rotation=20,
                ha="right",
            )
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle("GLM5.3 TP8, concurrency 32: repeat means and observed ranges")
        fig.tight_layout()
        savefig(fig, output, "mixed_workload_ablation")
    evolving = sorted({r["workload"] for r in rows if r["family"] == "evolving"})
    if evolving:
        fig, axes = plt.subplots(
            1, len(evolving), figsize=(5 * len(evolving), 4), squeeze=False
        )
        for ax, spec in zip(axes[0], evolving):
            bottom = np.zeros(3)
            for outcome, color in (
                ("gpu_hit", "#4f9b78"),
                ("cpu_hit", "#3976b8"),
                ("miss", "#bbb"),
            ):
                values = [
                    bykey.get((spec, 32, a), {}).get(outcome + "_percent", 0)
                    for a in ARMS
                ]
                ax.bar(ARMS, values, bottom=bottom, label=outcome, color=color)
                bottom += values
            ax.set_ylim(0, 100)
            ax.set_ylabel("Realized requests (%)")
            ax.set_title(spec.replace("_", " "))
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        savefig(fig, output, "evolving_cache_outcomes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/mnt/localnvme/jeongyoon"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cells, provenance = read(args.base)
    rows = aggregate(cells)
    pairs = matched(cells)
    coverage_path = args.base / "load_only_overlap_20261011/coverage_plan.json"
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    expected = coverage.get("expected_cells", 243)
    with (args.output / "cells.csv").open("w") as stream:
        columns = sorted(set().union(*(r for r in rows)))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        cells=len(cells),
        expected_cells=expected,
        full_illustrative_grid_and_mixed_cells=360,
        matched_arm_triplets=len(pairs) // 2,
        complete=len(cells) == expected
        and all(p["complete"] for p in provenance.values()),
        coverage_plan=coverage,
        rows=rows,
        timing_details=[
            dict(
                source=c["source_result"],
                arm=c["arm"],
                workload=c["workload"]["id"],
                concurrency=c["concurrency"],
                seed=c["seed"],
                **c["timing_details"],
            )
            for c in cells
        ],
        paired_gains=pairs,
        provenance=provenance,
        scope="64-request independent CPU-hit episodes; 300-request evolving streams. "
        "One grid repeat, two heterogeneous/evolving repeats. "
        "Means and ranges are descriptive, not confidence intervals.",
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    plots(rows, args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("cells", "expected_cells", "matched_arm_triplets", "complete")
            }
        )
    )


if __name__ == "__main__":
    main()
