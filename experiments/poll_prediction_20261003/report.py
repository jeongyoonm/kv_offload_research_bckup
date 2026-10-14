# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize matched controls and preserve every accepted polling cell."""

import argparse
import csv
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(root):
    grouped = defaultdict(list)
    for path in sorted(root.glob("*/p*/*/analysis.json")):
        row = json.loads(path.read_text())
        if row["accepted"]:
            policy = path.parent.name.split("_", 1)[1].split("_warmretry")[0]
            grouped[(path.parents[2].name, path.parents[1].name, policy)].append(row)
    rows = []
    for (phase, workload, policy), cells in grouped.items():
        row = dict(phase=phase, workload=workload, policy=policy, repeats=len(cells))
        for key in (
            "throughput",
            "mean_ttft_ms",
            "p95_ttft_ms",
            "compute_calls",
            "adaptive_sleeps",
            "adaptive_poll_cpu_ms",
            "hybrid_coarse_sleeps",
        ):
            row[key] = statistics.mean(c.get(key, 0) for c in cells)
        row["throughput_min"] = min(c["throughput"] for c in cells)
        row["missing_execution_timing_batches"] = sum(
            len(c.get("missing_execution_timing_steps", [])) for c in cells
        )
        row["throughput_max"] = max(c["throughput"] for c in cells)
        row["exposed_hold_ms"] = statistics.mean(
            c["exposed_lower_ms"].get("total", 0) for c in cells
        )
        opportunities = sum(c["kv_ready_during_compute"] for c in cells)
        recovered = sum(c["recovered_compute_opportunities"] for c in cells)
        row.update(
            opportunities=opportunities,
            recovered=recovered,
            recovery_percent=100 * recovered / opportunities if opportunities else None,
        )
        predictions = [p for c in cells for p in c["prediction_samples"]]
        for key in ("preparation_reserve_ms", "actual_b_preparation_ms"):
            values = [p[key] for p in predictions if p.get(key) is not None]
            row[key] = statistics.median(values) if values else None
        row["cells"] = [c["path"] for c in cells]
        rows.append(row)
    for row in rows:
        for control in ("fixed", "worker", "prediction_1ms", "readiness_1ms"):
            reference = next(
                (
                    c
                    for c in rows
                    if c["phase"] == row["phase"]
                    and c["workload"] == row["workload"]
                    and c["policy"] == control
                ),
                None,
            )
            row[f"throughput_gain_vs_{control}_percent"] = (
                100 * (row["throughput"] / reference["throughput"] - 1)
                if reference
                else None
            )
            if control == "fixed":
                for key in ("mean_ttft_ms", "p95_ttft_ms"):
                    row[f"{key}_reduction_percent"] = (
                        100 * (1 - row[key] / reference[key]) if reference else None
                    )
    out = root / "summary"
    out.mkdir(exist_ok=True)
    (out / "cells.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not rows:
        return
    with (out / "cells.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in rows[0] if k != "cells"])
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k != "cells"} for r in rows)
    columns = [
        "phase",
        "workload",
        "policy",
        "repeats",
        "throughput",
        "throughput_gain_vs_fixed_percent",
        "throughput_gain_vs_worker_percent",
        "mean_ttft_ms",
        "p95_ttft_ms",
        "compute_calls",
        "missing_execution_timing_batches",
        "recovery_percent",
        "exposed_hold_ms",
        "adaptive_sleeps",
        "adaptive_poll_cpu_ms",
        "hybrid_coarse_sleeps",
        "preparation_reserve_ms",
        "actual_b_preparation_ms",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append(
            "\t".join(
                f"{row[k]:.2f}" if isinstance(row[k], float) else str(row[k])
                for k in columns
            )
        )
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    body = (
        "<table><thead><tr>"
        + "".join(f"<th>{k}</th>" for k in columns)
        + "</tr></thead>"
    )
    for row in rows:
        body += (
            "<tr>"
            + "".join(
                "<td>"
                + html.escape(
                    f"{row[k]:.2f}" if isinstance(row[k], float) else str(row[k])
                )
                + "</td>"
                for k in columns
            )
            + "</tr>"
        )
    body += "</table>"
    (out / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>GLM TP8 polling study</title>'
        "<style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse}"
        "td,th{padding:8px;border:1px solid #ddd}"
        "th{position:sticky;top:0;background:#eee}"
        "</style><h1>GLM 5.3 TP8: polling and prediction</h1>"
        "<p>Worker is worker-optimized with late admission disabled. Fixed is 200 ms "
        "+ conditional 100 ms extension. Compare controls within each phase/workload. "
        "Short cells use 300 requests; long cells use 80. "
        "Concurrency 32, one output token. Single-replay screens are exploratory. "
        "Repeated throughput is an arithmetic mean; p95 is the mean of replay p95. "
        "Recovery denominators change with batching. Poll CPU excludes final, "
        "pre-ack and fixed iterations; preparation is wall time. "
        "One confirmation readiness-1ms short batch lacks rank6 GPU timing; "
        "GPU interval/admission diagnostics omit that interval. Batch counts, "
        "token work, raw throughput and TTFT retain the batch and its requests.</p>"
        + body
    )
    print(json.dumps({"groups": len(rows), "cells": sum(r["repeats"] for r in rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
