# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report paired serving outcomes and causal deadline errors from saved cells."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path("/mnt/localnvme/glm53_live_predictors_20261008")


def describe(values):
    return (
        None
        if not values
        else dict(
            count=len(values),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            p95=float(np.quantile(values, 0.95)),
            minimum=float(min(values)),
            maximum=float(max(values)),
        )
    )


def diagnostics(directory):
    events = [
        json.loads(line)
        for line in (directory / "engine.jsonl").read_text().splitlines()
    ]
    late = json.loads((directory / "late.json").read_text())["records"]
    completed = {
        e["step"]: e
        for e in events
        if e["event"] == "step_completion" and e.get("valid") and e.get("selected")
    }
    spans = {}
    for step, e in completed.items():
        ranks = e.get("execution_ranks", [])
        if len(ranks) == 8 and all(r[5] == 1 and r[6] > 0 for r in ranks):
            spans[step] = dict(
                start=max(r[3] for r in ranks),
                end=max(r[3] + r[6] for r in ranks),
                start_lower=max(r[2] for r in ranks),
                end_lower=max(r[7] for r in ranks),
                end_upper=max(r[8] for r in ranks),
                uncertainty=max(r[3] - r[2] for r in ranks),
            )
    intervals = sorted(spans.values(), key=lambda x: x["start"])
    gaps = [
        max(0, b["start"] - a["end"]) / 1e6 for a, b in zip(intervals, intervals[1:])
    ]
    gap_lower = sum(
        max(0, b["start_lower"] - a["end_upper"]) / 1e6
        for a, b in zip(intervals, intervals[1:])
    )
    gap_upper = sum(
        max(0, b["start"] - a["end_lower"]) / 1e6
        for a, b in zip(intervals, intervals[1:])
    )
    errors, costs, leads, enqueue_leads = [], [], [], []
    backends, reasons = Counter(), Counter()
    native_predictions = 0
    for e in late:
        if e["event"] == "seal_prediction":
            a = spans.get(e["predecessor_step"])
            if a:
                errors.append(
                    e["predicted_duration_ns"] / 1e6 - (a["end"] - a["start"]) / 1e6
                )
            costs.append(e["prediction_cost_ns"] / 1e3)
            backends[e.get("predictor_backend", "unknown")] += 1
            native_predictions += e.get("predictor_input") is not None
        elif e["event"] == "membership_sealed":
            transport = e.get("transport", {})
            reasons[transport.get("seal_reason", transport.get("path", "unknown"))] += 1
            a = spans.get((transport.get("prediction") or {}).get("predecessor_step"))
            if a:
                leads.append((a["end"] - transport["seal_ns"]) / 1e6)
                b = completed.get(e["step"])
                if b:
                    enqueue_leads.append((a["end"] - b["dispatch_ns"]) / 1e6)
    return dict(
        compute_rounds=len(spans),
        gpu_gap_ms=sum(gaps),
        gpu_gap_lower_ms=gap_lower,
        gpu_gap_upper_ms=gap_upper,
        gpu_compute_span_ms=sum((s["end"] - s["start"]) / 1e6 for s in spans.values()),
        prediction_errors_ms=errors,
        prediction_cost_us=costs,
        actual_seal_lead_ms=leads,
        actual_enqueue_lead_ms=enqueue_leads,
        backends=dict(backends),
        seal_reasons=dict(reasons),
        point_predictions=native_predictions,
    )


def main():
    cells = []
    for p in sorted((ROOT / "live").glob("*/r[0-3]_*/result.json")):
        result = json.loads(p.read_text())
        if not result["accepted"]:
            raise RuntimeError(f"rejected measured cell: {p}")
        result["diagnostics"] = diagnostics(p.parent)
        cells.append(result)
    groups = {}
    for workload in ["familiar60k", "short16k", "heterogeneous"]:
        for label in ["trees", "guarded_sparse_mlp", "envelope_sparse_mlp"]:
            selected = [
                c for c in cells if c["workload"] == workload and c["label"] == label
            ]
            if not selected:
                continue
            ds = [c["diagnostics"] for c in selected]
            times = [r["ttft_ms"] for c in selected for r in c["responses"]]
            errors = [v for d in ds for v in d["prediction_errors_ms"]]
            costs = [v for d in ds for v in d["prediction_cost_us"]]
            leads = [v for d in ds for v in d["actual_seal_lead_ms"]]
            recovered = sum(c["recovered_compute_opportunities"] for c in selected)
            ready = sum(c["kv_ready_during_compute"] for c in selected)
            groups[(workload, label)] = dict(
                workload=workload,
                predictor=label,
                episodes=len(selected),
                requests=sum(c["requests"] for c in selected),
                throughput_rps=sum(c["requests"] for c in selected)
                / sum(c["seconds"] for c in selected),
                mean_ttft_ms=float(np.mean(times)),
                p95_ttft_ms=float(np.quantile(times, 0.95)),
                compute_rounds=sum(d["compute_rounds"] for d in ds),
                mean_episode_gpu_gap_ms=float(np.mean([d["gpu_gap_ms"] for d in ds])),
                mean_episode_gpu_gap_lower_ms=float(
                    np.mean([d["gpu_gap_lower_ms"] for d in ds])
                ),
                mean_episode_gpu_gap_upper_ms=float(
                    np.mean([d["gpu_gap_upper_ms"] for d in ds])
                ),
                late_admissions=sum(c["late_admissions"] for c in selected),
                recovered_compute_opportunities=recovered,
                kv_ready_during_compute=ready,
                recovered_ready_percent=100 * recovered / ready if ready else None,
                prediction_mae_ms=float(np.mean(np.abs(errors))) if errors else None,
                prediction_bias_ms=float(np.mean(errors)) if errors else None,
                prediction_p95_absolute_ms=float(np.quantile(np.abs(errors), 0.95))
                if errors
                else None,
                prediction_cost_us=describe(costs),
                actual_seal_lead_ms=describe(leads),
                actual_enqueue_lead_ms=describe(
                    [v for d in ds for v in d["actual_enqueue_lead_ms"]]
                ),
                seal_reasons=dict(
                    sum((Counter(d["seal_reasons"]) for d in ds), Counter())
                ),
                predictor_backends=dict(
                    sum((Counter(d["backends"]) for d in ds), Counter())
                ),
            )
    for (workload, label), group in groups.items():
        reference = groups[(workload, "trees")]
        group["throughput_gain_percent"] = 100 * (
            group["throughput_rps"] / reference["throughput_rps"] - 1
        )
        for name in ["mean_ttft_ms", "p95_ttft_ms"]:
            group[name.replace("_ms", "_reduction_percent")] = 100 * (
                1 - group[name] / reference[name]
            )
        ratios = []
        for c in cells:
            if c["workload"] == workload and c["label"] == label:
                ref = next(
                    r
                    for r in cells
                    if r["workload"] == workload
                    and r["label"] == "trees"
                    and r["repeat"] == c["repeat"]
                )
                ratios.append(math.log(c["throughput_rps"] / ref["throughput_rps"]))
        if len(ratios) > 1:
            mean = float(np.mean(ratios))
            margin = float(
                t.ppf(0.975, len(ratios) - 1)
                * np.std(ratios, ddof=1)
                / math.sqrt(len(ratios))
            )
            group["paired_throughput_gain_percent"] = 100 * math.expm1(mean)
            group["paired_95pct_gain_bounds"] = [
                100 * math.expm1(mean - margin),
                100 * math.expm1(mean + margin),
            ]
    report = dict(
        complete=len(cells) == 36,
        measured_cells=len(cells),
        aggregate=list(groups.values()),
        scope=(
            "Four 75-request episodes per predictor/workload; throughput excludes "
            "priming/startup. Pooled p95 TTFT across 300 requests. Issued predictions "
            "have policy-dependent populations. Recovery counts distinct "
            "late-admitted requests whose KV became ready during some measured "
            "compute interval; it does not prove same-predecessor recovery. GPU "
            "gaps use all-rank start upper bounds and include load-only intervals."
        ),
    )
    (ROOT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Live predictor comparison: GLM 5.3 TP8",
        "",
        f"Complete: {report['complete']}; measured cells: {len(cells)}/36.",
        "",
        (
            "| Workload | Predictor | req/s | Throughput gain | Mean TTFT (ms) | "
            "p95 TTFT (ms) | Compute rounds | Mean GPU gap/episode (ms) | "
            "Issued MAE (ms) | Recovered KV-ready (%) |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in groups.values():
        mae = (
            "n/a" if g["prediction_mae_ms"] is None else f"{g['prediction_mae_ms']:.2f}"
        )
        recovery = (
            "n/a"
            if g["recovered_ready_percent"] is None
            else f"{g['recovered_ready_percent']:.1f}"
        )
        lines.append(
            f"| {g['workload']} | {g['predictor']} | {g['throughput_rps']:.3f} | "
            f"{g['throughput_gain_percent']:+.1f}% | {g['mean_ttft_ms']:.1f} | "
            f"{g['p95_ttft_ms']:.1f} | {g['compute_rounds']} | "
            f"{g['mean_episode_gpu_gap_ms']:.1f} | {mae} | {recovery} |"
        )
    lines.extend(
        [
            "",
            report["scope"],
            "",
            (
                "Checkpoint coverage differs as well as architecture. The control is "
                "the latest broader-data serving tree checkpoint. All policies use "
                "the same 5 ms deadline lead and optimized preparation. The original "
                "sparse selector is preserved. Prediction-cost and actual-seal-lead "
                "distributions and paired confidence intervals are in comparison.json."
            ),
            "",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
