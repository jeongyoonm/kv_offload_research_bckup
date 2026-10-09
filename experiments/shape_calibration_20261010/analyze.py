# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit serving, calibration and missed deadline windows without GPU work."""

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from experiments.native_scheduler_ab_20261010.report import diagnostics, gaps


def error_metrics(values):
    return dict(
        count=len(values),
        mean=statistics.mean(values) if values else None,
        mae=statistics.mean(abs(v) for v in values) if values else None,
        maximum_absolute=max(map(abs, values)) if values else None,
    )


def cell_audit(directory, result):
    events = list(
        map(json.loads, (directory / "engine.jsonl").read_text().splitlines())
    )
    records = json.loads((directory / "late.json").read_text())["records"]
    computes = {
        e["step"]: e
        for e in events
        if e["event"] == "step_completion" and e.get("selected")
    }
    assert len(computes) == result["compute_rounds"]
    assert all(e["valid"] and len(e["execution_ranks"]) == 8 for e in computes.values())
    ordered_steps = sorted(computes)
    following = dict(zip(ordered_steps, ordered_steps[1:]))
    predictions = {}
    for event in records:
        if event["event"] == "seal_prediction":
            predecessor = event["predecessor_step"]
            predictions.setdefault(predecessor, event)
            if following.get(predecessor) == event["step"]:
                predictions[predecessor] = event
    rows = []
    for step, prediction in predictions.items():
        if step not in computes:
            continue
        execution = computes[step]
        raw = prediction["raw_scaled_duration_ns"] / 1e6
        corrected = prediction["predicted_duration_ns"] / 1e6
        truth = (
            max(r[3] + r[6] for r in execution["execution_ranks"])
            - prediction["gpu_start_upper_ns"]
        ) / 1e6
        features = prediction["predictor_input"]
        assert int(features[1]) == len(execution["selected"])
        rows.append(
            dict(
                workload=result["workload"],
                arm=result["label"],
                repeat=result["repeat"],
                run=str(directory.parents[2]),
                step=prediction["step"],
                A_step=step,
                requests=len(execution["selected"]),
                shapes=execution["request_shapes"],
                features=features,
                raw_ms=raw,
                true_ms=truth,
                bias_ms=raw - truth,
                corrected_error_ms=corrected - truth,
                applied_bias_ms=prediction["applied_prediction_bias_ns"] / 1e6,
                fallback=prediction["shape_calibration_fallback"],
                adjacent_compute=following.get(step) == prediction["step"],
            )
        )
    completions = {
        e["job_id"]: max(e["rank_completion_ns"])
        for e in events
        if e["event"] == "kv_all_rank_completion"
    }
    ready = {
        e["request"]: completions[e["job_id"]]
        for e in events
        if e["event"] == "kv_load_dispatched"
    }
    intervals = [
        (
            max(r[3] for r in e["execution_ranks"]),
            max(r[3] + r[6] for r in e["execution_ranks"]),
        )
        for e in computes.values()
    ]
    opportunities = {
        rid for rid, t in ready.items() if any(a <= t < b for a, b in intervals)
    }
    seals = [e for e in records if e["event"] == "membership_sealed"]
    activated = {rid for e in seals for rid in e["activated"]}
    assert len(opportunities) == result["kv_ready_during_compute"]
    assert len(opportunities & activated) == result["recovered_compute_opportunities"]
    eligible = {e["step"]: e for e in records if e["event"] == "seal_eligibility"}
    window, prepared = set(), set()
    for seal in seals:
        t = seal["transport"]
        p = t.get("prediction")
        if t.get("seal_reason") != "compute_seal_deadline" or not p:
            continue
        predecessor = computes.get(p["predecessor_step"])
        if predecessor is None:
            continue
        end = max(r[3] + r[6] for r in predecessor["execution_ranks"])
        for slot, rid in enumerate(eligible[seal["step"]]["candidates"]):
            if rid in t["cancelled"] or rid in seal["activated"]:
                continue
            if rid not in ready or not t["seal_ns"] < ready[rid] <= end:
                continue
            window.add(rid)
            status = t["statuses"][8 * (slot + 1) : 8 * (slot + 2)]
            if len(status) == 8 and all(s == 1 for s in status):
                prepared.add(rid)
    missed = opportunities - activated
    first = min(max(r[3] for r in e["execution_ranks"]) for e in computes.values())
    end = max(max(r[3] + r[6] for r in e["execution_ranks"]) for e in computes.values())
    client_start = min(r["start_ns"] for r in result["responses"])
    client_end = max(r["end_ns"] for r in result["responses"])
    adjacent = [r for r in rows if r["adjacent_compute"]]
    return dict(
        before_first_compute_ms=(first - client_start) / 1e6,
        gpu_span_ms=(end - first) / 1e6,
        after_last_compute_ms=(client_end - end) / 1e6,
        client_harness_overhead_ms=result["seconds"] * 1000
        - (client_end - client_start) / 1e6,
        raw_prediction=error_metrics([r["bias_ms"] for r in adjacent]),
        corrected_prediction=error_metrics([r["corrected_error_ms"] for r in adjacent]),
        calibration_fallback_count=sum(r["fallback"] is True for r in adjacent),
        nonadjacent_prediction_count=len(rows) - len(adjacent),
        deadline_window_misses=len(window & missed),
        prepared_deadline_window_misses=len(prepared & missed),
        all_missed_compute_opportunities=len(missed),
        seal_reasons=dict(
            Counter(e["transport"].get("seal_reason", "ordinary") for e in seals)
        ),
    ), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"]
    plan = json.loads((root / "plan.json").read_text())
    gaps.PAIRS.update(
        store_reuse_fence_enqueue_begin="store_reuse_fence_enqueue_end",
        store_async_enqueue_begin="store_async_enqueue_end",
        store_descriptors_begin="store_descriptors_end",
        store_stream_dependencies_begin="store_stream_dependencies_end",
        store_copy_enqueue_begin="store_copy_enqueue_end",
    )
    stages, spans = gaps.worker_records(root)
    episodes, all_rows, detailed_gaps = [], [], []
    for result in summary["results"]:
        assert result["accepted"] and result["frozen_unchanged"]
        directory = (
            root
            / "live"
            / result["workload"]
            / f"r{result['repeat']}_{result['label']}"
        )
        row = dict(result)
        row.update(diagnostics(directory))
        detail, totals = gaps.audit_cell(directory, stages, spans)
        assert (
            abs(sum(g["gap_ms"] for g in detail) - row["inter_compute_gap_ms"]) < 1e-6
        )
        row.update(totals)
        audit, predictions = cell_audit(directory, result)
        row.update(audit)
        episodes.append(row)
        all_rows.extend(predictions)
        detailed_gaps.extend(
            dict(workload=row["workload"], arm=row["label"], repeat=row["repeat"], **g)
            for g in detail
        )
    metrics = (
        "throughput_rps",
        "mean_ttft_ms",
        "p95_ttft_ms",
        "compute_rounds",
        "late_admissions",
        "kv_ready_during_compute",
        "recovered_compute_opportunities",
        "compute_envelope_ms",
        "inter_compute_gap_ms",
        "hold_without_compute_ms",
        "prepared_deadline_window_misses",
        "all_missed_compute_opportunities",
        "dispatched_before_A_end",
        "forward_before_A_end",
        "before_first_compute_ms",
        "gpu_span_ms",
        "after_last_compute_ms",
        "client_harness_overhead_ms",
        *gaps.PHASES,
        *(
            kind + "_" + timing
            for kind in gaps.PAIRS
            for timing in ("full_ms", "exposed_ms")
        ),
    )
    levels, comparisons = [], []
    for workload in plan["workloads"]:
        groups = {
            a: sorted(
                [r for r in episodes if r["workload"] == workload and r["label"] == a],
                key=lambda r: r["repeat"],
            )
            for a in plan["arms"]
        }
        control = groups[plan["arms"][0]]
        for arm, selected in groups.items():
            assert [r["seed"] for r in selected] == [r["seed"] for r in control]
            predictions = [
                r
                for r in all_rows
                if r["workload"] == workload
                and r["arm"] == arm
                and r["adjacent_compute"]
            ]
            opportunities = sum(r["kv_ready_during_compute"] for r in selected)
            levels.append(
                dict(
                    workload=workload,
                    arm=arm,
                    repeats=len(selected),
                    **{m: statistics.mean(r[m] for r in selected) for m in metrics},
                    recovered_percent=100
                    * sum(r["recovered_compute_opportunities"] for r in selected)
                    / opportunities
                    if opportunities
                    else None,
                    raw_mae_ms=error_metrics([r["bias_ms"] for r in predictions])[
                        "mae"
                    ],
                    corrected_mae_ms=error_metrics(
                        [r["corrected_error_ms"] for r in predictions]
                    )["mae"],
                    corrected_bias_ms=error_metrics(
                        [r["corrected_error_ms"] for r in predictions]
                    )["mean"],
                    prediction_count=len(predictions),
                    calibration_fallback_count=sum(
                        r["fallback"] is True for r in predictions
                    ),
                )
            )
            comparisons.append(
                dict(
                    workload=workload,
                    arm=arm,
                    **{
                        m + "_change_percent": 100
                        * (
                            math.exp(
                                statistics.mean(
                                    math.log(r[m] / c[m])
                                    for r, c in zip(selected, control)
                                )
                            )
                            - 1
                        )
                        for m in ("throughput_rps", "mean_ttft_ms", "p95_ttft_ms")
                    },
                    per_repeat_throughput=[
                        100 * (r["throughput_rps"] / c["throughput_rps"] - 1)
                        for r, c in zip(selected, control)
                    ],
                )
            )
    payload = dict(
        levels=levels,
        comparisons=comparisons,
        episodes=episodes,
        gaps=detailed_gaps,
        definitions="Arithmetic episode levels, paired geometric changes, "
        "episode p95 not pooled. "
        "Prediction metrics use forecasts governing adjacent compute dispatches, "
        "matching the training audit scope; nonadjacent rows remain marked. "
        "Gap phases partition measured conservative execution envelopes, "
        "not pure kernel time. "
        "Deadline-window misses are retrospective opportunities, "
        "not throughput counterfactuals.",
    )
    (root / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    (root / "prediction_rows.json.gz").write_bytes(
        gzip.compress((json.dumps(all_rows, indent=2) + "\n").encode(), mtime=0)
    )
    with (root / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(levels[0]))
        writer.writeheader()
        writer.writerows(levels)
    print(json.dumps(dict(levels=levels, comparisons=comparisons), indent=2))


if __name__ == "__main__":
    main()
