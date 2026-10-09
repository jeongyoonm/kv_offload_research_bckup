# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report the four-arm serving comparison and connector/deadline diagnostics."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from experiments.h2d8_scheduler_ab_20261009.gap_audit import PHASES, STAGES
from experiments.h2d8_scheduler_ab_20261009.report import diagnostics

ARMS = ("control", "bias", "setup", "both")
PAIRS = {
    "deferred_store_submission_begin": "deferred_store_submission_end",
    "store_flush_wait_begin": "store_flush_wait_end",
}


def mean(values):
    return statistics.mean(values) if values else None


def worker_records(root):
    stages = defaultdict(dict)
    spans = defaultdict(lambda: defaultdict(list))
    for path in (root / "server/timing/queue_stages").glob("*.json"):
        export = json.loads(path.read_text())
        rank = export.get("rank", -1)
        if rank < 0:
            continue
        assert not export["overflow"], path
        pending = {}
        for event in export["events"]:
            name = event["event"]
            if name in STAGES:
                stages[event["step"], rank][name] = event["time_ns"]
            elif name in PAIRS:
                assert PAIRS[name] not in pending, event
                pending[PAIRS[name]] = event
            elif name in PAIRS.values():
                begin = pending.pop(name)
                assert event["time_ns"] >= begin["time_ns"]
                spans[rank][begin["event"]].append(
                    (begin["time_ns"], event["time_ns"], begin.get("jobs", []))
                )
        assert not pending, path
    return stages, spans


def audit_cell(directory, stages, spans):
    events = list(
        map(json.loads, (directory / "engine.jsonl").read_text().splitlines())
    )
    seals = {
        e["step"]: e["transport"]
        for e in json.loads((directory / "late.json").read_text())["records"]
        if e["event"] == "membership_sealed"
    }
    computes = sorted(
        [e for e in events if e["event"] == "step_completion" and e.get("selected")],
        key=lambda e: e["step"],
    )
    rows = []
    for previous, current in zip(computes, computes[1:]):
        step = current["step"]
        end = max(r[3] + r[6] for r in previous["execution_ranks"])
        start = max(r[3] for r in current["execution_ranks"])
        gap_end = max(end, start)
        rank_stages = {str(r): stages[step, r] for r in range(8)}
        points = [max(stages[step, r][s] for r in range(8)) for s in STAGES]
        bounds = [
            end,
            current["schedule_begin_ns"],
            current["dispatch_ns"],
            *points,
            start,
        ]
        clipped = [max(end, min(gap_end, t)) for t in bounds]
        assert sorted(clipped) == clipped, (directory, step, bounds)
        phases = {
            name: (b - a) / 1e6 for name, a, b in zip(PHASES, clipped, clipped[1:])
        }
        assert abs(sum(phases.values()) - (gap_end - end) / 1e6) < 1e-6
        store_times = {}
        for kind in PAIRS:
            rank_values = []
            for rank in range(8):
                meta = stages[step, rank]["attention_metadata_ready"]
                forward = stages[step, rank]["forward_submission_begin"]
                selected = [s for s in spans[rank][kind] if meta <= s[0] <= forward]
                assert all(s[1] <= forward for s in selected), (step, rank, kind)
                rank_values.append(
                    dict(
                        rank=rank,
                        full_ms=sum(b - a for a, b, _ in selected) / 1e6,
                        exposed_ms=sum(
                            max(0, min(start, b) - max(end, a)) for a, b, _ in selected
                        )
                        / 1e6,
                        jobs=[job for _, _, jobs in selected for job in jobs],
                    )
                )
            store_times[kind] = rank_values
        transport = seals.get(step, {})
        prediction = transport.get("prediction")
        deadline = transport.get("compute_seal_deadline_ns")
        raw_error = corrected_error = None
        if prediction and deadline:
            assert prediction["predecessor_step"] == previous["step"]
            raw_error = (
                prediction["gpu_start_upper_ns"]
                + prediction["raw_scaled_duration_ns"]
                - end
            ) / 1e6
            corrected_error = (
                prediction["gpu_start_upper_ns"]
                + prediction["predicted_duration_ns"]
                - end
            ) / 1e6
        rows.append(
            dict(
                step=step,
                preceding_step=previous["step"],
                gap_ms=(gap_end - end) / 1e6,
                envelope_overlap_ms=max(0, end - start) / 1e6,
                phases_ms=phases,
                A_end_ns=end,
                B_start_upper_ns=start,
                requests=len(current["selected"]),
                worker_stages=rank_stages,
                dispatch_before_A_end=current["dispatch_ns"] < end,
                forward_before_A_end=points[-1] < end,
                seal_reason=transport.get("seal_reason"),
                seal_minus_A_end_ms=(transport["seal_ns"] - end) / 1e6
                if transport.get("seal_ns")
                else None,
                deadline_minus_A_end_ms=(deadline - end) / 1e6 if deadline else None,
                raw_prediction_error_ms=raw_error,
                corrected_prediction_error_ms=corrected_error,
                store_times=store_times,
            )
        )
    totals = {name: sum(r["phases_ms"][name] for r in rows) for name in PHASES}
    for kind in PAIRS:
        for timing in ("full_ms", "exposed_ms"):
            totals[kind + "_" + timing] = sum(
                max(rank[timing] for rank in row["store_times"][kind]) for row in rows
            )
    totals.update(
        compute_transitions=len(rows),
        dispatched_before_A_end=sum(r["dispatch_before_A_end"] for r in rows),
        forward_before_A_end=sum(r["forward_before_A_end"] for r in rows),
    )
    return rows, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"]
    stages, spans = worker_records(root)
    episodes, gaps = [], []
    for result in summary["results"]:
        assert result["accepted"] and result["frozen_unchanged"]
        directory = root / "live/familiar60k" / f"r{result['repeat']}_{result['label']}"
        row = dict(result)
        row.update(diagnostics(directory))
        detail, totals = audit_cell(directory, stages, spans)
        assert (
            abs(sum(r["gap_ms"] for r in detail) - row["inter_compute_gap_ms"]) < 1e-6
        )
        row.update(totals)
        episodes.append(row)
        gaps.extend(dict(arm=row["label"], repeat=row["repeat"], **r) for r in detail)
    metrics = (
        "throughput_rps",
        "mean_ttft_ms",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "compute_rounds",
        "late_admissions",
        "kv_ready_during_compute",
        "recovered_compute_opportunities",
        "compute_envelope_ms",
        "inter_compute_gap_ms",
        "hold_without_compute_ms",
        "dispatched_before_A_end",
        "forward_before_A_end",
        "compute_transitions",
        *PHASES,
        *(
            kind + "_" + timing
            for kind in PAIRS
            for timing in ("full_ms", "exposed_ms")
        ),
    )
    groups = {
        arm: sorted(
            [r for r in episodes if r["label"] == arm], key=lambda r: r["repeat"]
        )
        for arm in ARMS
    }
    means, paired = {}, {}
    for arm, rows in groups.items():
        assert [r["seed"] for r in rows] == [r["seed"] for r in groups[ARMS[0]]]
        means[arm] = {m: statistics.mean(r[m] for r in rows) for m in metrics}
        opportunities = sum(r["kv_ready_during_compute"] for r in rows)
        means[arm]["recovered_opportunity_percent"] = (
            100
            * sum(r["recovered_compute_opportunities"] for r in rows)
            / opportunities
            if opportunities
            else None
        )
        arm_gaps = [r for r in gaps if r["arm"] == arm]
        for metric in (
            "raw_prediction_error_ms",
            "corrected_prediction_error_ms",
            "deadline_minus_A_end_ms",
            "seal_minus_A_end_ms",
        ):
            values = [r[metric] for r in arm_gaps if r[metric] is not None]
            means[arm][metric] = mean(values)
            means[arm][metric + "_mae"] = mean([abs(v) for v in values])
            means[arm][metric + "_positive_count"] = sum(v > 0 for v in values)
            means[arm][metric + "_count"] = len(values)
        means[arm]["transition_seal_reasons"] = dict(
            Counter(r["seal_reason"] for r in arm_gaps)
        )
        paired[arm] = {
            m: 100
            * (
                math.exp(
                    statistics.mean(
                        math.log(r[m] / c[m]) for r, c in zip(rows, groups[ARMS[0]])
                    )
                )
                - 1
            )
            for m in ("throughput_rps", "mean_ttft_ms", "p95_ttft_ms")
        }
        paired[arm]["per_repeat"] = [
            dict(
                repeat=r["repeat"],
                throughput_change_percent=100
                * (r["throughput_rps"] / c["throughput_rps"] - 1),
            )
            for r, c in zip(rows, groups[ARMS[0]])
        ]
    result = dict(
        means=means,
        paired_changes_percent=paired,
        episodes=episodes,
        gaps=gaps,
        definitions=dict(
            phases="All-rank maximum host boundaries clipped to inter-compute gaps; "
            "these phases partition each gap. Overlapping envelopes have zero "
            "gap and their overlap is recorded separately.",
            store_times="Sum across compute transitions of the largest per-rank "
            "store call duration. Exposed clips each rank's calls to GPU gaps. "
            "These diagnostics overlap the pre-forward phase and are not additive "
            "with it; different rank maxima need not identify a single critical path.",
            admission_ratio="Requests admitted at any seal intersected with requests "
            "whose all-rank KV completion occurred inside any measured compute "
            "interval, divided by those ready requests. "
            "Not a per-seal optimality bound.",
            inference="Four matched descriptive repeats, no significance claim. "
            "No changes to trained weights or CUDA/store ordering.",
        ),
    )
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=("arm", *metrics))
        writer.writeheader()
        writer.writerows(
            dict(arm=arm, **{m: values[m] for m in metrics})
            for arm, values in means.items()
        )
    print(json.dumps(dict(means=means, paired_changes_percent=paired), indent=2))


if __name__ == "__main__":
    main()
