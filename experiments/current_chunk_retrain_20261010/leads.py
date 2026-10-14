# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report a frozen-predictor deadline sweep and observed setup boundaries."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def distribution(values):
    if not values:
        return dict(samples=0)
    ordered = sorted(values)
    return dict(
        samples=len(values),
        mean_ms=statistics.mean(values),
        median_ms=statistics.median(values),
        p95_ms=ordered[round(0.95 * (len(ordered) - 1))],
        max_ms=max(values),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "plan.json").read_text())
    assert plan["experiment_kind"] == "deadline_leads"
    assert len({s["sha256"] for s in plan["sources"].values()}) == 1
    reference = plan["policies"][plan["arms"][0]]
    for policy in plan["policies"].values():
        assert policy["late_seal_prediction_bias_us"] == 0
        assert {k for k in policy if policy[k] != reference[k]} <= {
            "late_seal_compute_deadline_lead_us"
        }
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"]
    comparison = json.loads((root / "comparison.json").read_text())
    accuracy = json.loads((root / "live_accuracy.json").read_text())
    forecasts = {(r["workload"], r["arm"]): r for r in accuracy["levels"]}
    seals = {}
    for episode in summary["results"]:
        directory = (
            root
            / "live"
            / episode["workload"]
            / f"r{episode['repeat']}_{episode['label']}"
        )
        records = json.loads((directory / "late.json").read_text())["records"]
        for event in records:
            if event["event"] == "membership_sealed":
                seals[event["step"]] = event["transport"]
    groups = defaultdict(list)
    for gap in comparison["gaps"]:
        groups[gap["workload"], gap["arm"]].append(gap)
    levels = []
    for level in comparison["levels"]:
        key = level["workload"], level["arm"]
        measurements = defaultdict(list)
        for gap in groups[key]:
            stage = gap["worker_stages"]
            entry = max(r["worker_entry"] for r in stage.values())
            forward = max(r["forward_submission_begin"] for r in stage.values())
            measurements["worker_entry_to_forward"].append((forward - entry) / 1e6)
            transport = seals.get(gap["step"], {})
            if not transport:
                continue
            seal = transport["seal_ns"]
            assert seal <= entry <= forward
            measurements["seal_to_worker_entry"].append((entry - seal) / 1e6)
            measurements["seal_to_forward"].append((forward - seal) / 1e6)
            measurements["seal_to_gpu_start"].append(
                (gap["B_start_upper_ns"] - seal) / 1e6
            )
            measurements["seal_minus_A_end"].append((seal - gap["A_end_ns"]) / 1e6)
            measurements["forward_minus_A_end"].append(
                (forward - gap["A_end_ns"]) / 1e6
            )
            if entry >= gap["A_end_ns"]:
                measurements["worker_setup_after_A"].append((forward - entry) / 1e6)
            if transport.get("seal_reason") == "compute_seal_deadline":
                measurements["deadline_seal_overshoot"].append(
                    max(0, seal - transport["compute_seal_deadline_ns"]) / 1e6
                )
        pair = next(
            r for r in comparison["comparisons"] if (r["workload"], r["arm"]) == key
        )
        levels.append(
            dict(
                **level,
                lead_ms=plan["policies"][level["arm"]][
                    "late_seal_compute_deadline_lead_us"
                ]
                / 1000,
                prospective_accuracy=forecasts[key]["prospective"],
                setup={k: distribution(v) for k, v in measurements.items()},
                paired=pair,
            )
        )
    payload = dict(
        levels=levels,
        definitions="Identical frozen predictor, zero correction, only lead varies. "
        "Forward submission begin is a CPU readiness proxy, not proof every GPU "
        "dependency is ready. Seal-to-entry includes ordered FIFO waiting behind A; "
        "seal-to-GPU-start includes remaining A execution and GPU dependencies. "
        "Worker-entry-to-forward excludes FIFO wait but can include preparation "
        "fences and store submission. These measurements do not directly provide "
        "a counterfactual minimal safe lead. Paired changes use geometric means "
        "relative to the first arm; two repeats are not confidence intervals.",
    )
    (root / "lead_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "Frozen submission-envelope predictor: deadline lead comparison",
        payload["definitions"],
        "",
        "workload lead_ms req/s meanTTFT_ms p95TTFT_ms gaps_ms recovery_% rounds "
        "MAE_ms paired_RPS_change_% per_repeat_changes_%",
    ]
    for r in levels:
        lines.append(
            f"{r['workload']} {r['lead_ms']:g} {r['throughput_rps']:.4f} "
            f"{r['mean_ttft_ms']:.2f} {r['p95_ttft_ms']:.2f} "
            f"{r['inter_compute_gap_ms']:.2f} {r['recovered_percent']:.2f} "
            f"{r['compute_rounds']:.1f} {r['prospective_accuracy']['mae_ms']:.3f} "
            f"{r['paired']['throughput_rps_change_percent']:+.3f} "
            f"{r['paired']['per_repeat_throughput']}"
        )
    (root / "LEAD_REPORT.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(levels, indent=2), flush=True)


if __name__ == "__main__":
    main()
