# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retain paired seed comparisons without treating two replays as significance."""

import json
import statistics
from collections import defaultdict

from summarize import ROOT, load


def compare():
    rows = load(ROOT / "summary/cells.json")
    groups = defaultdict(dict)
    for row in rows:
        if row["accepted"]:
            key = row["workload"], row["policy"]
            assert row["phase"] not in groups[key], key
            groups[key][row["phase"]] = row
    assert len(groups) == 24 and all(len(group) == 2 for group in groups.values())
    result = []
    for (workload, policy), cells in sorted(groups.items()):
        references = ["capped50_fine2", "readiness2"]
        if policy.startswith("full_"):
            margin, fine = policy.split("_")[1:]
            if margin == "margin0":
                references.append("full_margin50_" + fine)
            if fine != "fine2":
                references.append("full_" + margin + "_fine2")
        for reference in references:
            if policy == reference:
                continue
            controls = groups[workload, reference]
            paired = []
            for phase in ("screen", "confirmation"):
                value, control = cells[phase], controls[phase]
                paired.append(
                    dict(
                        phase=phase,
                        throughput_gain_percent=100
                        * (value["throughput"] / control["throughput"] - 1),
                        mean_ttft_reduction_percent=100
                        * (1 - value["mean_ttft_ms"] / control["mean_ttft_ms"]),
                        p95_ttft_reduction_percent=100
                        * (1 - value["p95_ttft_ms"] / control["p95_ttft_ms"]),
                        compute_round_difference=value["compute_calls"]
                        - control["compute_calls"],
                        polling_cpu_difference_ms=value["accounted_polling_cpu_ms"]
                        - control["accounted_polling_cpu_ms"],
                        exposed_sleep_difference_ms=value[
                            "sleep_after_stop_estimate_ms"
                        ]
                        - control["sleep_after_stop_estimate_ms"],
                    )
                )
            result.append(
                dict(
                    workload=workload,
                    policy=policy,
                    reference=reference,
                    paired=paired,
                    throughput_wins=sum(
                        pair["throughput_gain_percent"] > 0 for pair in paired
                    ),
                    mean_paired_throughput_gain_percent=statistics.mean(
                        pair["throughput_gain_percent"] for pair in paired
                    ),
                )
            )
    report = dict(
        note="Two seeds are a repeatability check, not a significance estimate.",
        comparisons=result,
    )
    (ROOT / "summary/paired_comparisons.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"Retained {len(result)} paired comparisons.")


if __name__ == "__main__":
    compare()
