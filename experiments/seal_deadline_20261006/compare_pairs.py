# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize four paired 75-request episodes without pooling startup time."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def confidence(ratios):
    assert len(ratios) == 4
    logs = [math.log(r) for r in ratios]
    center = statistics.mean(logs)
    radius = 3.182446305284263 * statistics.stdev(logs) / math.sqrt(len(logs))
    return dict(
        paired_geometric_gain_pct=100 * (math.exp(center) - 1),
        paired_95pct_lower_gain_pct=100 * (math.exp(center - radius) - 1),
        paired_95pct_upper_gain_pct=100 * (math.exp(center + radius) - 1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = json.loads((args.root / "serving_comparison.json").read_text())
    groups = defaultdict(dict)
    for row in report["cells"]:
        repeat, _, policy = row["cell"].partition("_")
        policy = policy.partition("_warmretry")[0]
        assert repeat not in groups[policy]
        groups[policy][repeat] = row
    assert len(groups) == 2
    control = groups["compact_shadow"]
    assert len(control) == 4
    aggregate = []
    for policy, repeats in groups.items():
        assert set(repeats) == set(control)
        rows = list(repeats.values())
        assert all(row["requests"] == 75 for row in rows)
        aggregate.append(
            dict(
                policy=policy,
                episodes=4,
                requests=300,
                throughput=300 / sum(75 / row["throughput"] for row in rows),
                mean_ttft_ms=statistics.mean(row["mean_ttft_ms"] for row in rows),
                mean_cell_p95_ttft_ms=statistics.mean(
                    row["p95_ttft_ms"] for row in rows
                ),
                compute_rounds=sum(row["compute_rounds"] for row in rows),
                late_admissions=sum(row["late_admissions"] for row in rows),
                recovered_compute_opportunities=sum(
                    row["recovered_compute_opportunities"] for row in rows
                ),
                kv_ready_during_compute=sum(
                    row["kv_ready_during_compute"] for row in rows
                ),
                missed_ready_at_worker_entry_with_seal_ack=sum(
                    row["missed_ready_at_worker_entry_with_seal_ack"] for row in rows
                ),
                total_compute_span_ms=sum(row["compute_span_ms"] for row in rows),
                seal_reasons=dict(
                    sum((Counter(row["seal_reasons"]) for row in rows), Counter())
                ),
                mean_gap_lower_ms=statistics.mean(row["gap_lower_ms"] for row in rows),
                mean_gap_upper_ms=statistics.mean(row["gap_upper_ms"] for row in rows),
                mean_non_load_late_enqueue_lower_ms=statistics.mean(
                    row["non_load_late_enqueue_lower_ms"] for row in rows
                ),
                throughput_pair_statistics=confidence(
                    [
                        repeats[r]["throughput"] / control[r]["throughput"]
                        for r in control
                    ]
                ),
            )
        )
    result = dict(
        source_commit=report["source_commit"],
        aggregate=aggregate,
        scope=(
            "300 measured requests per policy in four independent 75-request "
            "episodes, with cache resets and excluded priming between episodes. "
            "This is not a continuous 300-request stream. Policy order alternates "
            "between pairs. The paired interval uses Student t with 3 degrees "
            "of freedom on log throughput ratios."
        ),
    )
    (args.root / "paired_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
