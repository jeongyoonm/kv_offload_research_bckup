# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write comparison tables, preserving independent and paired run definitions."""

import argparse
import json
import statistics
from pathlib import Path

from experiments.store_overlap_overnight_20261009.summarize import BASE, overview


def matched_gain(candidate, control):
    reference = {r["repeat"]: r for r in control["episodes"]}
    values = []
    for row in candidate["episodes"]:
        base = reference[row["repeat"]]
        assert row["seed"] == base["seed"] and row["requests"] == base["requests"]
        values.append(100 * (row["throughput_rps"] / base["throughput_rps"] - 1))
    return statistics.mean(values), values


def read_run(name):
    root = BASE / name
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"]
    data = json.loads((root / "comparison.json").read_text())
    assert all(row["accepted"] for row in data["episodes"])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current = overview()
    tables = {}
    lines = [
        "# GLM 5.3 TP8 store-overlap exploration",
        "",
        "All measurements preserve original model execution, FP8 KV and fixed",
        "cache capacity. Warm priming and startup are excluded. Experiments are",
        "descriptive; the small repeat counts do not establish statistical",
        "significance. Separate controls and acceptance checks are retained.",
        "",
        "## Paired 75-request experiments",
        "",
        "Each arm uses four independently primed 75-request episodes. The gain",
        "is the mean of same-seed throughput changes. Gaps are conservative",
        "all-rank inter-compute gaps per episode, not an additive CPU breakdown.",
        "",
        "| Proposal | Control req/s | Candidate req/s | Gain % | Gaps ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    paired = []
    for name in ("sf1", "se2", "sa1", "sc1", "sr1", "sp1", "so1", "sv1", "sdo1"):
        item = current[name]
        if item["status"] != "complete":
            continue
        data = read_run(name)
        a, b = data["means"]["control"], data["means"]["gpu"]
        gain = data["paired_changes_percent"]["gpu"]["throughput_rps"]
        label = item["label"]
        lines.append(
            f"| {label} | {a['throughput_rps']:.3f} | "
            f"{b['throughput_rps']:.3f} | {gain:+.2f} | "
            f"{a['inter_compute_gap_ms']:.1f} → {b['inter_compute_gap_ms']:.1f} |"
        )
        paired.append(dict(root=name, label=label, control=a, candidate=b, gain=gain))
    lines.extend(
        [
            "",
            "Store sorting merged no descriptors in block-major storage. One",
            "coalescing repeat coincided with unintended GPU test allocations;",
            "its audit is retained and it is not used as evidence for merging.",
            "Preencoded-client results use a changed client harness in both arms",
            "and are interpreted separately from the original arrival pattern.",
            "",
            "## Native D2H submission",
            "",
            "Modes use independent allocations with the same four seeds and",
            "300 measured requests each. The gain compares matching seed/size",
            "episodes across allocations, rather than alternating in one server.",
            "",
            "| Mode | Req/s | Gain % | Gaps ms | Exposed pre-fwd ms |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    native = []
    if current["sn0"]["status"] == "complete":
        reference = read_run("sn0")
        for name in ("sn0", "snf", "sn8"):
            if current[name]["status"] != "complete":
                continue
            data = read_run(name)
            means = next(iter(data["means"].values()))
            gain, changes = matched_gain(data, reference)
            label = current[name]["label"]
            lines.append(
                f"| {label} | {means['throughput_rps']:.3f} | {gain:+.2f} | "
                f"{means['inter_compute_gap_ms']:.1f} | {means['pre_forward']:.1f} |"
            )
            native.append(dict(root=name, means=means, gain=gain, changes=changes))
    lines.extend(
        [
            "",
            "## Continuous CPU-cache workload",
            "",
            "Each allocation uses two continuous 300-request episodes with 75",
            "prefixes and four distinct suffixes per prefix. CPU restoration is",
            "required on every request; GPU prefix hits and preemptions reject",
            "a cell. There is one reset before each episode, with none inside it.",
            "",
            "| Mode | Req/s | Gain % | Gaps ms | Mean TTFT ms |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    continuous = []
    if current["st0"]["status"] == "complete":
        reference = read_run("st0")
        for name in ("st0", "st1", "stm", "stc"):
            if current[name]["status"] != "complete":
                continue
            data = read_run(name)
            means = next(iter(data["means"].values()))
            gain, changes = matched_gain(data, reference)
            label = current[name]["label"]
            lines.append(
                f"| {label} | {means['throughput_rps']:.3f} | {gain:+.2f} | "
                f"{means['inter_compute_gap_ms']:.1f} | {means['mean_ttft_ms']:.1f} |"
            )
            continuous.append(dict(root=name, means=means, gain=gain, changes=changes))
    independent = None
    if current["si1"]["status"] == "complete":
        independent = read_run("si1")
        lines.extend(
            [
                "",
                "## Guarded independent load readiness",
                "",
                "Two matched continuous 300-request episodes per arm: 600 requests",
                "each. Both arms retain the same tensor-major physical optimization.",
                "",
                "| Mode | Req/s | Gaps ms | Mean TTFT ms | p95 TTFT ms |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for arm, means in independent["means"].items():
            lines.append(
                f"| {arm} | {means['throughput_rps']:.3f} | "
                f"{means['inter_compute_gap_ms']:.1f} | "
                f"{means['mean_ttft_ms']:.1f} | {means['p95_ttft_ms']:.1f} |"
            )
        gain = independent["paired_changes_percent"]["gpu"]["throughput_rps"]
        lines.extend(["", f"Mean paired throughput change: {gain:+.2f}%."])
    lead = None
    if current["snl"]["status"] == "complete":
        lead = read_run("snl")
        lines.extend(
            [
                "",
                "## Native store deadline lead",
                "",
                "Two matched continuous 300-request episodes per arm. Native full",
                "store submission and the frozen predictor are fixed; only the",
                "setup lead differs. The 90 ms prediction subtraction is fixed.",
                "",
                "| Mode | Req/s | Gaps ms | Mean TTFT ms | p95 TTFT ms |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for arm, means in lead["means"].items():
            lines.append(
                f"| {arm} | {means['throughput_rps']:.3f} | "
                f"{means['inter_compute_gap_ms']:.1f} | "
                f"{means['mean_ttft_ms']:.1f} | {means['p95_ttft_ms']:.1f} |"
            )
        gain = lead["paired_changes_percent"]["gpu"]["throughput_rps"]
        lines.extend(["", f"Mean paired throughput change: {gain:+.2f}%."])
    lines.extend(["", "## Resident-cache references", ""])
    reference_rows = {}
    for name in ("sg1", "sg2", "sg3", "sg4"):
        if name not in current:
            continue
        lines.extend(
            [
                f"### {name}",
                "",
                "| Prompt | Working set | Req/s | Mean TTFT ms | p95 TTFT ms |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in current[name]["rows"]:
            lines.append(
                f"| {row['workload']} | {row['requests_per_episode']} | "
                f"{row['throughput_rps']:.3f} | {row['mean_ttft_ms']:.1f} | "
                f"{row['p95_ttft_ms']:.1f} |"
            )
        reference_rows[name] = current[name]["rows"]
        lines.append("")
    resident_comparisons = []
    if all(name in current for name in ("sg2", "sg3", "sg4")):
        references = {r["workload"]: r for r in current["sg4"]["rows"]}
        lines.extend(
            [
                "## Matched resident working sets",
                "",
                "Independent allocations use identical seeds, request counts,",
                "precision and cache capacities. GPU-only sg4 verifies full",
                "prefix hits and process-local THP disabling on every worker.",
                "CPU runs reset GPU cache and require full CPU restoration on",
                "every request. Mean and p95 TTFT pool measured requests.",
                "",
                "| CPU run | Prompt | CPU/GPU req/s | Rate % | Mean TTFT % |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for name in ("sg2", "sg3"):
            for row in current[name]["rows"]:
                gpu = references[row["workload"]]
                assert row["measured_requests"] == gpu["measured_requests"]
                assert row["requests_per_episode"] == gpu["requests_per_episode"]
                gain = 100 * (row["throughput_rps"] / gpu["throughput_rps"] - 1)
                ttft = 100 * (1 - row["mean_ttft_ms"] / gpu["mean_ttft_ms"])
                lines.append(
                    f"| {name} | {row['workload']} | "
                    f"{row['throughput_rps']:.3f}/{gpu['throughput_rps']:.3f} | "
                    f"{gain:+.2f} | {ttft:+.2f} |"
                )
                resident_comparisons.append(
                    dict(
                        cpu_root=name,
                        gpu_root="sg4",
                        workload=row["workload"],
                        throughput_change_percent=gain,
                        mean_ttft_reduction_percent=ttft,
                        p95_ttft_reduction_percent=(
                            100 * (1 - row["p95_ttft_ms"] / gpu["p95_ttft_ms"])
                        ),
                    )
                )
        lines.append("")
    lines.extend(
        [
            "GPU resident references have ordinary scheduling and preparation;",
            "they are measured references, not mathematical upper bounds. The",
            "75-prefix 60K GPU-only case exceeds the unchanged GPU capacity.",
            "The initial sg1 run did not explicitly audit process-local THP;",
            "the corrected sg4 reference sets and checks it on all GPU workers.",
            "",
            "## Progress and provenance",
            "",
        ]
    )
    for name, item in current.items():
        if "status" in item:
            lines.append(f"- {name}: {item['status']} — {item['label']}")
    lines.extend(
        [
            "",
            "Raw inputs, outputs, cache counters, timing events, launch commands,",
            "ownership records and failure logs remain under the short run roots.",
            "Verified archives and results are committed on each proposal branch.",
            "See plan.json for branch/workspace mapping and the process logs for",
            "the chronological sequence. Byte tests validate transfer geometry;",
            "they do not establish model accuracy for every serving workload.",
        ]
    )
    args.output.write_text("\n".join(lines) + "\n")
    tables.update(
        paired=paired,
        native=native,
        continuous=continuous,
        resident=reference_rows,
        resident_comparisons=resident_comparisons,
        native_lead=(
            dict(
                means=lead["means"],
                paired_changes_percent=lead["paired_changes_percent"],
            )
            if lead
            else None
        ),
        independent=(
            dict(
                means=independent["means"],
                paired_changes_percent=independent["paired_changes_percent"],
            )
            if independent
            else None
        ),
    )
    args.output.with_suffix(".json").write_text(json.dumps(tables, indent=2) + "\n")
    print(json.dumps(dict(report=str(args.output), complete_runs=len(paired))))


if __name__ == "__main__":
    main()
