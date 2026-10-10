# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Show per-repeat load coverage and GPU-gap phases without causal attribution."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, largest = [], []
    for root in args.root:
        comparison = json.loads((root / "comparison.json").read_text())
        setup = json.loads((root / "setup_audit.json").read_text())
        initial = {(r["workload"], r["arm"], r["repeat"]): r for r in setup["episodes"]}
        for episode in comparison["episodes"]:
            key = episode["workload"], episode["label"], episode["repeat"]
            matching = [
                g
                for g in comparison["gaps"]
                if (g["workload"], g["arm"], g["repeat"]) == key
            ]
            by_reason = defaultdict(float)
            for gap in matching:
                by_reason[gap["seal_reason"]] += gap["gap_ms"]
            data = dict(
                allocation=root.name,
                workload=episode["workload"],
                arm=episode["label"],
                repeat=episode["repeat"],
                seed=episode["seed"],
                throughput_rps=episode["throughput_rps"],
                initial_loads=initial[key]["first_compute_load_state"][0],
                initial_load_tokens=initial[key]["first_compute_load_state"][1],
                initial_compute_sizes=initial[key]["compute_sizes"][:12],
                load_only_compute_seals=initial[key]["load_only_compute_seals"],
                compute_rounds=episode["compute_rounds"],
                gaps_ms=episode["inter_compute_gap_ms"],
                execution_envelope_ms=episode["compute_envelope_ms"],
                hold_without_compute_ms=episode["hold_without_compute_ms"],
                gap_by_seal_reason_ms=dict(by_reason),
                first_gap_ms=matching[0]["gap_ms"] if matching else None,
                scheduler_exposed_ms=episode["scheduler"],
            )
            rows.append(data)
            largest.extend(
                dict(
                    allocation=root.name,
                    workload=gap["workload"],
                    arm=gap["arm"],
                    repeat=gap["repeat"],
                    step=gap["step"],
                    gap_ms=gap["gap_ms"],
                    seal_reason=gap["seal_reason"],
                    phases_ms=gap["phases_ms"],
                )
                for gap in sorted(matching, key=lambda g: g["gap_ms"], reverse=True)[:3]
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "regime_analysis.json").write_text(
        json.dumps(
            dict(
                episodes=rows,
                largest_gaps=largest,
                interpretation="Initial load coverage is observed before setup "
                "forecasts govern the first batch. Associations with later "
                "load-only phases are diagnostic, not an isolated causal proof.",
                timing="Conservative execution-event envelopes, not pure GEMM.",
            ),
            indent=2,
        )
        + "\n"
    )
    text = [
        "Per-repeat pipeline diagnostics (execution-event envelopes)",
        "allocation workload arm repeat req/s initial_loads rounds gaps_ms "
        "hold_no_compute_ms load_only_compute_seals",
    ]
    text += [
        f"{r['allocation']} {r['workload']} {r['arm']} {r['repeat']} "
        f"{r['throughput_rps']:.3f} {r['initial_loads']} {r['compute_rounds']} "
        f"{r['gaps_ms']:.1f} {r['hold_without_compute_ms']:.1f} "
        f"{r['load_only_compute_seals']}"
        for r in rows
    ]
    (args.output / "regime_analysis.txt").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
