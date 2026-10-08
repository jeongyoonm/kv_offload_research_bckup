# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze confirmation nominees from the completed interference screen."""

import argparse
import json
import math
import statistics
from pathlib import Path


def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "plan.json").read_text())
    assert not plan.get("diagnostic_trace"), "Do not select from profiler timing"
    result = json.loads((root / "comparison.json").read_text())
    rows = {
        (r["shape"], r["profile"], r["chunk_mib"], r["depth"]): r
        for r in result["aggregate"]
        if r["mode"] == "concurrent"
    }
    profiles = [(s, p) for s in plan["shapes"] for p in ("4k", "16k", "60k")]
    scores = []
    for chunk, depth in plan["configs"]:
        ratios = {
            metric: [
                rows[s, p, 128, 2][metric] / rows[s, p, chunk, depth][metric]
                for s, p in profiles
            ]
            for metric in ("joint_finish_ms", "forward_enqueue_ms", "first_ready_ms")
        }
        scores.append(
            dict(
                chunk_mib=chunk,
                depth=depth,
                joint_speedup=geometric(ratios["joint_finish_ms"]),
                worst_joint_speedup=min(ratios["joint_finish_ms"]),
                enqueue_speedup=geometric(ratios["forward_enqueue_ms"]),
                first_ready_speedup=geometric(ratios["first_ready_ms"]),
            )
        )
    scores.sort(key=lambda r: r["joint_speedup"], reverse=True)
    best = scores[0]
    eligible = [
        r
        for r in scores
        if r["worst_joint_speedup"] >= 0.97
        and r["joint_speedup"] >= best["joint_speedup"] * 0.97
    ]
    ready = max(eligible, key=lambda r: r["first_ready_speedup"]) if eligible else best
    nominees = [(128, 2)]
    for r in (best, ready):
        key = (r["chunk_mib"], r["depth"])
        if key not in nominees:
            nominees.append(key)
    data = dict(
        scores=scores,
        joint_nominee=best,
        readiness_nominee=ready,
        confirmation_configs=nominees,
        rule="Equal weight across nine compute/transfer geometries; best joint "
        "completion plus fastest first readiness among candidates within 3% of "
        "best joint score and no geometry over 3% slower than current. "
        "These are nominees, not production recommendations.",
    )
    (root / "selection.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
