# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze heterogeneous classes from worker-only transfer/compute grid profiles."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    profiles = []
    for path in (args.root / "cells").glob("*/*/result.json"):
        data = json.loads(path.read_text())
        if data["arm"] != "worker" or data["repeat"] == "warmup":
            continue
        assert data["accepted"]
        transfer = sum(data["h2d_job_slowest_rank_ms"]) / data["requests"]
        compute = data["gpu_envelope_ms"] / data["requests"]
        profiles.append(
            dict(
                id=data["workload"]["id"],
                shape=data["workload"]["shapes"][0],
                concurrency=data["concurrency"],
                transfer_ms_per_request=transfer,
                compute_ms_per_request=compute,
                ratio=transfer / compute,
                source=str(path),
            )
        )
    assert len(profiles) == 16, "Require the complete 16-shape worker grid"
    assert len({p["concurrency"] for p in profiles}) == 1
    heavy = max(profiles, key=lambda p: p["ratio"])
    compute = min(profiles, key=lambda p: p["ratio"])
    remaining = [p for p in profiles if p["id"] not in {heavy["id"], compute["id"]}]
    balanced = min(remaining, key=lambda p: abs(math.log(p["ratio"])))
    selected = dict(transfer_heavy=heavy, compute_heavy=compute, balanced=balanced)
    args.output.write_text(
        json.dumps({k: v["shape"] for k, v in selected.items()}, indent=2)
    )
    args.output.with_suffix(".profiles.json").write_text(
        json.dumps(
            dict(
                selected=selected,
                profiles=profiles,
                scope=(
                    "Relative demand extremes of this grid. Transfer service "
                    "is the sum of slowest-rank DMA durations per job; compute "
                    "is the all-rank forward envelope. This descriptive ratio "
                    "does not remove submission gaps or assume disjoint DMA jobs."
                ),
                balanced_comparable=0.5 <= balanced["ratio"] <= 2,
            ),
            indent=2,
        )
    )
    print(json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
