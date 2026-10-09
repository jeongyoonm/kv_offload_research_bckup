# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Produce a small, auditable overview without copying full response traces."""

import argparse
import json
from pathlib import Path

BASE = Path("/mnt/localnvme/jeongyoon")
METRICS = (
    "throughput_rps",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "compute_rounds",
    "compute_envelope_ms",
    "inter_compute_gap_ms",
    "hold_without_compute_ms",
    "pre_forward",
    "forward_to_gpu_start",
    "deferred_store_submission_begin_exposed_ms",
    "store_copy_enqueue_begin_full_ms",
    "store_copy_enqueue_begin_exposed_ms",
    "recovered_opportunity_percent",
)
RUNS = {
    "sf1": "Scoped reuse fences",
    "se2": "Specific producer events",
    "sa1": "Allocator avoidance",
    "sc1": "Store descriptor sorting",
    "sr1": "Future load reservations",
    "sp1": "Critical stores first",
    "so1": "Ascending GPU assignment",
    "sn0": "Synchronous store control",
    "snf": "Native store, full submission",
    "sn8": "Native store, 8 MiB chunks",
    "st0": "Continuous block-major control",
    "st1": "Continuous tensor-major only",
    "stm": "Continuous tensor-major plus merging",
    "stc": "Continuous sorted recycled CPU rows",
    "si1": "Independent load ready stream",
    "sv1": "Preencoded allocator control",
    "sdo1": "Preserved descriptor order",
    "snl": "Native D2H setup lead recalibration",
}


def overview():
    result = {}
    for name, label in RUNS.items():
        root = BASE / name
        item = dict(label=label, root=str(root), status="not_started")
        summary = root / "summary.json"
        if summary.exists():
            data = json.loads(summary.read_text())
            item.update(
                status="complete" if data["complete"] else "failed",
                stopped=data["stopped"],
                measured_cells=len(data.get("results", [])),
                allocation_seconds=data.get("seconds"),
            )
        elif root.exists():
            item["status"] = "running"
            progress = root / "progress.json"
            if progress.exists():
                data = json.loads(progress.read_text())
                item["measured_cells"] = data["cells"]
        path = root / "comparison.json"
        if path.exists():
            data = json.loads(path.read_text())
            item["means"] = {
                arm: {k: v for k, v in values.items() if k in METRICS}
                for arm, values in data["means"].items()
            }
            item["paired_changes_percent"] = data.get("paired_changes_percent")
            item["all_episodes_accepted"] = all(
                episode["accepted"] for episode in data["episodes"]
            )
            item["requests_by_arm"] = {
                arm: sum(
                    episode["requests"]
                    for episode in data["episodes"]
                    if episode["label"] == arm
                )
                for arm in data["means"]
            }
        result[name] = item
    for name in ("sg1", "sg2", "sg3", "sg4"):
        path = BASE / name / "ceiling_summary.json"
        if path.exists():
            data = json.loads(path.read_text())
            result[name] = dict(
                label="GPU/CPU resident reference",
                rows=[
                    {k: v for k, v in row.items() if k != "per_repeat"}
                    for row in data["rows"]
                ],
                definitions=data["definitions"],
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = overview()
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v.get("status", "reference") for k, v in data.items()}))


if __name__ == "__main__":
    main()
