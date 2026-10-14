# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay preparation reserves on recorded snapshots using past observations."""

import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
OLD = Path("/mnt/localnvme/glm53_adaptive_seal_20261002")
ROOT = Path("/mnt/localnvme/glm53_poll_prediction_20261003")
spec = importlib.util.spec_from_file_location(
    "reserve_model", SOURCE / "vllm/v1/kv_offload/seal_predictor.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    events, completed = [], {}
    trace = next((OLD / "adaptive_screen_v2/server/timing").glob("engine-*.jsonl"))
    for line in trace.read_text().splitlines():
        row = json.loads(line)
        if row["event"] == "step_completion":
            completed[row["step"]] = row
            events.append((row["time_ns"], "completed", row))
    for phase in ("adaptive_screen_v2", "final_confirmation"):
        for path in (OLD / phase).glob("*/*/late.json"):
            latest = {}
            for row in json.loads(path.read_text())["records"]:
                if row["event"] == "seal_preparation_prediction":
                    latest[row["step"]] = dict(
                        row, workload=path.parents[1].name, phase=phase
                    )
            events.extend((r["time_ns"], "query", r) for r in latest.values())
    estimator = module.ComputeEstimator()
    rows = []
    for _, kind, row in sorted(events, key=lambda e: e[0]):
        if kind == "completed":
            estimator.observe_preparation(row.get("ranks", []), row)
            continue
        upcoming = {"request_shapes": row["upcoming_shapes"]}
        if not upcoming["request_shapes"]:
            continue
        successor = completed.get(row["step"], {})
        records = successor.get("ranks", [])
        if len(records) != 8 or not all(r and r[2] >= r[1] > 0 for r in records):
            continue
        rows.append(
            dict(
                phase=row["phase"],
                workload=row["workload"],
                step=row["step"],
                old_ms=row["preparation_reserve_ns"] / 1e6,
                recent_ms=estimator.recent_preparation_reserve(10_000_000, upcoming)
                / 1e6,
                actual_ms=max(r[2] - r[1] for r in records) / 1e6,
                upcoming_shapes=row["upcoming_shapes"],
            )
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["workload"])].append(row)
    summary = []
    for (phase, workload), group in sorted(grouped.items()):
        item = dict(phase=phase, workload=workload, samples=len(group))
        for key in ("old_ms", "recent_ms", "actual_ms"):
            item[key] = statistics.median(r[key] for r in group)
        for key in ("old_ms", "recent_ms"):
            item[key + "_under_fraction"] = sum(
                r[key] < r["actual_ms"] for r in group
            ) / len(group)
        summary.append(item)
    (ROOT / "preparation_shadow.json").write_text(
        json.dumps(
            dict(
                note="Past-only reserves on recorded snapshots; actual successor "
                "membership comes from the old run. This is not live throughput.",
                summary=summary,
                samples=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
