# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare frozen models on causal last proposals matching physical B geometry."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.setup_prediction_20261011.train import metrics, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"]
    models = {
        name: runtime.SetupPredictor(json.loads(path.read_text()))
        for name, path in (
            ("baseline", args.baseline),
            ("snapshot", args.models / "mean.json"),
        )
    }
    shapes = {}
    for episode in summary["results"]:
        directory = (
            args.root
            / "live"
            / episode["workload"]
            / f"r{episode['repeat']}_{episode['label']}"
        )
        for line in (directory / "engine.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "step_completion" and event.get("selected"):
                key = (
                    episode["workload"],
                    episode["label"],
                    episode["repeat"],
                    event["step"],
                )
                shapes[key] = event["request_shapes"]
    frames = json.loads(
        gzip.decompress((args.root / "setup_rows.json.gz").read_bytes())
    )["frames"]
    rows, audit = [], Counter()
    for row in frames:
        if not row["last"] or row["inputs"] is None:
            continue
        actual = runtime.setup_features(
            shapes[row["workload"], row["arm"], row["repeat"], row["step"]],
            row["inputs"][7],
            np.zeros(28),
        )
        if actual is None or not np.array_equal(
            actual[:7], np.asarray(row["inputs"][:7], np.float32)
        ):
            audit["geometry_mismatch"] += 1
            continue
        values = {
            name: model.predict_ms(row["inputs"]) for name, model in models.items()
        }
        if any(value is None for value in values.values()):
            audit["unsupported"] += 1
            continue
        rows.append(dict(row, weight=1.0, first=True, values=values))
    assert rows
    results = {}
    for workload in ["overall", *sorted({r["workload"] for r in rows})]:
        selected = [
            r for r in rows if workload == "overall" or r["workload"] == workload
        ]
        results[workload] = dict(
            batches=len(selected),
            **{
                name: metrics(
                    selected,
                    np.array([r["values"][name] for r in selected]),
                    "target_ready_ms",
                )
                for name in models
            },
        )
    payload = dict(
        root=str(args.root),
        physical_last_snapshot=True,
        future_membership_in_inputs=False,
        uniform_batch_weight=True,
        audit=dict(audit),
        results=results,
    )
    (args.models / (args.root.name + "_evaluation.json")).write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
