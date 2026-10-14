# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Add causal mixed last snapshots to training without changing old dev groups."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.setup_prediction_20261011.train import runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    summary = json.loads((args.root / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"]
    shapes, seeds = {}, {}
    for episode in summary["results"]:
        if episode["workload"] != "mixed_supported":
            continue
        seeds[episode["repeat"]] = episode["seed"]
        directory = (
            args.root
            / "live"
            / "mixed_supported"
            / f"r{episode['repeat']}_{episode['label']}"
        )
        for line in (directory / "engine.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "step_completion" and event.get("selected"):
                shapes[episode["label"], episode["repeat"], event["step"]] = event[
                    "request_shapes"
                ]
    rows = json.loads(gzip.decompress(args.original.read_bytes()))
    frames = json.loads(
        gzip.decompress((args.root / "setup_rows.json.gz").read_bytes())
    )["frames"]
    counts = Counter()
    for row in frames:
        if row["workload"] != "mixed_supported" or not row["last"]:
            continue
        final = shapes[row["arm"], row["repeat"], row["step"]]
        actual = runtime.setup_features(final, row["inputs"][7], np.zeros(28))
        if actual is None or not np.array_equal(
            actual[:7], np.asarray(row["inputs"][:7], np.float32)
        ):
            counts["discarded_geometry_mismatch"] += 1
            continue
        group = f"{args.root.name}:mixed_supported:{seeds[row['repeat']]}"
        rows.append(
            dict(
                row,
                source=args.root.name,
                group=group,
                batch=f"{group}:{row['arm']}:{row['step']}",
                split="train",
                features=row["inputs"],
                shapes=final,
                final_shapes=final,
                weight=1.0,
                first=True,
            )
        )
        counts[f"seed{seeds[row['repeat']]}_{row['arm']}"] += 1
    assert sum(counts.values()) > 100
    (args.output / "pool.json.gz").write_bytes(
        gzip.compress(json.dumps(rows).encode(), mtime=0)
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                original_sha256=hashlib.sha256(args.original.read_bytes()).hexdigest(),
                additional_counts=dict(counts),
                retained_original_dev_and_final=True,
                added_labels_are_training_not_holdout=True,
                causal_proposal_inputs=True,
                future_geometry_used_for_filter_only=True,
                additional_weight_one_per_batch=True,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
