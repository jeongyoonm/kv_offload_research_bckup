# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize fixed-fold fitting sensitivity without modifying served models."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/localnvme/glm53_offset_predictor_20261008")


def main():
    freeze = json.loads((ROOT / "freeze.json").read_text())
    for name, digest in freeze["models"].items():
        assert (
            hashlib.sha256((ROOT / "models" / f"{name}.json").read_bytes()).hexdigest()
            == digest
        )
    groups = defaultdict(list)
    selections = {}
    for seed in [73, 74, 75]:
        source = ROOT if seed == 73 else ROOT / f"seed{seed}"
        protocol = json.loads((source / "protocol.json").read_text())
        assert protocol["seed"] == seed
        assert not protocol["original_final_benchmark_used_for_training"]
        selections[seed] = json.loads((source / "selection.json").read_text())
        for row in json.loads((source / "cpu_comparison.json").read_text()):
            groups[(row["source"], row["name"])].append(dict(seed=seed, **row))
    summary = []
    for (source, name), rows in groups.items():
        assert len(rows) == 3
        values = [r["macro_mae_ms"] for r in rows]
        summary.append(
            dict(
                source=source,
                name=name,
                macro_mean_ms=float(np.mean(values)),
                macro_min_ms=min(values),
                macro_max_ms=max(values),
                rows=rows,
            )
        )
    result = dict(
        seeds=[73, 74, 75],
        groups=summary,
        selections=selections,
        serving_seed=73,
        serving_checkpoints_unchanged=True,
        scope="Same outer and inner grouped folds, data, weights and hyperparameters.",
    )
    (ROOT / "cpu_seed_robustness.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
