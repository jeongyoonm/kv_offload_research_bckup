# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fit a frozen correction with episode-grouped validation and no elapsed feature."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from vllm.v1.kv_offload.seal_calibration import FEATURES


def design(rows):
    requests = np.array([r["requests"] for r in rows])
    context = np.array([r["features"][3] for r in rows])
    raw = np.array([r["raw_ms"] for r in rows])
    return np.c_[
        np.ones(len(rows)),
        context / requests / 60000,
        requests / 16,
        context / 960000,
        raw / 1000,
    ]


def solve(x, y, alpha):
    return np.linalg.solve(x.T @ x + np.diag([0, alpha, alpha, alpha, alpha]), x.T @ y)


def metrics(error):
    return dict(
        count=len(error),
        mean=float(np.mean(error)),
        mae=float(np.mean(abs(error))),
        p90_absolute=float(np.quantile(abs(error), 0.9)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve existing fits"
    data = args.rows.read_bytes()
    rows = json.loads(data)
    x = design(rows)
    y = np.array([r["bias_ms"] for r in rows])
    groups = np.array([r["repeat"] for r in rows])
    held = np.empty_like(y)
    for group in sorted(set(groups)):
        train = groups != group
        held[~train] = x[~train] @ solve(x[train], y[train], args.alpha)
    calibration = dict(
        schema=1,
        features=FEATURES,
        coefficients=solve(x, y, args.alpha).tolist(),
        support=dict(requests=[1, 16], context=[15552, 58368], suffix=[832, 3072]),
        bias_bounds_ms=[-500, 500],
        provenance=dict(
            input_sha256=hashlib.sha256(data).hexdigest(),
            rows=len(rows),
            alpha=args.alpha,
            group="repeat across paired workloads",
            elapsed_feature=False,
            selection="shape_raw, alpha=.01 before live tests",
            worker="GLM 5.3 TP8 H2D8/depth2, native D2H1GiB/depth2, "
            "allocator avoidance",
        ),
    )
    report = dict(
        calibration=calibration,
        held_episode_out=metrics(y - held),
        fixed90=metrics(y - 90),
        raw=metrics(y),
        per_workload={
            name: metrics((y - held)[np.array([r["workload"] == name for r in rows])])
            for name in sorted({r["workload"] for r in rows})
        },
        caveat="Grouped cross-validation follows a small exploratory screen. "
        "Independent live seeds and context are the validation gate.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
