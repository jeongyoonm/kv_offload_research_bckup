# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze a geometry calibration after holding complete paired seed groups out."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.shape_calibration_20261010.fit import metrics
from vllm.v1.kv_offload.seal_calibration import shape_query

BASE = [
    "intercept",
    "requests_per_16",
    "suffix_sum_per_61440",
    "mean_context_per_60000",
]
RECIPES = {
    "tokens_direct": BASE,
    "attention_direct": BASE + ["max_suffix_per_4096", "prefix_index_pairs_per_1e9"],
    "attention_residual": BASE
    + [
        "max_suffix_per_4096",
        "prefix_index_pairs_per_1e9",
        "raw_duration_per_1000_ms",
    ],
}


def solve(x, y, alpha):
    regularizer = np.eye(x.shape[1]) * alpha
    regularizer[0, 0] = 0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    rows, sources = [], []
    for path in args.rows:
        data = path.read_bytes()
        payload = gzip.decompress(data) if path.suffix == ".gz" else data
        selected = [r for r in json.loads(payload) if r.get("adjacent_compute", True)]
        sources.append(
            dict(
                path=str(path),
                sha256=hashlib.sha256(data).hexdigest(),
                rows=len(selected),
            )
        )
        rows.extend(
            dict(r, group=f"{path.parent.name}:{r['repeat']}") for r in selected
        )
    raw = np.array([r["raw_ms"] for r in rows])
    truth = np.array([r["true_ms"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    workloads = sorted({r["workload"] for r in rows})
    masks = {w: np.array([r["workload"] == w for r in rows]) for w in workloads}
    candidates = []
    for recipe, features in RECIPES.items():
        x = np.array(
            [shape_query(int(r["raw_ms"] * 1e6), r["features"], features) for r in rows]
        )
        y = raw - truth if "residual" in recipe else truth
        for alpha in (0.001, 0.01, 0.1, 1.0):
            held = np.empty_like(y)
            for group in sorted(set(groups)):
                train = groups != group
                held[~train] = x[~train] @ solve(x[train], y[train], alpha)
            error = (raw - held if "residual" in recipe else held) - truth
            by_workload = {w: metrics(error[mask]) for w, mask in masks.items()}
            geometry_held = np.empty_like(y)
            for mask in masks.values():
                geometry_held[mask] = x[mask] @ solve(x[~mask], y[~mask], alpha)
            geometry_error = (
                raw - geometry_held if "residual" in recipe else geometry_held
            ) - truth
            geometry_workloads = {
                w: metrics(geometry_error[mask]) for w, mask in masks.items()
            }
            candidates.append(
                dict(
                    recipe=recipe,
                    alpha=alpha,
                    features=features,
                    coefficients=solve(x, y, alpha).tolist(),
                    held=metrics(error),
                    workload_macro_mae=float(
                        np.mean([m["mae"] for m in by_workload.values()])
                    ),
                    workloads=by_workload,
                    leave_geometry_out=geometry_workloads,
                    geometry_macro_mae=float(
                        np.mean([m["mae"] for m in geometry_workloads.values()])
                    ),
                    rank=int(np.linalg.matrix_rank(x)),
                    columns=x.shape[1],
                )
            )
    best = min(candidates, key=lambda r: r["workload_macro_mae"])
    features, coefficients = best["features"], best["coefficients"]
    if "residual" not in best["recipe"]:
        features = ["raw_duration_per_1000_ms"] + features
        coefficients = [1000] + [-c for c in coefficients]
    calibration = dict(
        schema=3,
        features=features,
        coefficients=coefficients,
        support=dict(
            requests=[1, max(r["requests"] for r in rows)],
            context=[
                min(r["features"][6] for r in rows),
                max(r["features"][7] for r in rows),
            ],
            suffix=[
                min(r["features"][4] for r in rows),
                max(r["features"][5] for r in rows),
            ],
        ),
        bias_bounds_ms=[-500, 500],
        geometry_support=dict(
            suffix_per_context=[
                0.95 * min(r["features"][4] / r["features"][7] for r in rows),
                1.05 * max(r["features"][5] / r["features"][6] for r in rows),
            ]
        ),
        provenance=dict(
            sources=sources,
            group="all arms/workloads per allocation repeat",
            elapsed_feature=False,
            selection=best["recipe"],
            alpha=best["alpha"],
            worker="GLM5.3 TP8 H2D8/depth2 nativeD2H1GiB/depth2 allocator avoidance",
        ),
    )
    report = dict(
        calibration=calibration,
        candidates=candidates,
        selected=best,
        fixed90=metrics(raw - 90 - truth),
        rows=len(rows),
        groups=sorted(set(groups)),
        caveat="Grouped development selection, not final held-out serving accuracy. "
        "Leave-geometry-out queries are shadow extrapolation before support/clip "
        "guards; they diagnose generalization but are not live accuracy. "
        "Use independent seeds and new geometry for frozen300 confirmations.",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                recipe=best["recipe"],
                alpha=best["alpha"],
                rows=len(rows),
                held=best["held"],
                workloads=best["workloads"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
