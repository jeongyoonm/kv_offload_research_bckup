# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate a submission floor using only the frozen development allocations."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.shape_margin_20261010.audit import shape_bin


def query(row, names):
    x = row["features"]
    values = dict(
        intercept=1,
        requests_per_16=x[1] / 16,
        suffix_sum_per_61440=x[0] / 61440,
        mean_context_per_60000=x[3] / x[1] / 60000,
        max_suffix_per_4096=x[5] / 4096,
        prefix_index_pairs_per_1e9=x[23] / 1e9,
        raw_duration_per_1000_ms=row["raw_ms"] / 1000,
    )
    return [values[n] for n in names]


def floors(rows, quantile):
    bins = {}
    for r in rows:
        if r["requests"] <= 4:
            key = shape_bin(r["requests"], r["features"][5])
            bins.setdefault(key, []).append(r["cpu_full_ms"])
    return {k: float(np.quantile(v, quantile)) for k, v in bins.items() if len(v) >= 20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("rows", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    fit = json.loads(args.fit.read_text())
    rows = json.loads(args.rows.read_text())
    assert len(rows) == fit["selected"]["held"]["count"]
    assert all(r["allocation"] in ("ns1", "yc1", "yc2") for r in rows)
    names = fit["calibration"]["features"]
    x = np.array([query(r, names) for r in rows])
    y = np.array([r["raw_ms"] - r["true_ms"] for r in rows])
    groups = np.array([f"{r['allocation']}:{r['repeat']}" for r in rows])
    regularizer = np.eye(len(names)) * fit["selected"]["alpha"]
    regularizer[0, 0] = 0
    held = np.empty(len(rows))
    floor_predictions = {q: np.empty(len(rows)) for q in (0.1, 0.25, 0.5)}
    for group in sorted(set(groups)):
        train = groups != group
        coefficients = np.linalg.solve(
            x[train].T @ x[train] + regularizer, x[train].T @ y[train]
        )
        training = [r for r, keep in zip(rows, train) if keep]
        for i in np.flatnonzero(~train):
            r = rows[i]
            held[i] = r["raw_ms"] - np.clip(
                x[i] @ coefficients, *fit["calibration"]["bias_bounds_ms"]
            )
        for quantile, predictions in floor_predictions.items():
            bins = floors(training, quantile)
            for i in np.flatnonzero(~train):
                r = rows[i]
                h = bins.get(shape_bin(r["requests"], r["features"][5]))
                predictions[i] = max(
                    held[i], h - r["start_offset_ms"] if h is not None else held[i]
                )
    truth = np.array([r["true_ms"] for r in rows])
    workloads = np.array([r["workload"] for r in rows])

    def metrics(predictions):
        e = predictions - truth
        per = {w: float(np.abs(e[workloads == w]).mean()) for w in set(workloads)}
        return dict(
            mae_ms=float(np.abs(e).mean()),
            workload_macro_mae_ms=float(np.mean(list(per.values()))),
            workload_mae_ms=per,
            maximum_absolute_ms=float(np.abs(e).max()),
        )

    candidates = [dict(quantile=q, **metrics(p)) for q, p in floor_predictions.items()]
    selected = min(candidates, key=lambda c: c["workload_macro_mae_ms"])
    calibration = copy.deepcopy(fit["calibration"])
    calibration.update(
        schema=7,
        submission_floor=dict(
            host_full_ms=floors(rows, selected["quantile"]),
            offset_feature_index=26,
            homogeneous_only=True,
        ),
    )
    calibration["provenance"].update(
        submission_rows_sha256=hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        submission_floor_quantile=selected["quantile"],
        submission_floor_minimum_samples=20,
        submission_floor_scope="Small homogeneous batches only; subtract the "
        "observed enqueue-to-GPU-start offset. No elapsed prediction feature.",
    )
    result = dict(
        calibration=calibration,
        baseline=metrics(held),
        selected=selected,
        candidates=candidates,
        caveat="Grouped development CV. No yc3 fitting. Empirical "
        "submission floor is not a guaranteed execution bound; "
        "unsupported and heterogeneous batches retain the correction.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "calibration"}, indent=2))


if __name__ == "__main__":
    main()
