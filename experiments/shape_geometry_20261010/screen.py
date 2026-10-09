# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exploratory shape fits; use independent serving seeds for final selection."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.shape_calibration_20261010.fit import metrics
from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration


def load(path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return [r for r in json.loads(data) if r.get("adjacent_compute", True)]


def basis(rows, kind):
    f = np.array([r["features"] for r in rows])
    raw = np.array([r["raw_ms"] for r in rows])
    columns = [np.ones(len(rows)), f[:, 1] / 16, f[:, 0] / 61440]
    if "mean_context" in kind:
        columns.append(f[:, 3] / f[:, 1] / 60000)
    if "pairs" in kind:
        columns.append(f[:, 23] / 1e9)
    if "residual" in kind:
        columns.append(raw / 1000)
    return np.array(columns).T


def solve(x, y, alpha):
    penalties = np.eye(x.shape[1]) * alpha
    penalties[0, 0] = 0
    return np.linalg.solve(x.T @ x + penalties, x.T @ y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path)
    parser.add_argument("development", type=Path)
    parser.add_argument("frozen_fit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    train, dev = load(args.training), load(args.development)
    truth = np.array([r["true_ms"] for r in dev])
    raw = np.array([r["raw_ms"] for r in dev])
    calibrator = BatchShapeCalibration(
        json.loads(args.frozen_fit.read_text())["calibration"]
    )
    original = np.array(
        [
            raw[i]
            - calibrator.bias_ns(int(raw[i] * 1e6), r["features"], 90_000_000)[0] / 1e6
            for i, r in enumerate(dev)
        ]
    )
    candidates = [("fixed90", None, raw - 90), ("original_shape", None, original)]
    coefficients, geometry_rank = {}, {}
    for kind in (
        "tokens_direct",
        "tokens_mean_context_direct",
        "tokens_pairs_direct",
        "tokens_mean_context_pairs_direct",
        "tokens_pairs_residual",
        "tokens_mean_context_pairs_residual",
    ):
        x, query = basis(train, kind), basis(dev, kind)
        condition = float(np.linalg.cond(x))
        geometry_rank[kind] = dict(
            columns=x.shape[1],
            rank=int(np.linalg.matrix_rank(x)),
            condition=condition if np.isfinite(condition) else None,
        )
        y = np.array(
            [r["bias_ms"] if "residual" in kind else r["true_ms"] for r in train]
        )
        for alpha in (0.01, 0.1, 1.0):
            coef = solve(x, y, alpha)
            predicted = query @ coef
            if "residual" in kind:
                predicted = raw - predicted
            key = f"{kind}:{alpha}"
            coefficients[key] = coef.tolist()
            candidates.append((kind, alpha, np.maximum(1, predicted)))
    output = []
    for kind, alpha, predicted in candidates:
        error = predicted - truth
        output.append(
            dict(
                kind=kind,
                alpha=alpha,
                all=metrics(error),
                workloads={
                    w: metrics(error[np.array([r["workload"] == w for r in dev])])
                    for w in sorted({r["workload"] for r in dev})
                },
            )
        )
    args.output.write_text(
        json.dumps(
            dict(
                training_rows=len(train),
                development_rows=len(dev),
                candidates=output,
                coefficients=coefficients,
                geometry_rank=geometry_rank,
                caveat="Exploratory selection using prior live labels as development. "
                "Context/suffix remain correlated. No crossed labels or final serving "
                "seeds are used. New candidate queries are unconstrained exploratory "
                "fits; support/clip guards require later serving validation. These "
                "fits are not loaded into the running server.",
            ),
            indent=2,
        )
        + "\n"
    )
    for row in sorted(output, key=lambda r: r["all"]["mae"])[:8]:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
