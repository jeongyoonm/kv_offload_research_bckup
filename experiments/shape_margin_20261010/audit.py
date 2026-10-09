# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate grouped finish-error uncertainty from observed successor host setup."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def shape_bin(requests, suffix):
    size = "small" if requests <= 4 else "medium" if requests <= 8 else "large"
    length = "short" if suffix <= 1024 else "mid" if suffix <= 2048 else "long"
    return f"{size}_{length}"


def summarize(values):
    x = np.asarray(values)
    return dict(
        count=len(x),
        mean=float(x.mean()),
        mae=float(np.abs(x).mean()),
        p75=float(np.quantile(x, 0.75)),
        p80=float(np.quantile(x, 0.8)),
        p90=float(np.quantile(x, 0.9)),
        p95=float(np.quantile(x, 0.95)),
        maximum=float(x.max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--comparisons", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    fit = json.loads(args.fit.read_text())
    selected = fit["selected"]
    assert selected["recipe"] == "attention_residual"
    rows = []
    for source in fit["calibration"]["provenance"]["sources"]:
        path = Path(source["path"])
        data = path.read_bytes()
        data = gzip.decompress(data) if path.suffix == ".gz" else data
        rows.extend(
            dict(r, group=f"{path.parent.name}:{r['repeat']}")
            for r in json.loads(data)
            if r.get("adjacent_compute", True)
        )
    queries = []
    for row in rows:
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
        queries.append([values[name] for name in selected["features"]])
    queries = np.array(queries)
    target = np.array([r["raw_ms"] - r["true_ms"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    held = np.empty_like(target)
    regularizer = np.eye(queries.shape[1]) * selected["alpha"]
    regularizer[0, 0] = 0
    for group in sorted(set(groups)):
        mask = groups != group
        x, y = queries[mask], target[mask]
        coefficients = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
        held[~mask] = queries[~mask] @ coefficients
    errors = target - held
    assert np.isclose(np.mean(abs(errors)), selected["held"]["mae"])
    finish_bins, setup_bins, setup_rows = {}, {}, []
    for row, error in zip(rows, errors):
        key = shape_bin(row["requests"], row["features"][5])
        finish_bins.setdefault(key, []).append(error)
    for path in args.comparisons:
        comparison = json.loads(path.read_text())
        episodes = {
            (e["workload"], e["label"], e["repeat"]): e for e in comparison["episodes"]
        }
        for gap in comparison["gaps"]:
            episode = episodes[gap["workload"], gap["arm"], gap["repeat"]]
            batch = next(b for b in episode["batch_shapes"] if b["step"] == gap["step"])
            suffix = max(s[0] for s in batch["request_shapes"])
            stages = gap["worker_stages"].values()
            setup = (
                max(s["forward_submission_begin"] - s["worker_entry"] for s in stages)
                / 1e6
            )
            meta = (
                max(s["attention_metadata_ready"] - s["worker_entry"] for s in stages)
                / 1e6
            )
            key = shape_bin(gap["requests"], suffix)
            setup_bins.setdefault(key, []).append(setup)
            setup_rows.append(
                dict(
                    source=str(path),
                    workload=gap["workload"],
                    arm=gap["arm"],
                    repeat=gap["repeat"],
                    step=gap["step"],
                    shape_bin=key,
                    requests=gap["requests"],
                    max_suffix=suffix,
                    entry_to_forward_ms=setup,
                    entry_to_metadata_ms=meta,
                    seal_reason=gap["seal_reason"],
                    gap_ms=gap["gap_ms"],
                )
            )
    finish = {k: summarize(v) for k, v in finish_bins.items()}
    setup = {k: summarize(v) for k, v in setup_bins.items()}
    result = dict(
        finish_errors_by_A_shape=finish,
        host_setup_by_B_shape=setup,
        setup_rows=setup_rows,
        definitions="Finish errors are grouped out-of-fold development predictions. "
        "Positive finish error can postpone the seal. Setup is maximum per-rank "
        "worker-entry to forward submission; it excludes RPC FIFO waiting but "
        "includes concurrent-work interference and connector handling. "
        "A and B shape bins are separate; summing equal-named bins is invalid. "
        "These distributions diagnose margins, not a causal throughput forecast.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(finish=finish, setup=setup), indent=2))


if __name__ == "__main__":
    main()
