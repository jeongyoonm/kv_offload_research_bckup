# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bootstrap frozen-model accuracy by workload and repeated serving block."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "frozen_score", SOURCE / "experiments/predictor_portable_20261006/score.py"
)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def prediction(path, rows):
    state = json.loads(path.read_text())["state"]
    if score.overlap(state, rows):
        raise ValueError("held-out evaluation overlaps frozen training")
    legacy = dict(state)
    surface = legacy.pop("latency_surface", None)
    legacy["schema"] = 1
    model = score.estimation.ComputeEstimator.from_state_dict(legacy, state["context"])
    if surface:
        model.latency_surface = score.production.LatencySurface(surface)
    model.frozen = True
    return np.array(
        [
            model.predict(
                dict(
                    row,
                    execution_snapshot=row["execution_ranks"],
                    forecast_ns=row["forecast_time_ns"],
                    forecast_load_features=row["forecast_features"][:10],
                )
            )
            / 1e6
            for row in rows
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    rows = [
        row
        for row in rows
        if all(
            0 < r[2] <= r[3] <= row["forecast_time_ns"] and r[3] - r[2] <= 2_000_000
            for r in row["execution_ranks"]
        )
    ]
    if not rows:
        raise ValueError("no valid all-rank initial forecasts")
    target = np.array(
        [
            (
                max(r[3] + r[6] for r in row["execution_ranks"])
                - max(r[3] for r in row["execution_ranks"])
            )
            / 1e6
            for row in rows
        ]
    )
    control = prediction(args.control, rows) - target
    candidate = prediction(args.candidate, rows) - target
    # All policies of the same workload/repetition share one resampling block.
    blocks = np.array(
        [f"{row['workload']}/{Path(row['episode']).name.split('_')[0]}" for row in rows]
    )
    workloads = np.array([row["workload"] for row in rows])
    rng = np.random.default_rng(17)
    by_workload = {}
    distributions = []
    for workload in sorted(set(workloads)):
        unique = sorted(set(blocks[workloads == workload]))
        summaries = np.array(
            [
                [
                    sum(blocks == block),
                    np.abs(control[blocks == block]).sum(),
                    np.abs(candidate[blocks == block]).sum(),
                ]
                for block in unique
            ]
        )
        draws = rng.integers(0, len(unique), size=(10000, len(unique)))
        totals = summaries[draws].sum(axis=1)
        distribution = (totals[:, 2] - totals[:, 1]) / totals[:, 0]
        distributions.append(distribution)
        mask = workloads == workload
        by_workload[workload] = dict(
            blocks=len(unique),
            samples=int(mask.sum()),
            control_mae_ms=float(np.abs(control[mask]).mean()),
            candidate_mae_ms=float(np.abs(candidate[mask]).mean()),
            candidate_minus_control_95pct_ms=np.quantile(
                distribution, [0.025, 0.975]
            ).tolist(),
            control_late_gt5ms=int(sum(control[mask] > 5)),
            candidate_late_gt5ms=int(sum(candidate[mask] > 5)),
            control_early_gt5ms=int(sum(control[mask] < -5)),
            candidate_early_gt5ms=int(sum(candidate[mask] < -5)),
        )
    result = dict(
        control=str(args.control),
        candidate=str(args.candidate),
        samples=len(rows),
        independent_blocks=len(set(blocks)),
        control_mae_ms=float(np.abs(control).mean()),
        candidate_mae_ms=float(np.abs(candidate).mean()),
        workload_macro_difference_95pct_ms=np.quantile(
            np.mean(distributions, axis=0), [0.025, 0.975]
        ).tolist(),
        per_workload=by_workload,
        caveat=(
            "Repeated workloads/policies are clustered together. Few repeats "
            "limit interval precision; intervals do not establish universal "
            "generalization or an irreducible error bound. Positive error means "
            "the predicted physical end occurs too late."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
