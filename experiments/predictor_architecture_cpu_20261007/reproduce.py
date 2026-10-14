# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explain archived versus refitted recipe differences without serving changes."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import study
from threadpoolctl import threadpool_limits


def main():
    import evaluate

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = joblib.load(args.output / "dataset.joblib")
    batches = [r for r in pool["batches"] if r["_phase"] == "historical"]
    train = [r for r in pool["forecasts"] if r["_phase"] == "historical"]
    test = [r for r in pool["forecasts"] if r["_phase"] != "historical"]
    checkpoint = study.DATA / "boosted_residual/model.json"
    state = json.loads(checkpoint.read_text())["state"]["latency_surface"]
    old = study.surface.LatencySurface(state)
    archive = np.array(
        [
            old.predict(
                dict(
                    row,
                    execution_snapshot=row["execution_ranks"],
                    forecast_ns=row["forecast_time_ns"],
                    forecast_load_features=row["forecast_features"][:10],
                )
            )
            / 1e6
            for row in test
        ]
    )
    tx = study.runtime_features(test)
    x = study.features(batches)
    np.testing.assert_array_equal(x, np.stack([evaluate.features(r) for r in batches]))
    np.testing.assert_array_equal(study.target(batches), evaluate.target(batches))
    with threadpool_limits(limits=1), patch.object(study, "weights", evaluate.weights):
        base, correction, _ = study.full_fit(
            study.FIXED_BASE, study.FIXED_CORRECTION, batches, train, 17
        )
        bp = study.predict(base, tx[:, :11])
        legacy = np.maximum(
            1, bp + study.predict(correction, np.column_stack((tx, bp)))
        )
    delta = float(np.max(np.abs(archive - legacy)))
    assert delta < 3e-6, delta
    with np.load(args.output / "folds/historical_wide/predictions.npz") as array:
        current = array["trees_fixed"]
        np.testing.assert_allclose(array["y"], study.target(test), rtol=0, atol=0)
    recorded = study.metrics(study.target(test), archive, test)
    assert abs(recorded["mae_ms"] - 19.6244341373) < 1e-4
    assert abs(recorded["mape_percent"] - 5.0368601527) < 1e-4
    weights_new, weights_old = study.weights(batches), evaluate.weights(batches)
    report = {
        "archived_checkpoint": str(checkpoint),
        "archived_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "archived_scores": recorded,
        "legacy_refit_max_difference_ms": delta,
        "new_refit_scores": study.metrics(study.target(test), current, test),
        "new_refit_max_difference_ms": float(np.max(np.abs(archive - current))),
        "weight_max_difference": float(np.max(np.abs(weights_new - weights_old))),
        "features_and_targets_identical": True,
        "cause": "numpy.sum versus legacy built-in sum for weight normalization; "
        "roundoff changes tied Extra Trees split selection",
    }
    study.dump(args.output / "reproduction.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
