# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check exported inference, signed corrections, and training-only scaling."""

import unittest
from unittest.mock import patch

import numpy as np
import study
from sklearn.ensemble import HistGradientBoostingRegressor
from study import FrozenComponent, ScaledMLP, export_component, surface
from threadpoolctl import threadpool_limits


class ExportContract(unittest.TestCase):
    def test_correction_selection_scores_complete_duration(self):
        configs = [{"prediction": 0.0}, {"prediction": -10.0}]

        class Constant:
            def __init__(self, prediction):
                self.value = prediction

            def predict(self, x):
                return np.full(len(x), self.value)

        def fit(config, *args):
            return Constant(config["prediction"]), 0.0

        x = np.zeros((4, 24))
        residual = np.full(4, -10.0)
        rows = [{"workload": "example", "episode": "sample"}] * 4
        log = []
        with (
            patch.object(study, "configs", return_value=configs),
            patch.object(study, "fit", side_effect=fit),
        ):
            chosen, _ = study.select(
                "boost",
                x,
                residual,
                rows,
                x,
                residual,
                rows,
                17,
                log,
                offset=np.full(4, 110.0),
            )
        self.assertEqual(chosen["prediction"], -10.0)
        self.assertEqual(log[0]["validation_macro_mae_ms"], 10.0)
        self.assertEqual(log[1]["validation_macro_mae_ms"], 0.0)

    def test_mlp_export_preserves_signed_correction_on_unseen_inputs(self):
        rng = np.random.default_rng(11)
        x = rng.uniform(0, 10, size=(100, 24))
        y = 20 * (x[:, 0] - 5) + np.sin(x[:, 1])
        weight = np.linspace(0.5, 2, len(x))
        config = dict(layers=[32], input="log1p", solver="lbfgs", alpha=0.01)
        with threadpool_limits(limits=1):
            model = ScaledMLP(config, 17).fit(x, y, weight)
            frozen = FrozenComponent(model.frozen())
            unseen = rng.uniform(0, 20, size=(40, 24))
            expected = model.predict(unseen)
            actual = np.array([frozen.predict_one(row) for row in unseen])
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        self.assertTrue(np.any(actual < 0))
        before = model.x_mean.copy()
        frozen.predict_one(np.full(24, 1e6))
        np.testing.assert_array_equal(before, model.x_mean)

    def test_booster_export_preserves_negative_leaf_outputs(self):
        rng = np.random.default_rng(17)
        x = rng.uniform(0, 10, size=(100, 23)).astype(np.float32)
        y = np.where(x[:, 0] < 5, -80.0, 60.0)
        with threadpool_limits(limits=1):
            model = HistGradientBoostingRegressor(
                max_iter=15, min_samples_leaf=5, early_stopping=False
            ).fit(x, y)
            frozen = FrozenComponent(
                export_component(model, surface.FEATURE_NAMES + surface.FORECAST_NAMES)
            )
            expected = model.predict(x)
            actual = np.array([frozen.predict_one(row) for row in x])
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
        self.assertTrue(np.any(actual < 0))


if __name__ == "__main__":
    unittest.main()
