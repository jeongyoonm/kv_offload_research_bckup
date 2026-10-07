# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard signed absolute-loss exports and episode-purged calibration labels."""

import unittest
from unittest.mock import patch

import improve
import numpy as np
import study
import trend
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


class CalibrationContract(unittest.TestCase):
    def test_exported_continuous_trend_extrapolates_positive_work(self):
        x = np.zeros((100, 11))
        x[:, 0] = np.linspace(512, 32768, len(x))
        x[:, 1] = 1
        y = 5 + 0.015 * x[:, 0]
        with (
            patch.object(study.arch, "features", return_value=x),
            patch.object(study.arch, "target", return_value=y),
            patch.object(study.arch, "weights", return_value=np.ones(len(x))),
        ):
            state = trend.fit_linear([{}] * len(x), 0.1)
        model = improve.FrozenTrend(dict(base=state))
        unseen = x[-1].copy()
        unseen[0] = 65536
        self.assertGreater(model.predict_one(unseen), model.predict_one(x[-1]))
        self.assertLess(abs(model.predict_one(unseen) - (5 + 0.015 * 65536)), 2)

    def test_absolute_loss_export_preserves_negative_corrections(self):
        rng = np.random.default_rng(73)
        x = rng.uniform(0, 10, size=(100, 23)).astype(np.float32)
        y = np.where(x[:, 0] < 5, -80.0, 60.0)
        with threadpool_limits(limits=1):
            model = HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=20,
                min_samples_leaf=5,
                early_stopping=False,
            ).fit(x, y)
            frozen = study.arch.FrozenComponent(
                study.arch.export_component(
                    model,
                    study.arch.surface.FEATURE_NAMES
                    + study.arch.surface.FORECAST_NAMES,
                )
            )
            actual = [frozen.predict_one(v) for v in x]
        np.testing.assert_allclose(actual, model.predict(x), atol=1e-4, rtol=0)
        self.assertTrue(np.any(np.array(actual) < 0))

    def test_full_architecture_calibration_purges_entire_held_episodes(self):
        rows = [dict(episode=str(i), duration=10.0 * i) for i in range(6)]
        batches = rows + [dict(episode=str(i), duration=-1) for i in range(6)]

        def fit_one(name, selected, training_batches, training_rows, seed):
            return dict(
                episodes={r["episode"] for r in training_batches + training_rows},
                mean=np.mean([r["duration"] for r in training_rows]),
            )

        def predict_state(state, held):
            fitted = state["models"][0]
            self.assertFalse(fitted["episodes"] & {r["episode"] for r in held})
            return np.full(len(held), fitted["mean"])

        with (
            patch.object(improve, "fit_one", side_effect=fit_one),
            patch.object(improve, "predict_state", side_effect=predict_state),
        ):
            predicted, audit = improve.oof_architecture(
                "trees_fixed", {}, batches, rows, 73
            )
        self.assertTrue(np.isfinite(predicted).all())
        self.assertEqual(len(audit), 3)
        self.assertTrue(all(a["episode_overlap"] == 0 for a in audit))

    def test_sparse_calibration_group_uses_global_training_offset(self):
        x = np.zeros((60, 23))
        x[-5:, 0] = 50000
        rows = [{}] * len(x)
        residual = np.r_[np.full(55, 7.0), np.full(5, 100.0)]
        with (
            patch.object(study.arch, "target", return_value=residual),
            patch.object(study.arch, "runtime_features", return_value=x),
            patch.object(study.arch, "weights", return_value=np.ones(len(x))),
        ):
            state = improve.calibrate(rows, np.zeros(len(x)))
        self.assertEqual(state["global_shift_ms"], 7)
        self.assertNotIn(improve.calibration_group(x[-1]), state["group_shift_ms"])
        self.assertEqual(state["counts"][improve.calibration_group(x[-1])], 5)


if __name__ == "__main__":
    unittest.main()
