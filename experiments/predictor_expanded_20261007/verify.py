# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard signed absolute-loss exports and episode-purged calibration labels."""

import importlib
import sys
import unittest
from unittest.mock import Mock, patch

import enqueue
import improve
import numpy as np
import sparse
import study
import trend
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


class CalibrationContract(unittest.TestCase):
    def test_launch_envelope_queue_lead_can_hide_host_but_not_gpu_work(self):
        sys.path.insert(
            0, str(study.SOURCE / "experiments/predictor_launch_envelope_20261007")
        )
        envelope = importlib.import_module("envelope")
        state = dict(
            base=dict(
                core_coefficients=[70.0] + [0.0] * 8,
                host_coefficients=[125.0] + [0.0] * 8,
            )
        )
        model = envelope.Frozen(state)
        x = np.zeros(27)
        self.assertEqual(model.predict_one(x), 125)
        x[-1] = 25
        self.assertEqual(model.predict_one(x), 100)
        x[-1] = 100
        self.assertEqual(model.predict_one(x), 70)
        x[-1] = 10000
        self.assertEqual(model.predict_one(x), 70)

    def test_support_guard_uses_fallback_only_outside_training_bounds(self):
        neural = Mock()
        neural.predict_one.return_value = 10.0
        fallback = Mock()
        fallback.predict_one.return_value = 30.0
        state = dict(
            models=[{}, {}],
            weights=[1.0, 0.0],
            support_guard=dict(
                indices=[0], minimum=[2.0], maximum=[10.0], fallback_model_index=1
            ),
        )
        with patch.object(
            study.arch, "FrozenArchitecture", side_effect=[neural, fallback]
        ):
            model = improve.FrozenCandidate(state)
        x = np.zeros(23)
        x[0] = 10
        self.assertEqual(model.predict_one(x), 10)
        fallback.predict_one.assert_not_called()
        x[0] = 11
        self.assertEqual(model.predict_one(x), 30)
        neural.predict_one.assert_called_once()

    def test_sparse_attention_geometry_matches_causal_selected_token_counts(self):
        for q in [1, 64, 2047, 2048, 2049, 4096]:
            for c in [0, 1, 2047, 2048, 60000]:
                got = sparse.extra_features([dict(request_shapes=[(q, c)])])[0]
                np.testing.assert_array_equal(
                    got,
                    [
                        q * c,
                        sum(range(1, q + 1)),
                        sum(min(sparse.TOPK, c + i) for i in range(1, q + 1)),
                    ],
                )

    def test_gpu_enqueue_features_use_only_starts_known_by_forecast(self):
        ranks = [[1, 1000, 2000, 2100 + i, 0, 1, 10000, 0, 0] for i in range(8)]
        row = dict(execution_ranks=ranks, forecast_time_ns=3000)
        before = enqueue.extra_features([row])
        for rank in ranks:
            rank[4:] = [900000000, 0, 800000000, 700000000, 600000000]
        np.testing.assert_array_equal(before, enqueue.extra_features([row]))
        row["forecast_time_ns"] = 2000
        with self.assertRaises(AssertionError):
            enqueue.extra_features([row])

    def test_median_tree_leaves_respect_training_weights_and_frozen_export(self):
        x = np.zeros((100, 11))
        y = np.r_[np.full(98, 10.0), np.full(2, 500.0)]
        config = dict(kind="base_tree", trees=1, leaf=100, depth=1)
        weights = np.ones(len(x))
        try:
            improve.configure(median_leaves=True)
            with patch.object(study.arch, "weights", side_effect=lambda rows: weights):
                model, _ = improve.fit(config, x, y, [{}] * len(x), 73)
                self.assertEqual(float(model.predict(x[:1])[0]), 10)
                weights[-2:] = 100
                model, _ = improve.fit(config, x, y, [{}] * len(x), 73)
                self.assertEqual(float(model.predict(x[:1])[0]), 500)
                frozen = study.arch.FrozenComponent(
                    study.arch.export_component(model, study.arch.surface.FEATURE_NAMES)
                )
                self.assertEqual(frozen.predict_one(x[0]), 500)
        finally:
            improve.configure()

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
