# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen batch-shape correction for an experimental finish-time predictor."""

import numpy as np

FEATURES = [
    "intercept",
    "mean_context_per_60000",
    "requests_per_16",
    "context_sum_per_960000",
    "raw_duration_per_1000_ms",
]


class BatchShapeCalibration:
    def __init__(self, state):
        if (
            not isinstance(state, dict)
            or set(state)
            != {
                "schema",
                "features",
                "coefficients",
                "support",
                "bias_bounds_ms",
                "provenance",
            }
            or state["schema"] != 1
            or state["features"] != FEATURES
        ):
            raise ValueError("invalid shape calibration schema")
        self.coefficients = np.asarray(state["coefficients"], dtype=float)
        self.bounds = np.asarray(state["bias_bounds_ms"], dtype=float)
        self.support = state["support"]
        if (
            self.coefficients.shape != (5,)
            or not np.isfinite(self.coefficients).all()
            or self.bounds.shape != (2,)
            or not np.isfinite(self.bounds).all()
            or not -1000 <= self.bounds[0] <= self.bounds[1] <= 1000
            or not isinstance(self.support, dict)
            or set(self.support) != {"requests", "context", "suffix"}
        ):
            raise ValueError("invalid shape calibration parameters")
        for name, values in self.support.items():
            limits = np.asarray(values, dtype=float)
            if (
                limits.shape != (2,)
                or not np.isfinite(limits).all()
                or not 0 <= limits[0] <= limits[1]
                or (name == "requests" and not 1 <= limits[0] <= limits[1] <= 32)
            ):
                raise ValueError("invalid shape calibration support")

    def bias_ns(self, raw_duration_ns, features, fallback_ns):
        if features is None:
            return fallback_ns, True
        x = np.asarray(features, dtype=float)
        if x.ndim != 1 or len(x) < 11 or not np.isfinite(x[:11]).all():
            return fallback_ns, True
        for name, low, high in (
            ("requests", x[1], x[1]),
            ("context", x[6], x[7]),
            ("suffix", x[4], x[5]),
        ):
            minimum, maximum = self.support[name]
            if low < minimum or high > maximum:
                return fallback_ns, True
        query = np.array(
            [1, x[3] / x[1] / 60000, x[1] / 16, x[3] / 960000, raw_duration_ns / 1e9]
        )
        bias = float(query @ self.coefficients)
        if not np.isfinite(bias):
            raise RuntimeError("nonfinite shape calibration result")
        return int(np.clip(bias, *self.bounds) * 1e6), False
