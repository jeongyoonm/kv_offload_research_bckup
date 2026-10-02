# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online compute duration estimates for experimental late sealing."""

from collections import deque

import numpy as np


class ComputeEstimator:
    """Fit positive costs for tokens, attention work, and request count."""

    def __init__(self):
        self.samples = deque(maxlen=128)
        self.coefficients = None
        self.errors = deque(maxlen=32)
        self.preparation_ns = deque(maxlen=64)

    def observe_preparation(self, records):
        if not records or any(
            not r
            or len(r) != 9
            or r[0] != records[0][0]
            or r[7] != 1
            or not 0 < r[1] <= r[2]
            for r in records
        ):
            return
        self.preparation_ns.append(max(r[2] - r[1] for r in records))

    def preparation_reserve(self, extra_ns):
        observed = (
            int(np.quantile(self.preparation_ns, 0.9)) if self.preparation_ns else 0
        )
        return observed + extra_ns

    @staticmethod
    def features(data):
        shapes = data.get("request_shapes", [])
        if not shapes or any(n <= 0 or context < 0 for n, context in shapes):
            return None
        return np.array(
            [
                1.0,
                sum(n for n, _ in shapes) / 8192,
                sum(n * (context + n / 2) for n, context in shapes) / 100_000_000,
                len(shapes) / 8,
            ]
        )

    def predict(self, data):
        features = self.features(data)
        if self.coefficients is None or features is None:
            return None
        duration = float(features @ self.coefficients)
        # Online residuals compensate systematic underestimation without a floor.
        correction = (
            max(1.0, float(np.quantile(self.errors, 0.75))) if self.errors else 1.0
        )
        return max(1_000_000, int(duration * correction * 1_000_000))

    def observe(self, data, execution):
        features = self.features(data)
        if (
            features is None
            or not execution
            or any(
                not r or r[0] != data["step"] or r[5] != 1 or not r[6] or not r[8]
                for r in execution
            )
        ):
            return
        duration_ms = max(r[6] for r in execution) / 1_000_000
        estimate = self.predict(data)
        if estimate:
            self.errors.append(duration_ms / (estimate / 1_000_000))
        self.samples.append((features, duration_ms))
        if len(self.samples) < 8:
            return
        x = np.vstack([sample[0] for sample in self.samples])
        y = np.array([sample[1] for sample in self.samples])
        coefficients = (
            np.zeros(4) if self.coefficients is None else self.coefficients.copy()
        )
        # Coordinate descent solves nonnegative least squares on four features.
        for _ in range(40):
            for column in range(4):
                residual = y - x @ coefficients + x[:, column] * coefficients[column]
                coefficients[column] = max(
                    0.0,
                    float(x[:, column] @ residual)
                    / max(float(x[:, column] @ x[:, column]), 1e-12),
                )
        self.coefficients = coefficients
