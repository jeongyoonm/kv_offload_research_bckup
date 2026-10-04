# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online compute duration estimates for experimental late sealing."""

from collections import deque

import numpy as np


class ComputeEstimator:
    """Fit positive costs for tokens, attention work, and request count."""

    def __init__(self):
        self.frozen = False
        self.samples: deque[tuple[np.ndarray, float]] = deque(maxlen=128)
        self.coefficients = None
        self.errors: deque[float] = deque(maxlen=32)
        self.preparation_ns: deque[int] = deque(maxlen=64)
        self.preparation_shapes: deque[tuple[np.ndarray, int]] = deque(maxlen=128)
        self.recent_preparation: deque[tuple[np.ndarray, int, str]] = deque(maxlen=128)

    def observe_preparation(self, records, data=None):
        if self.frozen:
            return
        if not records or any(
            not r
            or len(r) != 9
            or r[0] != records[0][0]
            or r[7] != 1
            or not 0 < r[1] <= r[2]
            for r in records
        ):
            return
        duration = max(r[2] - r[1] for r in records)
        self.preparation_ns.append(duration)
        shape = self.preparation_shape(data) if data else None
        if shape is not None:
            self.preparation_shapes.append((shape, duration))
            self.recent_preparation.append(
                (shape, duration, self.preparation_regime(data))
            )

    @staticmethod
    def preparation_shape(data):
        shapes = data.get("request_shapes", [])
        if not shapes or any(n <= 0 or context < 0 for n, context in shapes):
            return None
        return np.array(
            [
                sum(n for n, _ in shapes),
                sum(context for _, context in shapes),
                len(shapes),
            ],
            dtype=float,
        )

    def preparation_reserve(self, extra_ns, upcoming=None):
        if upcoming is not None:
            query = self.preparation_shape(upcoming)
            if query is None:
                return extra_ns
            if self.preparation_shapes:
                distances = [
                    float(np.linalg.norm(np.log1p(query) - np.log1p(shape)))
                    for shape, _ in self.preparation_shapes
                ]
                order = np.argsort(distances)[:8]
                near = [i for i in order if distances[i] <= 1.5]
                if near:
                    return extra_ns + int(
                        np.quantile([self.preparation_shapes[i][1] for i in near], 0.9)
                    )
                # Extrapolate per-token preparation from the closest shape.
                index = int(order[0])
                shape, duration = self.preparation_shapes[index]
                return extra_ns + int(duration * query[0] / shape[0])
            return extra_ns
        observed = (
            int(np.quantile(self.preparation_ns, 0.9)) if self.preparation_ns else 0
        )
        return observed + extra_ns

    @staticmethod
    def preparation_regime(data):
        contexts = [context > 0 for _, context in data.get("request_shapes", [])]
        return (
            "cached"
            if contexts and all(contexts)
            else ("cold" if not any(contexts) else "mixed")
        )

    def recent_preparation_reserve(self, extra_ns, upcoming):
        """Use recent nearby shapes from the same cold/cache regime."""
        query = self.preparation_shape(upcoming)
        if query is None:
            return extra_ns
        regime = self.preparation_regime(upcoming)
        candidates = [
            (shape, duration)
            for shape, duration, observed_regime in self.recent_preparation
            if observed_regime == regime
        ][-32:]
        distances = [
            float(np.linalg.norm(np.log1p(query) - np.log1p(shape)))
            for shape, _ in candidates
        ]
        near = [i for i in np.argsort(distances)[:8] if distances[i] <= 0.75]
        if not near:
            return extra_ns
        return extra_ns + int(np.quantile([candidates[i][1] for i in near], 0.75))

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
        if self.frozen:
            return
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
