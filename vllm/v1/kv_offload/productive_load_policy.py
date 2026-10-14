# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in frozen shape model for an earlier productive load-only seal."""

import numpy as np

from vllm.v1.kv_offload.seal_calibration import shape_query

DIRECT_FEATURES = [
    "intercept",
    "requests_per_16",
    "suffix_sum_per_61440",
    "mean_context_per_60000",
    "max_suffix_per_4096",
    "prefix_index_pairs_per_1e9",
]


class ProductiveLoadPolicy:
    def __init__(self, state, support, geometry_bounds):
        if (
            not isinstance(state, dict)
            or set(state)
            != {
                "schema",
                "features",
                "coefficients",
                "work_coefficients",
                "minimum_efficiency",
                "minimum_ready_candidates",
                "provenance",
            }
            or state["schema"] != 1
            or state["features"] != DIRECT_FEATURES
        ):
            raise ValueError("invalid productive load policy schema")
        self.coefficients = np.asarray(state["coefficients"], dtype=float)
        self.work_coefficients = np.asarray(state["work_coefficients"], dtype=float)
        if (
            self.coefficients.shape != (6,)
            or self.work_coefficients.shape != (2,)
            or not np.isfinite(self.coefficients).all()
            or not np.isfinite(self.work_coefficients).all()
            or np.any(self.work_coefficients <= 0)
        ):
            raise ValueError("invalid productive load policy coefficients")
        self.efficiency = state["minimum_efficiency"]
        self.minimum_ready = state["minimum_ready_candidates"]
        if (
            not 0 < self.efficiency <= 1
            or type(self.minimum_ready) is not int
            or not 1 <= self.minimum_ready <= 16
        ):
            raise ValueError("invalid productive load policy target")
        self.support = support
        self.geometry_bounds = geometry_bounds

    def rate(self, shapes):
        """Return normalized predicted uncached-work rate, within observed support."""
        x = np.asarray(shapes, dtype=float)
        if (
            x.ndim != 2
            or x.shape[1] != 2
            or not 1 <= len(x) <= self.support["requests"][1]
            or not np.isfinite(x).all()
        ):
            return None
        for column, key in ((0, "suffix"), (1, "context")):
            low, high = self.support[key]
            if np.any(x[:, column] < low) or np.any(x[:, column] > high):
                return None
        ratios = x[:, 0] / x[:, 1]
        if np.any(ratios < self.geometry_bounds[0]) or np.any(
            ratios > self.geometry_bounds[1]
        ):
            return None
        f = np.zeros(24)
        f[0], f[1], f[3] = x[:, 0].sum(), len(x), x[:, 1].sum()
        f[5], f[23] = x[:, 0].max(), (x[:, 0] * x[:, 1]).sum()
        duration = float(shape_query(0, f, DIRECT_FEATURES) @ self.coefficients)
        work = float(np.array([f[0] / 61440, f[23] / 1e9]) @ self.work_coefficients)
        if duration <= 0 or work <= 0:
            return None
        return work / max(work, duration)

    def ready(self, base_shapes, ready_shapes, active_shapes):
        """Compare only observed ready work to the already-published cohort."""
        if (
            base_shapes is None
            or ready_shapes is None
            or active_shapes is None
            or len(ready_shapes) < self.minimum_ready
        ):
            return False, None
        current = self.rate(base_shapes + ready_shapes)
        full = self.rate(base_shapes + active_shapes)
        if current is None or full is None:
            return False, None
        efficiency = current / full
        return efficiency >= self.efficiency, efficiency
