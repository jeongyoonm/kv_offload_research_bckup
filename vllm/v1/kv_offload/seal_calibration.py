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
GEOMETRY_FEATURES = FEATURES + [
    "suffix_sum_per_61440",
    "mean_suffix_per_4096",
    "max_suffix_per_4096",
    "prefix_index_pairs_per_1e9",
]


def shape_bin(requests, suffix):
    size = "small" if requests <= 4 else "medium" if requests <= 8 else "large"
    length = "short" if suffix <= 1024 else "mid" if suffix <= 2048 else "long"
    return f"{size}_{length}"


def shape_query(raw_duration_ns, x, features):
    """Build the same causal shape basis for frozen fitting and inference."""
    values = dict(
        intercept=1,
        mean_context_per_60000=x[3] / x[1] / 60000,
        requests_per_16=x[1] / 16,
        context_sum_per_960000=x[3] / 960000,
        raw_duration_per_1000_ms=raw_duration_ns / 1e9,
        suffix_sum_per_61440=x[0] / 61440,
        mean_suffix_per_4096=x[0] / x[1] / 4096,
        max_suffix_per_4096=x[5] / 4096,
    )
    if "prefix_index_pairs_per_1e9" in features:
        if len(x) < 24 or not np.isfinite(x[23]):
            return None
        values["prefix_index_pairs_per_1e9"] = x[23] / 1e9
    return np.array([values[name] for name in features])


class BatchShapeCalibration:
    def __init__(self, state):
        required = {
            "schema",
            "features",
            "coefficients",
            "support",
            "bias_bounds_ms",
            "provenance",
        }
        schema = state.get("schema") if isinstance(state, dict) else None
        if schema in (2, 3, 4, 5, 6, 7):
            required.add("geometry_support")
        if schema == 7:
            required.add("submission_floor")
        if schema in (4, 6):
            required.add("deadline_margin")
        if schema in (5, 6):
            required.add("productive_load_policy")
        if (
            not isinstance(state, dict)
            or set(state) != required
            or schema not in (1, 2, 3, 4, 5, 6, 7)
            or (schema in (1, 2) and state["features"] != FEATURES)
            or not isinstance(state["features"], list)
            or not state["features"]
            or any(f not in GEOMETRY_FEATURES for f in state["features"])
            or len(set(state["features"])) != len(state["features"])
        ):
            raise ValueError("invalid shape calibration schema")
        self.features = tuple(state["features"])
        self.coefficients = np.asarray(state["coefficients"], dtype=float)
        self.bounds = np.asarray(state["bias_bounds_ms"], dtype=float)
        self.support = state["support"]
        if (
            self.coefficients.shape != (len(self.features),)
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
        self.geometry_bounds = None
        if schema in (2, 3, 4, 5, 6, 7):
            geometry = state["geometry_support"]
            if not isinstance(geometry, dict) or set(geometry) != {
                "suffix_per_context"
            }:
                raise ValueError("invalid shape calibration geometry support")
            self.geometry_bounds = np.asarray(
                geometry["suffix_per_context"], dtype=float
            )
            if (
                self.geometry_bounds.shape != (2,)
                or not np.isfinite(self.geometry_bounds).all()
                or not 0 < self.geometry_bounds[0] <= self.geometry_bounds[1] <= 1
            ):
                raise ValueError("invalid shape calibration geometry support")

        self.submission_floor = None
        if schema == 7:
            floor = state["submission_floor"]
            if (
                not isinstance(floor, dict)
                or set(floor)
                != {"host_full_ms", "offset_feature_index", "homogeneous_only"}
                or floor["offset_feature_index"] != 26
                or floor["homogeneous_only"] is not True
                or not isinstance(floor["host_full_ms"], dict)
                or not floor["host_full_ms"]
                or any(
                    key not in ("small_short", "small_mid", "small_long")
                    or not np.isfinite(value)
                    or not 0 < value <= 5000
                    for key, value in floor["host_full_ms"].items()
                )
            ):
                raise ValueError("invalid calibrated submission floor")
            self.submission_floor = floor
        self.deadline_margin = None
        if schema in (4, 6):
            margin = state["deadline_margin"]
            required_margin = {
                "finish_allowance_ms",
                "setup_allowance_ms",
                "finish_fallback_ms",
                "setup_fallback_ms",
                "extra_ms",
                "bounds_ms",
            }
            if not isinstance(margin, dict) or set(margin) != required_margin:
                raise ValueError("invalid calibrated deadline margin")
            allowed_bins = {
                f"{n}_{q}"
                for n in ("small", "medium", "large")
                for q in ("short", "mid", "long")
            }
            for key in ("finish_allowance_ms", "setup_allowance_ms"):
                values = margin[key]
                if not isinstance(values, dict) or set(values) - allowed_bins:
                    raise ValueError("invalid calibrated deadline margin bins")
                if any(
                    not np.isfinite(v) or not 0 <= v <= 1000 for v in values.values()
                ):
                    raise ValueError("invalid calibrated deadline margin values")
            for key in ("finish_fallback_ms", "setup_fallback_ms", "extra_ms"):
                if not np.isfinite(margin[key]) or not 0 <= margin[key] <= 1000:
                    raise ValueError("invalid calibrated deadline margin values")
            bounds = np.asarray(margin["bounds_ms"], dtype=float)
            if (
                bounds.shape != (2,)
                or not np.isfinite(bounds).all()
                or not 0 <= bounds[0] <= bounds[1] <= 250
            ):
                raise ValueError("invalid calibrated deadline margin bounds")
            self.deadline_margin = margin
        self.productive_load_policy = None
        if schema in (5, 6):
            from vllm.v1.kv_offload.productive_load_policy import ProductiveLoadPolicy

            self.productive_load_policy = ProductiveLoadPolicy(
                state["productive_load_policy"], self.support, self.geometry_bounds
            )

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
        if self.geometry_bounds is not None:
            if x[6] <= 0:
                return fallback_ns, True
            lower, upper = self.geometry_bounds
            if x[4] / x[7] < lower or x[5] / x[6] > upper:
                return fallback_ns, True
        query = shape_query(raw_duration_ns, x, self.features)
        if query is None:
            return fallback_ns, True
        bias = float(query @ self.coefficients)
        if not np.isfinite(bias):
            raise RuntimeError("nonfinite shape calibration result")
        bias = float(np.clip(bias, *self.bounds))
        floor_ns = self.submission_floor_ns(x)
        if floor_ns is not None:
            bias = float(
                np.clip(min(bias, (raw_duration_ns - floor_ns) / 1e6), *self.bounds)
            )
        return int(bias * 1e6), False

    def submission_floor_ns(self, features):
        """Subtract the observed start offset from a frozen submission allowance."""
        if self.submission_floor is None or features is None:
            return None
        x = np.asarray(features, dtype=float)
        if (
            x.ndim != 1
            or len(x) <= 26
            or not np.isfinite(x[[1, 4, 5, 6, 7, 26]]).all()
            or not 1 <= x[1] <= 4
            or x[4] != x[5]
            or x[6] != x[7]
            or x[26] < 0
        ):
            return None
        host_ms = self.submission_floor["host_full_ms"].get(shape_bin(x[1], x[5]))
        if host_ms is None:
            return None
        return max(0, int((host_ms - x[26]) * 1e6))

    def deadline_lead_ns(
        self, raw_duration_ns, features, base_shapes, candidate_shapes, fallback_ns
    ):
        """Freeze a bounded lead using A uncertainty and possible B setup shapes."""
        margin = self.deadline_margin
        if margin is None or self.bias_ns(raw_duration_ns, features, 0)[1]:
            return fallback_ns, True
        if base_shapes is None or candidate_shapes is None:
            return fallback_ns, True
        shapes = np.asarray(base_shapes + candidate_shapes, dtype=float)
        if (
            shapes.ndim != 2
            or shapes.shape[1] != 2
            or not 0 < len(shapes) <= min(32, self.support["requests"][1])
            or not np.isfinite(shapes).all()
        ):
            return fallback_ns, True
        for column, key in ((0, "suffix"), (1, "context")):
            low, high = self.support[key]
            if np.any(shapes[:, column] < low) or np.any(shapes[:, column] > high):
                return fallback_ns, True
        a_bin = shape_bin(features[1], features[5])
        error = margin["finish_allowance_ms"].get(a_bin, margin["finish_fallback_ms"])
        suffix = float(shapes[:, 0].max())
        setup = max(
            margin["setup_allowance_ms"].get(
                shape_bin(n, suffix), margin["setup_fallback_ms"]
            )
            for n in range(max(1, len(base_shapes)), len(shapes) + 1)
        )
        lead_ms = float(
            np.clip(error + setup + margin["extra_ms"], *margin["bounds_ms"])
        )
        return int(lead_ms * 1e6), False
