# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, JSON-persisted latency trees with NumPy-only serving inference."""

import copy
import ctypes
import functools
import math
import os

import numpy as np


@functools.lru_cache(maxsize=1)
def native_walk(path):
    library = ctypes.CDLL(path)
    library.predictor_forest_abi.restype = ctypes.c_int
    if library.predictor_forest_abi() != 1:
        raise ValueError("incompatible native predictor forest ABI")
    walk = library.predictor_forest_walk
    walk.argtypes = [ctypes.c_size_t] * 3 + [ctypes.c_void_p] * 6
    walk.restype = ctypes.c_double
    return walk


FEATURE_NAMES = [
    "suffix_tokens",
    "requests",
    "attention_pairs",
    "context_sum",
    "suffix_min",
    "suffix_max",
    "context_min",
    "context_max",
    "padded128",
    "outstanding_loads_at_dispatch",
    "outstanding_load_tokens_at_dispatch",
]

FORECAST_NAMES = [
    "current_load_count",
    "current_load_tokens",
    "load_min_age_ms",
    "load_max_age_ms",
    "load_weighted_age_ms",
    "old_remaining_count",
    "old_remaining_tokens",
    "new_load_count",
    "new_load_tokens",
    "completed_old_tokens",
    "elapsed_execution_ms",
    "queue_to_gpu_start_ms",
]
SKEW_NAMES = ["rank_start_skew_ms"]
OFFSET_NAMES = [f"rank_{rank}_start_offset_ms" for rank in range(8)]
BASE_PREDICTION_NAMES = ["cached_base_prediction_ms"]
SUBMISSION_NAMES = [
    "cpu_submitted_ranks",
    "cpu_submit_min_offset_ms",
    "cpu_submit_max_offset_ms",
    "cpu_submit_median_offset_ms",
    "cpu_submit_offset_std_ms",
    "cpu_last_submit_age_ms",
]
FEATURE_SCHEMAS = [
    FEATURE_NAMES[:9],
    FEATURE_NAMES,
    FEATURE_NAMES + SKEW_NAMES,
    FEATURE_NAMES + FORECAST_NAMES,
    FEATURE_NAMES + FORECAST_NAMES + SKEW_NAMES,
    FEATURE_NAMES + FORECAST_NAMES + OFFSET_NAMES,
    FEATURE_NAMES + FORECAST_NAMES + BASE_PREDICTION_NAMES,
    FEATURE_NAMES + FORECAST_NAMES + SUBMISSION_NAMES,
    FEATURE_NAMES + FORECAST_NAMES + SUBMISSION_NAMES + BASE_PREDICTION_NAMES,
]


def batch_features(data):
    shapes = data.get("request_shapes") or []
    if not shapes or any(n <= 1 or c <= 0 for n, c in shapes):
        return None
    cached = data.get("cpu_cached_prefix_tokens")
    if cached is not None and (
        not isinstance(cached, list)
        or len(cached) != len(shapes)
        or any(type(n) is not int or n <= 0 for n in cached)
    ):
        return None
    suffix, context = zip(*shapes)
    loads = data.get("load_state")
    if (
        not isinstance(loads, (list, tuple))
        or len(loads) != 2
        or any(type(v) is not int or v < 0 for v in loads)
    ):
        return None
    return np.array(
        [
            sum(suffix),
            len(shapes),
            sum(n * (c + n / 2) for n, c in shapes),
            sum(context),
            min(suffix),
            max(suffix),
            min(context),
            max(context),
            sum((n + 127) // 128 * 128 for n in suffix),
            *loads,
        ],
        dtype=float,
    )


class LatencySurface:
    """Walk all regression trees together without importing a training library."""

    feature_names: list[str]
    target_schema: str

    def __init__(self, state):
        self.base = None
        if isinstance(state, dict) and state.get("schema") == 3:
            keys = {
                "schema",
                "base",
                "residual",
                "residual_shift_ms",
                "target_schema",
                "provenance",
            }
            shift = state.get("residual_shift_ms")
            if not isinstance(shift, (int, float)):
                raise ValueError("invalid residual shift")
            try:
                valid_shift = (
                    type(shift) in (int, float)
                    and math.isfinite(shift)
                    and 0 < shift <= 10_000
                )
            except OverflowError:
                valid_shift = False
            if (
                set(state) != keys
                or type(state["schema"]) is not int
                or not valid_shift
                or not all(
                    isinstance(state[key], dict) and state[key].get("schema") in (1, 2)
                    for key in ("base", "residual")
                )
            ):
                raise ValueError("incompatible composed latency surface")
            self.base = LatencySurface(state["base"])
            self.residual = LatencySurface(state["residual"])
            if (
                self.base.feature_names != FEATURE_NAMES
                or self.residual.feature_names
                not in (
                    FEATURE_NAMES + FORECAST_NAMES + BASE_PREDICTION_NAMES,
                    FEATURE_NAMES
                    + FORECAST_NAMES
                    + SUBMISSION_NAMES
                    + BASE_PREDICTION_NAMES,
                )
                or self.base.target_schema != self.residual.target_schema
                or state["target_schema"] != self.base.target_schema
                or sum(
                    len(t)
                    for child in ("base", "residual")
                    for t in state[child]["trees"]
                )
                > 1_000_000
            ):
                raise ValueError("incompatible composed latency features")
            self.requires_forecast = True
            self.requires_load_progress = True
            self.target_schema = state["target_schema"]
            self.residual_shift_ms = shift
            self._state = copy.deepcopy(state)
            return
        keys = {"schema", "feature_names", "trees", "provenance"}
        if isinstance(state, dict) and state.get("schema") == 2:
            keys.add("target_schema")
        if (
            not isinstance(state, dict)
            or set(state) != keys
            or type(state["schema"]) is not int
            or state["schema"] not in (1, 2)
            or state["feature_names"] not in FEATURE_SCHEMAS
            or (
                state["schema"] == 2
                and state["target_schema"]
                not in ("cuda_span_max.v1", "critical_end_after_all_rank_start.v1")
            )
        ):
            raise ValueError("incompatible latency surface")
        trees = state["trees"]
        if (
            not isinstance(trees, list)
            or not 1 <= len(trees) <= 512
            or any(not isinstance(t, list) for t in trees)
            or sum(len(t) for t in trees) > 1_000_000
        ):
            raise ValueError("latency surface exceeds capacity")
        self.feature_count = len(state["feature_names"])
        self.feature_names = state["feature_names"]
        self.requires_forecast = self.feature_count > 11
        self.requires_load_progress = "current_load_count" in state["feature_names"]
        self.target_schema = state.get("target_schema", "cuda_span_max.v1")
        width = max(len(t) for t in trees)
        if width > 32768 or len(trees) * width > 2_000_000 or any(not t for t in trees):
            raise ValueError("invalid latency tree size")
        self.left = np.zeros((len(trees), width), dtype=np.int32)
        self.right = np.zeros_like(self.left)
        self.feature = np.zeros_like(self.left)
        self.threshold = np.zeros_like(self.left, dtype=float)
        self.value = np.zeros_like(self.threshold)
        for tree_index, tree in enumerate(trees):
            depths = {0: 0}
            for node_index, row in enumerate(tree):
                if not isinstance(row, list) or len(row) != 5:
                    raise ValueError("invalid latency tree row")
                left, right, feature, threshold, value = row
                try:
                    finite = all(
                        type(v) in (int, float) and math.isfinite(v)
                        for v in (threshold, value)
                    )
                except OverflowError:
                    finite = False
                if (
                    any(type(v) is not int for v in (left, right, feature))
                    or not finite
                ):
                    raise ValueError("invalid latency tree numeric value")
                if node_index not in depths or depths[node_index] > 32:
                    raise ValueError("invalid latency tree topology")
                leaf = left == right == -1
                if leaf:
                    if feature != -2 or value <= 0:
                        raise ValueError("invalid latency tree leaf")
                else:
                    if not (
                        node_index < left < len(tree)
                        and node_index < right < len(tree)
                        and 0 <= feature < self.feature_count
                    ):
                        raise ValueError("invalid latency tree branch")
                    for child in (left, right):
                        if child in depths:
                            raise ValueError("latency tree has shared children")
                        depths[child] = depths[node_index] + 1
                self.left[tree_index, node_index] = left
                self.right[tree_index, node_index] = right
                self.feature[tree_index, node_index] = feature
                self.threshold[tree_index, node_index] = threshold
                self.value[tree_index, node_index] = value
        self._state = copy.deepcopy(state)
        self.tree_indices = np.arange(len(trees))
        self.native = None
        native_path = os.environ.get("VLLM_DEFERRAL_FAST_FOREST")
        if native_path:
            self.native = functools.partial(
                native_walk(native_path),
                len(trees),
                width,
                self.feature_count,
                self.left.ctypes.data,
                self.right.ctypes.data,
                self.feature.ctypes.data,
                self.threshold.ctypes.data,
                self.value.ctypes.data,
            )

    def state_dict(self):
        return copy.deepcopy(self._state)

    def input_features(self, data):
        if self.base is not None:
            base_ns = self.base.predict(data)
            if base_ns is None:
                return None
            return self.residual.input_features(
                dict(data, cached_base_prediction_ms=base_ns / 1e6)
            )
        features = batch_features(data)
        if features is None:
            return None
        if self.requires_forecast:
            records = data.get("execution_snapshot") or []
            now = data.get("forecast_ns", 0)
            if len(records) != 8 or any(
                not r
                or len(r) != 9
                or r[0] != data.get("step")
                or r[5] != 1
                or not 0 < r[2] <= r[3] <= now
                or r[3] - r[2] > 2_000_000
                for r in records
            ):
                return None
            starts = np.array([r[3] for r in records], dtype=np.int64)
            offsets = (starts - min(starts)) / 1e6
            extra = []
            if self.requires_load_progress:
                loads = data.get("forecast_load_features")
                if not isinstance(loads, list) or len(loads) != 10:
                    return None
                extra.extend(loads)
                extra += [
                    (now - max(starts)) / 1e6,
                    (max(starts) - data["dispatch_ns"]) / 1e6,
                ]
            if "cpu_submitted_ranks" in self.feature_names:
                primary = data.get("primary_snapshot") or []
                if len(primary) != 8 or any(
                    not p
                    or len(p) != 9
                    or p[0] != data.get("step")
                    or p[7] != 1
                    or not 0 < p[3] <= p[6] <= now
                    for p in primary
                ):
                    return None
                submitted = np.array([p[3] for p in primary], dtype=np.int64)
                offsets_ms = (submitted - max(starts)) / 1e6
                extra.extend(
                    [
                        len(submitted),
                        min(offsets_ms),
                        max(offsets_ms),
                        np.median(offsets_ms),
                        np.std(offsets_ms),
                        (now - max(submitted)) / 1e6,
                    ]
                )
            if self.feature_names[-1:] == BASE_PREDICTION_NAMES:
                value = data.get("cached_base_prediction_ms")
                if type(value) not in (int, float):
                    return None
                extra.append(value)
            elif self.feature_count in (12, 24):
                extra.append(max(offsets))
            elif self.feature_count == 31:
                extra.extend(offsets)
            features = np.concatenate((features, extra))
        if len(features) < self.feature_count or not np.all(np.isfinite(features)):
            return None
        return features[: self.feature_count]

    def predict(self, data):
        if self.base is not None:
            base_ns = self.base.predict(data)
            if base_ns is None:
                return None
            correction = self.residual.predict(
                dict(data, cached_base_prediction_ms=base_ns / 1e6)
            )
            if correction is None:
                return None
            return max(
                1_000_000, int(base_ns + correction - self.residual_shift_ms * 1e6)
            )
        features = self.input_features(data)
        if features is None:
            return None
        # Training libraries compare float32 inputs with float64 thresholds.
        features = features.astype(np.float32)
        if data.get("native_predictor", False) is True:
            if self.native is None:
                raise RuntimeError("native predictor library was not configured")
            value = self.native(features.ctypes.data)
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError("native forest walk exceeded validated bounds")
            return max(1_000_000, int(value * 1e6))
        nodes = np.zeros(len(self.tree_indices), dtype=np.int32)
        for _ in range(33):
            left = self.left[self.tree_indices, nodes]
            active = left != -1
            if not np.any(active):
                return max(
                    1_000_000, int(np.mean(self.value[self.tree_indices, nodes]) * 1e6)
                )
            tree = self.tree_indices[active]
            current = nodes[active]
            go_right = (
                features[self.feature[tree, current]] > self.threshold[tree, current]
            )
            nodes[active] = np.where(go_right, self.right[tree, current], left[active])
        raise RuntimeError("latency surface walk exceeded validated depth")
