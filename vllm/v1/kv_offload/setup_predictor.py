# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal features for the remaining setup of a proposed text-prefill batch."""

import ctypes
import functools
import os

import numpy as np

FEATURE_NAMES = [
    "B_requests",
    "B_tokens",
    "B_context_sum",
    "B_suffix_max",
    "B_context_max",
    "B_blocks64",
    "B_padded128",
    "published_candidates",
    "A_requests",
    "A_tokens",
    "A_suffix_max",
    "A_context_max",
    "pending_loads",
    "pending_load_tokens",
    "load_min_age_ms",
    "load_max_age_ms",
    "load_weighted_age_ms",
    "old_loads",
    "old_load_tokens",
    "new_loads",
    "new_load_tokens",
    "completed_old_tokens",
    "cpu_submitted_fraction",
]


def setup_features(shapes, candidates, a_features, primary=None):
    if shapes is None or a_features is None or len(a_features) < 23:
        return None
    if any(q <= 0 or c < 0 for q, c in shapes) or candidates < 0:
        return None
    values = [
        len(shapes),
        sum(q for q, _ in shapes),
        sum(c for _, c in shapes),
        max((q for q, _ in shapes), default=0),
        max((c for _, c in shapes), default=0),
        sum((q + c + 63) // 64 for q, c in shapes),
        sum((q + 127) // 128 * 128 for q, _ in shapes),
        candidates,
        a_features[1],
        a_features[0],
        a_features[5],
        a_features[7],
        *a_features[11:21],
        sum(bool(r and r[3]) for r in primary) / len(primary) if primary else 0,
    ]
    result = np.asarray(values, dtype=np.float32)
    return result if np.isfinite(result).all() and (result >= 0).all() else None


class SetupPredictor:
    """Frozen setup trees with a guarded, optional native evaluation path."""

    def __init__(self, state):
        if state["schema"] != 1 or state["feature_names"] != FEATURE_NAMES:
            raise ValueError("incompatible setup predictor")
        self.model_id = state["model_id"]
        self.shift = float(state["shift_ms"])
        trees = state["trees"]
        if not 1 <= len(trees) <= 256 or not np.isfinite(self.shift):
            raise ValueError("invalid setup forest size or shift")
        width = max(map(len, trees))
        if not 1 <= width <= 8192:
            raise ValueError("invalid setup tree capacity")
        self.left = np.full((len(trees), width), -1, np.int32)
        self.right = self.left.copy()
        self.feature = np.full_like(self.left, -2)
        self.threshold = np.zeros(self.left.shape)
        self.value = self.threshold.copy()
        for t, tree in enumerate(trees):
            depths = {0: 0}
            for n, (left, right, feature, threshold, value) in enumerate(tree):
                if n not in depths or depths[n] > 32:
                    raise ValueError("invalid setup tree topology")
                if not np.isfinite([threshold, value]).all():
                    raise ValueError("nonfinite setup tree")
                if left == right == -1:
                    if feature != -2:
                        raise ValueError("invalid setup leaf")
                else:
                    if not (n < left < len(tree) and n < right < len(tree)):
                        raise ValueError("invalid setup children")
                    if not 0 <= feature < len(FEATURE_NAMES):
                        raise ValueError("invalid setup feature index")
                    for child in (left, right):
                        if child in depths:
                            raise ValueError("shared setup child")
                        depths[child] = depths[n] + 1
                self.left[t, n], self.right[t, n] = left, right
                self.feature[t, n] = feature
                self.threshold[t, n], self.value[t, n] = threshold, value
        guard = state["support"]
        self.indices = guard["indices"]
        self.minimum = np.asarray(guard["minimum"], np.float32)
        self.maximum = np.asarray(guard["maximum"], np.float32)
        if (
            any(
                type(i) is not int or not 0 <= i < len(FEATURE_NAMES)
                for i in self.indices
            )
            or self.minimum.shape != (len(self.indices),)
            or self.maximum.shape != self.minimum.shape
            or not np.isfinite([self.minimum, self.maximum]).all()
            or (self.minimum > self.maximum).any()
        ):
            raise ValueError("invalid setup support")
        self.native = None
        path = os.environ.get("VLLM_DEFERRAL_FAST_FOREST")
        if path:
            library = ctypes.CDLL(path)
            library.predictor_forest_abi.restype = ctypes.c_int
            if library.predictor_forest_abi() != 1:
                raise ValueError("incompatible setup forest ABI")
            walk = library.predictor_forest_walk
            walk.argtypes = [ctypes.c_size_t] * 3 + [ctypes.c_void_p] * 6
            walk.restype = ctypes.c_double
            self.native = functools.partial(
                walk,
                len(trees),
                width,
                len(FEATURE_NAMES),
                self.left.ctypes.data,
                self.right.ctypes.data,
                self.feature.ctypes.data,
                self.threshold.ctypes.data,
                self.value.ctypes.data,
            )
        self.tree_indices = np.arange(len(trees))

    def predict_ms(self, x, native=True):
        if x is None:
            return None
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
            return None
        if (x < 0).any():
            return None
        guarded = x[self.indices]
        if ((guarded < self.minimum) | (guarded > self.maximum)).any():
            return None
        if native and self.native is not None:
            value = self.native(x.ctypes.data)
        else:
            nodes = np.zeros(len(self.left), dtype=int)
            for _ in range(33):
                branch = self.left[self.tree_indices, nodes] >= 0
                if not branch.any():
                    break
                idx = self.tree_indices[branch]
                n = nodes[branch]
                choose_left = x[self.feature[idx, n]] <= self.threshold[idx, n]
                nodes[branch] = np.where(
                    choose_left, self.left[idx, n], self.right[idx, n]
                )
            value = self.value[self.tree_indices, nodes].mean()
        return max(0.0, float(value) - self.shift)
