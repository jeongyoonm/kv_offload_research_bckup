# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, JSON-persisted latency trees with NumPy-only serving inference."""

import copy
import math

import numpy as np

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


def batch_features(data):
    shapes = data.get("request_shapes") or []
    if not shapes or any(n <= 1 or c <= 0 for n, c in shapes):
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

    def __init__(self, state):
        keys = {"schema", "feature_names", "trees", "provenance"}
        if (
            not isinstance(state, dict)
            or set(state) != keys
            or type(state["schema"]) is not int
            or state["schema"] != 1
            or state["feature_names"] not in (FEATURE_NAMES[:9], FEATURE_NAMES)
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

    def state_dict(self):
        return copy.deepcopy(self._state)

    def predict(self, data):
        features = batch_features(data)
        if features is None:
            return None
        # Training libraries compare float32 inputs with float64 thresholds.
        features = features[: self.feature_count].astype(np.float32)
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
