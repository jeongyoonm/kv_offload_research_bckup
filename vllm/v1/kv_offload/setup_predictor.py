# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal features for the remaining setup of a proposed text-prefill batch."""

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
