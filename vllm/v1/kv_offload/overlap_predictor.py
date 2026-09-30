# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen overlap evaluator; parity with the batch-building offline model."""

import math

from vllm.v1.kv_offload.batch_predictor import CompletionEstimate, evaluate
from vllm.v1.kv_offload.execution_predictor import execution_features, shape_key
from vllm.v1.kv_offload.execution_timing import valid_snapshot
from vllm.v1.kv_offload.scheduling_deferral import Predictor


def key(row):
    return shape_key(row["request_shapes"]) + (
        ":pending" if row["load_state"][0] else ":empty"
    )


def features(row, starts):
    count, tokens = row["load_state"]
    delay = (min(r[2] for r in starts) - row["dispatch_ns"]) / 1e6
    if count < 0 or tokens < 0 or delay < 0:
        raise ValueError("inconsistent observed load/start state")
    return execution_features(row["request_shapes"]) + [count, tokens / 1024, delay]


def raw_completion(models, row, starts):
    x = features(row, starts)
    durations = [evaluate(m, x) for m in models]
    if not all(math.isfinite(v) and v > 0 for v in durations):
        raise ValueError("invalid predicted execution duration")
    return max(r[2] + d * 1e6 for r, d in zip(starts, durations))


class OverlapPredictor:
    def __init__(self, artifact):
        self.data = artifact
        self.fallback = Predictor(artifact["fallback"])
        self.reserve_ns = artifact["reserve_ns"]
        if not math.isfinite(self.reserve_ns) or self.reserve_ns < 0:
            raise ValueError("invalid overlap reserve")
        if len(artifact["models"]) != self.fallback.data["ranks"]:
            raise ValueError("invalid overlap rank count")
        for model in artifact["models"]:
            if any(len(model[k]) != 15 for k in ("mean", "scale", "coefficients")):
                raise ValueError("invalid overlap dimensions")
            values = [model["intercept"]] + sum(
                [model[k] for k in ("mean", "scale", "coefficients")], []
            )
            if not all(math.isfinite(v) for v in values) or min(model["scale"]) <= 0:
                raise ValueError("invalid overlap coefficients")

    def estimate_completion(self, row, now, ranks):
        base = self.fallback.estimate_completion(row, now, ranks)
        if base.completion_ns is None:
            return base
        starts = row.get("execution_snapshot", [])
        if not valid_snapshot(starts, row["step"], now):
            return CompletionEstimate(None, "overlap", "invalid_start")
        try:
            support = self.data["support"].get(key(row))
            if support is None:
                return CompletionEstimate(
                    base.completion_ns, "fallback", "unsupported_overlap_shape"
                )
            x = features(row, starts)[-3:]
            if any(
                v < lo or v > hi
                for v, lo, hi in zip(x, support["feature_min"], support["feature_max"])
            ):
                return CompletionEstimate(
                    base.completion_ns, "fallback", "overlap_extrapolation"
                )
            completion = raw_completion(self.data["models"], row, starts)
            return CompletionEstimate(
                completion + support["correction_ms"] * 1e6,
                "overlap",
                "observed_start_and_load_state",
            )
        except (KeyError, ValueError, TypeError):
            return CompletionEstimate(
                base.completion_ns, "fallback", "invalid_overlap_features"
            )
