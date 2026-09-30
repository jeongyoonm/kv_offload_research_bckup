# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execution duration model anchored exclusively to observed GPU start events."""

import json
import math
from collections import Counter

from vllm.v1.kv_offload.batch_predictor import CompletionEstimate, evaluate
from vllm.v1.kv_offload.execution_timing import valid_snapshot

FEATURE_VERSION = 1


def shape_key(requests):
    """Joint empirical support; marginal feature ranges cannot authorize shapes."""
    return json.dumps(sorted(Counter(tuple(r) for r in requests).items()))


def execution_features(requests):
    if not requests or any(
        len(r) != 2 or any(type(v) is not int for v in r) or r[0] <= 0 or r[1] < 0
        for r in requests
    ):
        raise ValueError("invalid execution shape")
    n = len(requests)
    new = [r[0] for r in requests]
    context = [r[1] for r in requests]
    prefill = sum(v > 1 for v in new)
    return [
        n,
        n * n,
        sum(new) / 1024,
        sum(v * v for v in new) / 1024**2,
        sum(q * c + q * (q + 1) / 2 for q, c in requests) / 1e6,
        sum(context) / 1024,
        min(context) / 1024,
        max(context) / 1024,
        min(new) / 1024,
        max(new) / 1024,
        prefill / n,
        float(0 < prefill < n),
    ]


class ExecutionPredictor:
    def __init__(self, data):
        self.data = data
        self.reserve_ns = data["reserve_ns"]
        if (
            data.get("feature_version") != FEATURE_VERSION
            or not isinstance(data.get("execution_config"), str)
            or not data["execution_config"]
            or type(data.get("ranks")) is not int
            or data["ranks"] <= 0
            or not math.isfinite(self.reserve_ns)
            or self.reserve_ns < 0
            or len(data["models"]) != data["ranks"]
        ):
            raise ValueError("invalid execution model configuration")
        fallback = data.get("fallback_models", [])
        if fallback and len(fallback) != data["ranks"]:
            raise ValueError("invalid fallback rank count")
        for model in data["models"] + fallback:
            if any(len(model[k]) != 12 for k in ("coefficients", "mean", "scale")):
                raise ValueError("invalid execution model dimensions")
            values = [model["intercept"]]
            values += model["coefficients"] + model["mean"] + model["scale"]
            if not all(math.isfinite(v) for v in values) or min(model["scale"]) <= 0:
                raise ValueError("invalid execution coefficients")
        for support in data["support"].values():
            family = support.get("model", "primary")
            if family not in ("primary", "fallback") or (
                family == "fallback" and not fallback
            ):
                raise ValueError("invalid execution support model")
            if (
                support["batches"] < 20
                or support["cells"] < 3
                or not math.isfinite(support["correction_ms"])
            ):
                raise ValueError("invalid execution support")

    def estimate_completion(self, data, now, ranks):
        name = "gpu_start"
        if data.get("execution_config") != self.data["execution_config"]:
            return CompletionEstimate(None, name, "execution_config_mismatch")
        starts = data.get("execution_snapshot", [])
        if len(starts) != self.data["ranks"] or not valid_snapshot(
            starts, data["step"], now
        ):
            return CompletionEstimate(None, name, "gpu_start_unavailable")
        try:
            requests = data["request_shapes"]
            x = execution_features(requests)
            if (
                len(requests) != data["shape"][1]
                or sum(r[0] for r in requests) != data["shape"][2]
            ):
                raise ValueError("inconsistent shape")
            support = self.data["support"].get(shape_key(requests))
            if support is None:
                return CompletionEstimate(None, name, "unvalidated_execution_shape")
            family = (
                "fallback_models" if support.get("model") == "fallback" else "models"
            )
            durations = [evaluate(model, x) for model in self.data[family]]
            if any(not math.isfinite(v) or v <= 0 for v in durations):
                raise ValueError("invalid execution duration")
            completion = max(r[2] + d * 1e6 for r, d in zip(starts, durations))
            completion += support["correction_ms"] * 1e6
            return CompletionEstimate(completion, name, "observed_all_rank_gpu_start")
        except (KeyError, ValueError, TypeError, IndexError):
            return CompletionEstimate(None, name, "invalid_execution_features")
