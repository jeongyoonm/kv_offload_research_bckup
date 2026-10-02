# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check estimates against synthetic compute costs and cold start fallback."""

import pytest

from vllm.v1.kv_offload.seal_predictor import ComputeEstimator


def test_predictor_learns_uncached_tokens_and_attention_cost():
    estimator = ComputeEstimator()
    assert estimator.predict({"request_shapes": [[1024, 4096]]}) is None
    for step in range(1, 65):
        n = 512 * (1 + step % 16)
        context = 1024 * (1 + step % 23)
        data = {"step": step, "request_shapes": [[n, context]] * (1 + step % 4)}
        f = estimator.features(data)
        duration_ns = int((5 + 90 * f[1] + 15 * f[2] + 2 * f[3]) * 1e6)
        estimator.observe(data, [(step, 1, 2, 3, 10, 1, duration_ns, 8, 9)] * 8)
    short = {"request_shapes": [[1024, 8192]] * 4}
    long = {"request_shapes": [[8192, 8192]] * 4}
    features = estimator.features(long)
    expected = int((5 + 90 * features[1] + 15 * features[2] + 2 * features[3]) * 1e6)
    assert estimator.predict(long) > estimator.predict(short)
    assert estimator.predict(long) == pytest.approx(expected, rel=0.15)


def test_predictor_rejects_incomplete_rank_observation():
    estimator = ComputeEstimator()
    estimator.observe({"step": 1, "request_shapes": [[1024, 0]]}, [None] * 8)
    assert not estimator.samples
