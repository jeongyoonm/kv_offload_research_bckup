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


def test_preparation_reserve_requires_a_complete_consistent_rank_snapshot():
    estimator = ComputeEstimator()
    records = [(1, 1, 20_000_001, 0, 0, 0, 0, 1, 0)] * 8
    estimator.observe_preparation(records)
    assert estimator.preparation_reserve(10_000_000) == 30_000_000
    estimator.observe_preparation(records[:7] + [None])
    estimator.observe_preparation(records[:7] + [(2, 1, 100_000_001, 0, 0, 0, 0, 1, 0)])
    assert len(estimator.preparation_ns) == 1


def test_cached_successor_reserve_does_not_use_cold_priming_cost():
    """A cold 61K batch must not force a 700ms reserve on a cached 4K batch."""
    estimator = ComputeEstimator()
    cold = {"request_shapes": [[61440, 0]]}
    warm = {"request_shapes": [[1024, 30720]] * 4}
    for step in range(1, 25):
        duration = 700_000_000 if step <= 16 else 20_000_000
        data = cold if step <= 16 else warm
        records = [(step, 1, duration + 1, 0, 0, 0, 0, 1, 0)] * 8
        estimator.observe_preparation(records, data)
    assert estimator.preparation_reserve(10_000_000) == 710_000_000
    assert estimator.preparation_reserve(10_000_000, warm) == 30_000_000
    assert (
        estimator.preparation_reserve(10_000_000, {"request_shapes": []}) == 10_000_000
    )


def test_unknown_successor_shape_uses_measured_token_rate():
    estimator = ComputeEstimator()
    records = [(1, 1, 700_000_001, 0, 0, 0, 0, 1, 0)] * 8
    estimator.observe_preparation(records, {"request_shapes": [[61440, 0]]})
    upcoming = {"request_shapes": [[3072, 58368]]}
    assert estimator.preparation_reserve(10_000_000, upcoming) == 45_000_000
