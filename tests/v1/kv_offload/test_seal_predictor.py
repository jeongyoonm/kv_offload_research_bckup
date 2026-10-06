# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check estimates against synthetic compute costs and cold start fallback."""

import numpy as np
import pytest

from vllm.v1.kv_offload.seal_predictor import ComputeEstimator


def transfer_surface_state():
    from vllm.v1.kv_offload.latency_surface import FEATURE_NAMES

    return dict(
        schema=1,
        feature_names=FEATURE_NAMES,
        provenance={},
        trees=[
            [
                [1, 2, 9, 2.5, 125.0],
                [-1, -1, -2, -2.0, 100.0],
                [-1, -1, -2, -2.0, 150.0],
            ]
        ],
    )


def test_refinement_requires_all_cpu_submissions_and_round_trips_separately():
    """An incomplete or future submission cannot update the initial predictor."""
    from vllm.v1.kv_offload.latency_surface import (
        FEATURE_NAMES,
        FORECAST_NAMES,
        SUBMISSION_NAMES,
        LatencySurface,
    )

    state = transfer_surface_state()
    state.update(
        schema=2,
        target_schema="critical_end_after_all_rank_start.v1",
        feature_names=FEATURE_NAMES + FORECAST_NAMES + SUBMISSION_NAMES,
    )
    state["trees"][0][0][2:4] = [25, 5.0]
    model = ComputeEstimator()
    model.coefficients = np.ones(4)
    model.latency_surface = LatencySurface(transfer_surface_state())
    model.refinement_surface = LatencySurface(state)
    snapshot = model.state_dict({"model": "test"})
    restored = ComputeEstimator.from_state_dict(snapshot, snapshot["context"])
    assert restored.state_dict(snapshot["context"]) == snapshot
    data = dict(
        step=1,
        request_shapes=[[3072, 58368]],
        load_state=[0, 0],
        dispatch_ns=1_000_000,
        forecast_ns=10_000_000,
        forecast_load_features=[0] * 10,
        execution_snapshot=[[1, 2_000_000, 2_000_000, 2_000_000, 3_000_000, 1, 0, 0, 0]]
        * 8,
        primary_snapshot=[[1, 1_000_000, 1_500_000, 6_000_000, 0, 0, 6_000_000, 1, 0]]
        * 8,
    )
    assert restored.predict(data) == 100_000_000
    assert restored.refinement_surface.predict(data) == 100_000_000
    assert restored.refinement_surface.input_features(data)[23:].tolist() == [
        8,
        4,
        4,
        4,
        0,
        4,
    ]
    for index, value in [(0, 2), (3, 0), (6, 11_000_000), (7, 0)]:
        records = data["primary_snapshot"]
        assert isinstance(records, list)
        primary = [r.copy() for r in records]
        primary[-1][index] = value
        assert (
            restored.refinement_surface.predict(dict(data, primary_snapshot=primary))
            is None
        )
        assert restored.predict(dict(data, primary_snapshot=primary)) == 100_000_000


@pytest.mark.parametrize("native_backend", [False, True])
def test_saved_surface_uses_transfer_pressure_and_keeps_legacy_fallback(native_backend):
    """A reusable point model changes with loads; cold/decode keep old estimates."""
    import json
    import os

    from vllm.v1.kv_offload.latency_surface import LatencySurface

    if native_backend and not os.environ.get("VLLM_DEFERRAL_FAST_FOREST"):
        pytest.skip("native evaluator requires the optional CPU library")

    model = ComputeEstimator()
    model.coefficients = np.array([20.0, 0.0, 0.0, 0.0])
    model.latency_surface = LatencySurface(transfer_surface_state())
    state = json.loads(json.dumps(model.state_dict({"model": "test"})))
    restored = ComputeEstimator.from_state_dict(state, state["context"])
    restored.frozen = True
    warm = {
        "request_shapes": [[3072, 58368]],
        "load_state": [0, 0],
        "native_predictor": native_backend,
    }
    busy = dict(warm, load_state=[3, 175104])
    assert restored.predict(warm) == 100_000_000
    assert restored.predict(busy) == 150_000_000
    assert restored.predict({"request_shapes": [[3072, 0]]}) == 20_000_000
    assert restored.predict({"request_shapes": [[1, 58368]]}) == 20_000_000
    assert restored.predict(dict(warm, cpu_cached_prefix_tokens=[58368])) == 100_000_000
    assert restored.predict(dict(warm, cpu_cached_prefix_tokens=[0])) == 20_000_000
    assert restored.predict(dict(warm, cpu_cached_prefix_tokens=[])) == 20_000_000
    assert restored.state_dict(state["context"]) == state
    state["latency_surface"]["trees"][0][1][4] = 999
    assert restored.predict(warm) == 100_000_000
    assert (
        restored.state_dict(state["context"])["latency_surface"]
        == transfer_surface_state()
    )


def test_causal_surface_uses_current_receipts_and_validated_start_generation():
    from vllm.v1.kv_offload.latency_surface import (
        FEATURE_NAMES,
        FORECAST_NAMES,
        LatencySurface,
    )

    state = transfer_surface_state()
    state.update(
        schema=2,
        target_schema="critical_end_after_all_rank_start.v1",
        feature_names=FEATURE_NAMES + FORECAST_NAMES,
    )
    state["trees"][0][0][2] = 11
    model = LatencySurface(state)
    data = dict(
        step=1,
        request_shapes=[[3072, 58368]],
        load_state=[3, 300],
        dispatch_ns=1_000_000,
        forecast_ns=7_000_000,
        execution_snapshot=[[1, 4_000_000, 4_000_000, 4_000_000, 5_000_000, 1, 0, 0, 0]]
        * 8,
        forecast_load_features=[1, 100, 3.0, 3.0, 3.0, 1, 100, 0, 0, 0],
    )
    assert model.input_features(data)[11:].tolist() == [
        1,
        100,
        3.0,
        3.0,
        3.0,
        1,
        100,
        0,
        0,
        0,
        3.0,
        3.0,
    ]
    assert model.predict(data) == 100_000_000
    records = data["execution_snapshot"]
    assert isinstance(records, list)
    records[-1] = [
        2,
        4_000_000,
        4_000_000,
        4_000_000,
        5_000_000,
        1,
        0,
        0,
        0,
    ]
    assert model.predict(data) is None


def test_composed_surface_restores_signed_residual_and_keeps_state_private():
    """An encoded negative correction survives JSON without changing its base."""
    import copy
    import json

    from vllm.v1.kv_offload.latency_surface import (
        BASE_PREDICTION_NAMES,
        FEATURE_NAMES,
        FORECAST_NAMES,
        LatencySurface,
    )

    base = transfer_surface_state()
    base.update(schema=2, target_schema="critical_end_after_all_rank_start.v1")
    residual = dict(
        base,
        feature_names=FEATURE_NAMES + FORECAST_NAMES + BASE_PREDICTION_NAMES,
        trees=[[[-1, -1, -2, -2.0, 1004.0]]],
    )
    state = dict(
        schema=3,
        base=base,
        residual=residual,
        residual_shift_ms=1024.0,
        target_schema=base["target_schema"],
        provenance={},
    )
    model = LatencySurface(json.loads(json.dumps(state)))
    data = dict(
        step=1,
        request_shapes=[[3072, 58368]],
        load_state=[0, 0],
        dispatch_ns=1,
        forecast_ns=7_000_000,
        execution_snapshot=[[1, 4_000_000, 4_000_000, 4_000_000, 5_000_000, 1, 0, 0, 0]]
        * 8,
        forecast_load_features=[0] * 10,
    )
    assert model.predict(data) == 80_000_000
    assert model.input_features(data)[-1] == 100.0
    duration, used_features = model.predict_with_features(data)
    assert duration == model.predict(data)
    np.testing.assert_array_equal(used_features, model.input_features(data))
    assert model.predict_with_features(dict(data, forecast_load_features=None)) == (
        None,
        None,
    )
    exported = model.state_dict()
    exported["residual"]["trees"][0][0][4] = 9999.0
    assert model.predict(data) == 80_000_000
    assert model.state_dict() == state
    assert model.predict(dict(data, forecast_load_features=None)) is None
    bad = copy.deepcopy(state)
    bad["residual_shift_ms"] = 10**400
    with pytest.raises(ValueError):
        LatencySurface(bad)
    bad = copy.deepcopy(state)
    bad["base"] = copy.deepcopy(state)
    with pytest.raises(ValueError):
        LatencySurface(bad)


@pytest.mark.parametrize("mutation", ["cycle", "shared_child", "nan", "huge", "leaf"])
def test_surface_rejects_invalid_graphs_before_estimator_publication(mutation):
    """Untrusted saved graphs cannot hang inference or publish invalid costs."""
    from vllm.v1.kv_offload.latency_surface import LatencySurface

    state = transfer_surface_state()
    if mutation == "cycle":
        state["trees"][0][0][0] = 0
    elif mutation == "shared_child":
        state["trees"][0][0][1] = 1
    elif mutation == "nan":
        state["trees"][0][1][4] = float("nan")
    elif mutation == "huge":
        state["trees"][0][0][3] = 10**1000
    else:
        state["trees"][0][1][4] = -10
    with pytest.raises(ValueError):
        LatencySurface(state)


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


def test_recent_reserve_excludes_cold_and_mixed_preparation_for_cached_successor():
    estimator = ComputeEstimator()
    warm = {"request_shapes": [[1024, 30720]] * 4}
    mixed = {"request_shapes": [[1024, 30720]] * 3 + [[1024, 0]]}
    cold = {"request_shapes": [[61440, 0]]}
    for step, data in enumerate([cold] * 16 + [mixed] * 8 + [warm] * 8, 1):
        duration = 20_000_000 if data is warm else 700_000_000
        estimator.observe_preparation(
            [(step, 1, duration + 1, 0, 0, 0, 0, 1, 0)] * 8, data
        )
    assert estimator.recent_preparation_reserve(10_000_000, warm) == 30_000_000
    assert (
        estimator.recent_preparation_reserve(
            10_000_000, {"request_shapes": [[4096, 0]] * 4}
        )
        == 10_000_000
    )


def test_recent_reserve_tracks_changed_cost_and_resists_a_single_outlier():
    estimator = ComputeEstimator()
    shape = {"request_shapes": [[1024, 30720]] * 4}
    for step in range(1, 49):
        duration = 700_000_000 if step <= 8 or step == 48 else 20_000_000
        estimator.observe_preparation(
            [(step, 1, duration + 1, 0, 0, 0, 0, 1, 0)] * 8, shape
        )
    assert estimator.recent_preparation_reserve(10_000_000, shape) == 30_000_000


def test_frozen_estimator_preserves_forecast_and_preparation_reserve():
    """An A/B replay must not retrain the common forecast or reserve."""
    import numpy as np

    estimator = ComputeEstimator()
    data = {"step": 1, "request_shapes": [[1024, 30720]]}
    estimator.coefficients = np.array([100.0, 0.0, 0.0, 0.0])
    records = [(1, 1, 20_000_001, 0, 0, 0, 0, 1, 0)] * 8
    estimator.observe_preparation(records, data)
    forecast = estimator.predict(data)
    reserve = estimator.preparation_reserve(10_000_000, data)
    estimator.frozen = True
    estimator.observe(data, [(1, 1, 1, 1, 1, 1, 700_000_000, 1, 1)] * 8)
    estimator.observe_preparation([(1, 1, 700_000_001, 0, 0, 0, 0, 1, 0)] * 8, data)
    assert not estimator.samples
    assert len(estimator.preparation_ns) == 1
    assert estimator.predict(data) == forecast
    assert estimator.preparation_reserve(10_000_000, data) == reserve


def test_raw_calibration_does_not_dampen_its_previous_correction():
    """A constant raw underestimate must receive the full measured correction."""
    estimator = ComputeEstimator()
    estimator.coefficients = np.array([100.0, 0.0, 0.0, 0.0])
    estimator.calibration_mode = "raw"
    data = {"step": 1, "request_shapes": [[3072, 58368]] * 2}
    execution = [(1, 1, 2, 3, 10, 1, 200_000_000, 8, 9)] * 8
    for _ in range(6):
        estimator.observe(data, execution)
        assert estimator.predict(data) == 200_000_000


def test_shape_calibration_separates_small_cached_batches_and_cold_work():
    """Global residuals must not hide a repeatable small cached-batch cost."""
    estimator = ComputeEstimator()
    estimator.coefficients = np.array([100.0, 0.0, 0.0, 0.0])
    for size, context, duration in [(2, 58368, 180), (8, 58368, 100), (1, 0, 1000)]:
        for _ in range(2):
            data = {"step": 1, "request_shapes": [[3072, context]] * size}
            estimator.observe(
                data, [(1, 1, 2, 3, 10, 1, duration * 1_000_000, 8, 9)] * 8
            )
    estimator.calibration_mode = "shape"
    assert estimator.predict({"request_shapes": [[3072, 58368]] * 2}) == 180_000_000
    assert estimator.predict({"request_shapes": [[3072, 58368]] * 8}) == 100_000_000
    assert estimator.predict({"request_shapes": [[3072, 0]]}) == 1_000_000_000


@pytest.mark.parametrize("mode", ["legacy", "raw", "shape"])
def test_json_snapshot_preserves_predictions_and_reserves_without_retraining(mode):
    import json

    estimator = ComputeEstimator()
    for step in range(1, 25):
        data = {"step": step, "request_shapes": [[512 * (1 + step % 8), 8192]]}
        f = estimator.features(data)
        n = int((5 + 90 * f[1] + 15 * f[2]) * 1e6)
        estimator.observe(data, [(step, 1, 2, 3, 10, 1, n, 8, 9)] * 8)
        estimator.observe_preparation(
            [(step, 1, 20_000_001, 0, 0, 0, 0, 1, 0)] * 8, data
        )
    context = {"model": "test-model", "tp": 8, "kv_dtype": "fp8"}
    state = json.loads(json.dumps(estimator.state_dict(context), allow_nan=False))
    restored = ComputeEstimator.from_state_dict(state, context)
    estimator.calibration_mode = restored.calibration_mode = mode
    estimator.frozen = restored.frozen = True
    query = {"step": 100, "request_shapes": [[2048, 8192]]}
    restored.observe(query, [(100, 1, 2, 3, 10, 1, 900_000_000, 8, 9)] * 8)
    restored.observe_preparation([(100, 1, 900_000_001, 0, 0, 0, 0, 1, 0)] * 8, query)
    assert restored.state_dict(context) == state
    assert restored.predict(query) == estimator.predict(query)
    assert restored.preparation_reserve(
        10_000_000, query
    ) == estimator.preparation_reserve(10_000_000, query)
    assert restored.recent_preparation_reserve(
        10_000_000, query
    ) == estimator.recent_preparation_reserve(10_000_000, query)


@pytest.mark.parametrize(
    "corruption", ["context", "schema", "nan", "vector", "overflow", "regime"]
)
def test_snapshot_rejects_incompatible_or_corrupt_training_state(corruption):
    estimator = ComputeEstimator()
    estimator.coefficients = np.array([100.0, 0.0, 0.0, 0.0])
    context = {"model": "test-model", "tp": 8}
    state = estimator.state_dict(context)
    if corruption == "context":
        state["context"] = {"model": "other-model", "tp": 8}
    elif corruption == "schema":
        state["schema"] = 99
    elif corruption == "nan":
        state["coefficients"][0] = float("nan")
    elif corruption == "vector":
        state["coefficients"] = [100.0]
    elif corruption == "overflow":
        state["errors"] = [1.0] * 33
    else:
        state["calibration_samples"] = [[[1, 1, 1, 1], [1, 1, 1], "invalid", 100]]
    with pytest.raises(ValueError):
        ComputeEstimator.from_state_dict(state, context)
