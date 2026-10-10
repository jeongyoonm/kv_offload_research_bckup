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


def shape_correction_state():
    from vllm.v1.kv_offload.seal_calibration import FEATURES

    return dict(
        schema=1,
        features=FEATURES,
        coefficients=[10, 0, 0, 0, 100],
        support=dict(requests=[1, 32], context=[15552, 58368], suffix=[832, 3072]),
        bias_bounds_ms=[-500, 500],
        provenance={},
    )


def test_shape_correction_changes_with_prediction_and_ignores_forecast_age():
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    correction = BatchShapeCalibration(shape_correction_state())
    x = np.zeros(28)
    x[:9] = [3328, 4, 0, 62208, 832, 832, 15552, 15552, 3584]
    assert correction.bias_ns(200_000_000, x, 90_000_000) == (30_000_000, False)
    x[21] = 1000
    assert correction.bias_ns(400_000_000, x, 90_000_000) == (50_000_000, False)


@pytest.mark.parametrize("context,suffix", [(8192, 832), (61440, 3072), (32768, 8192)])
def test_shape_correction_uses_original_bias_outside_calibrated_support(
    context, suffix
):
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    correction = BatchShapeCalibration(shape_correction_state())
    x = np.zeros(28)
    x[:9] = [suffix * 4, 4, 0, context * 4, suffix, suffix, context, context, 0]
    assert correction.bias_ns(200_000_000, x, 90_000_000) == (90_000_000, True)
    assert correction.bias_ns(200_000_000, None, 90_000_000) == (90_000_000, True)


@pytest.mark.parametrize("mutation", ["nonfinite", "width", "support", "bounds"])
def test_shape_correction_rejects_corrupt_artifacts_before_publication(mutation):
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    state = shape_correction_state()
    if mutation == "nonfinite":
        state["coefficients"][0] = float("nan")
    elif mutation == "width":
        state["coefficients"].pop()
    elif mutation == "support":
        state["support"]["requests"] = [0, 32]
    else:
        state["bias_bounds_ms"] = [500, -500]
    with pytest.raises(ValueError, match="shape calibration"):
        BatchShapeCalibration(state)


@pytest.mark.parametrize(
    "context,suffix,supported",
    [
        (15552, 832, True),
        (58368, 3072, True),
        (30720, 1664, True),
        (15552, 3072, False),
        (58368, 832, False),
        (30720, 2048, False),
    ],
)
def test_joint_guard_distinguishes_geometry_inside_marginal_bounds(
    context, suffix, supported
):
    """Context/suffix combinations require support beyond separate ranges."""
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    state = shape_correction_state()
    old = BatchShapeCalibration(state)
    state.update(schema=2, geometry_support=dict(suffix_per_context=[0.05, 0.057]))
    guarded = BatchShapeCalibration(state)
    x = np.zeros(28)
    x[:9] = [suffix * 4, 4, 0, context * 4, suffix, suffix, context, context, 0]
    assert old.bias_ns(200_000_000, x, 90_000_000) == (30_000_000, False)
    assert guarded.bias_ns(200_000_000, x, 90_000_000) == (
        (30_000_000, False) if supported else (90_000_000, True)
    )


@pytest.mark.parametrize("bounds", [[0, 0.1], [0.1, 0.05], [0.05], [0.05, np.nan]])
def test_joint_guard_rejects_invalid_support_before_checkpoint_publication(bounds):
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    state = shape_correction_state()
    state.update(schema=2, geometry_support=dict(suffix_per_context=bounds))
    with pytest.raises(ValueError, match="geometry support"):
        BatchShapeCalibration(state)


def test_explicit_suffix_basis_changes_bias_with_equal_context_and_request_count():
    """Crossed suffix length must influence correction independently of context."""
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    state = shape_correction_state()
    state.update(
        schema=3,
        features=["intercept", "suffix_sum_per_61440"],
        coefficients=[10, 100],
        geometry_support=dict(suffix_per_context=[0.01, 0.25]),
    )
    correction = BatchShapeCalibration(state)
    x = np.zeros(28)
    x[:9] = [3072, 1, 0, 15552, 3072, 3072, 15552, 15552, 0]
    assert correction.bias_ns(200_000_000, x, 90_000_000) == (15_000_000, False)
    x[0], x[4], x[5] = 832, 832, 832
    value, fallback = correction.bias_ns(200_000_000, x, 90_000_000)
    assert 11_000_000 < value < 12_000_000 and not fallback


def test_attention_basis_falls_back_when_required_work_feature_is_missing():
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    state = shape_correction_state()
    state.update(
        schema=3,
        features=["prefix_index_pairs_per_1e9"],
        coefficients=[100],
        geometry_support=dict(suffix_per_context=[0.01, 0.25]),
    )
    correction = BatchShapeCalibration(state)
    x = np.array([832, 1, 0, 15552, 832, 832, 15552, 15552, 0, 0, 0])
    assert correction.bias_ns(200_000_000, x, 90_000_000) == (90_000_000, True)
    expanded = np.r_[x, np.zeros(12), 100_000_000]
    assert correction.bias_ns(200_000_000, expanded, 90_000_000) == (10_000_000, False)


def test_calibration_schema_rejects_elapsed_and_unknown_features():
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    state = shape_correction_state()
    state.update(schema=3, geometry_support=dict(suffix_per_context=[0.01, 0.25]))
    for feature in ("elapsed_execution_ms", "unrecognized"):
        state["features"] = [feature]
        state["coefficients"] = [1]
        with pytest.raises(ValueError, match="schema"):
            BatchShapeCalibration(state)


def shape_floor_state():
    state = shape_correction_state()
    state.update(
        schema=7,
        geometry_support=dict(suffix_per_context=[0.01, 0.25]),
        submission_floor=dict(
            host_full_ms=dict(small_short=180),
            offset_feature_index=26,
            homogeneous_only=True,
        ),
    )
    return state


def test_submission_floor_subtracts_start_offset_without_learning_elapsed():
    """An early device start leaves more host submission work on the critical path."""
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    model = BatchShapeCalibration(shape_floor_state())
    x = np.zeros(28)
    x[:9] = [832, 1, 0, 15552, 832, 832, 15552, 15552, 0]
    assert model.bias_ns(200_000_000, x, 90_000_000) == (20_000_000, False)
    x[21] = 100  # Forecast elapsed time does not alter the duration model.
    assert model.bias_ns(200_000_000, x, 90_000_000) == (20_000_000, False)
    x[26] = 80
    assert model.bias_ns(200_000_000, x, 90_000_000) == (30_000_000, False)


@pytest.mark.parametrize("change", ["mixed_suffix", "mixed_context", "missing", "nan"])
def test_submission_floor_retains_correction_for_untrained_or_unknown_offset(change):
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    model = BatchShapeCalibration(shape_floor_state())
    x = np.zeros(28)
    x[:9] = [1664, 2, 0, 31104, 832, 832, 15552, 15552, 0]
    if change == "mixed_suffix":
        x[5] = 1000
    elif change == "mixed_context":
        x[7] = 16000
    elif change == "missing":
        x = x[:26]
    else:
        x[26] = np.nan
    assert model.submission_floor_ns(x) is None
    assert model.bias_ns(200_000_000, x, 90_000_000) == (30_000_000, False)


def test_submission_floor_preserves_legacy_correction_bounds():
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    state = shape_floor_state()
    state["submission_floor"]["host_full_ms"]["small_short"] = 1000
    model = BatchShapeCalibration(state)
    x = np.zeros(28)
    x[:9] = [832, 1, 0, 15552, 832, 832, 15552, 15552, 0]
    assert model.bias_ns(200_000_000, x, 90_000_000) == (-500_000_000, False)


def shape_margin_state():
    state = shape_correction_state()
    state.update(
        schema=4,
        geometry_support=dict(suffix_per_context=[0.01, 0.25]),
        deadline_margin=dict(
            finish_allowance_ms=dict(small_short=30, large_short=0),
            setup_allowance_ms=dict(
                small_short=8,
                medium_short=15,
                large_short=12,
                small_long=20,
                medium_long=30,
                large_long=40,
            ),
            finish_fallback_ms=40,
            setup_fallback_ms=35,
            extra_ms=2,
            bounds_ms=[10, 100],
        ),
    )
    return state


@pytest.mark.parametrize("combined", [False, True])
def test_deadline_margin_separates_predecessor_error_from_successor_setup(combined):
    """Identical A shape can need different margins for already-known B shapes."""
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    state = shape_margin_state()
    if combined:
        state.update(
            schema=6,
            productive_load_policy=productive_correction_state()[
                "productive_load_policy"
            ],
        )
    calibration = BatchShapeCalibration(state)
    if combined:
        assert calibration.productive_load_policy.ready(
            [], [[3072, 58368]] * 3, [[3072, 58368]] * 16
        )[0]
    x = np.zeros(28)
    x[:9] = [9984, 12, 0, 186624, 832, 832, 15552, 15552, 9984]
    short = [[832, 15552]]
    long = [[3072, 58368]]
    assert calibration.deadline_lead_ns(
        200_000_000, x, short, short * 7, 50_000_000
    ) == (17_000_000, False)
    assert calibration.deadline_lead_ns(
        200_000_000, x, long, long * 15, 50_000_000
    ) == (42_000_000, False)
    assert calibration.deadline_lead_ns(200_000_000, x, short * 12, [], 50_000_000) == (
        14_000_000,
        False,
    )
    x[0], x[1], x[3] = 1664, 2, 31104
    assert calibration.deadline_lead_ns(200_000_000, x, short * 12, [], 50_000_000) == (
        44_000_000,
        False,
    )


@pytest.mark.parametrize(
    "base,candidates",
    [
        (None, []),
        ([], None),
        ([], []),
        ([[1, 15552]], []),
        ([[832, 15552]], [[832, 15552]] * 32),
    ],
)
def test_deadline_margin_falls_back_for_unknown_or_unsupported_successor(
    base, candidates
):
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    calibration = BatchShapeCalibration(shape_margin_state())
    x = np.zeros(28)
    x[:9] = [832, 1, 0, 15552, 832, 832, 15552, 15552, 896]
    assert calibration.deadline_lead_ns(
        200_000_000, x, base, candidates, 50_000_000
    ) == (50_000_000, True)


@pytest.mark.parametrize("mutation", ["nan", "unknown_bin", "bounds"])
def test_deadline_margin_rejects_corrupt_frozen_allowances(mutation):
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    state = shape_margin_state()
    margin = state["deadline_margin"]
    if mutation == "nan":
        margin["extra_ms"] = float("nan")
    elif mutation == "unknown_bin":
        margin["finish_allowance_ms"]["unseen"] = 1
    else:
        margin["bounds_ms"] = [100, 10]
    with pytest.raises(ValueError, match="deadline margin"):
        BatchShapeCalibration(state)


def productive_correction_state():
    from vllm.v1.kv_offload.productive_load_policy import DIRECT_FEATURES

    state = shape_correction_state()
    state.update(
        schema=5,
        geometry_support=dict(suffix_per_context=[0.01, 0.25]),
        productive_load_policy=dict(
            schema=1,
            features=DIRECT_FEATURES,
            coefficients=[80, 0, 1000, 0, 0, 10],
            work_coefficients=[1000, 10],
            minimum_efficiency=0.7,
            minimum_ready_candidates=1,
            provenance={},
        ),
    )
    return state


def test_productive_exit_uses_ready_work_shape_instead_of_fixed_request_fraction():
    """Larger suffixes provide a productive batch with fewer ready requests."""
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    model = BatchShapeCalibration(productive_correction_state()).productive_load_policy
    targets = []
    for shape in ([832, 15552], [3072, 58368]):
        cohort = [shape] * 16
        targets.append(
            next(n for n in range(1, 17) if model.ready([], cohort[:n], cohort)[0])
        )
    assert targets[1] < targets[0] < 8
    assert model.ready([[3072, 58368]] * 8, [], [[3072, 58368]] * 8) == (False, None)


@pytest.mark.parametrize(
    "shapes", [None, [], [[8192, 15552]], [[832, 0]], [[832, 15552]] * 33]
)
def test_productive_exit_preserves_fixed_policy_for_unknown_or_unsupported_shapes(
    shapes,
):
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    model = BatchShapeCalibration(productive_correction_state()).productive_load_policy
    assert model.ready([], [[832, 15552]], shapes) == (False, None)


def test_productive_exit_weights_mixed_uncached_work_without_future_inputs():
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    model = BatchShapeCalibration(productive_correction_state()).productive_load_policy
    short, long = [832, 15552], [3072, 58368]
    active = [short] * 8 + [long] * 8
    short_score = model.ready([], [short] * 4, active)[1]
    long_score = model.ready([], [long] * 4, active)[1]
    assert long_score > short_score


@pytest.mark.parametrize(
    "mutation", ["nonfinite", "negative_work", "efficiency", "width"]
)
def test_productive_exit_rejects_invalid_frozen_models(mutation):
    from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

    state = productive_correction_state()
    policy = state["productive_load_policy"]
    if mutation == "nonfinite":
        policy["coefficients"][0] = float("nan")
    elif mutation == "negative_work":
        policy["work_coefficients"][0] = -1
    elif mutation == "efficiency":
        policy["minimum_efficiency"] = 0
    else:
        policy["coefficients"].pop()
    with pytest.raises(ValueError, match="productive load policy"):
        BatchShapeCalibration(state)


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_offset_predictor_ignores_elapsed_time_in_primary_and_fallback(
    fallback, packed, monkeypatch
):
    """Even an elapsed-sensitive component cannot move an offset deadline."""
    import copy

    from vllm.v1.kv_offload.frozen_seal_predictor import FrozenSealPredictor
    from vllm.v1.kv_offload.latency_surface import FEATURE_NAMES, FORECAST_NAMES

    tree = transfer_surface_state()
    tree.update(
        schema=2,
        target_schema="critical_end_after_all_rank_start.v1",
        feature_names=FEATURE_NAMES + FORECAST_NAMES,
    )
    tree["trees"][0][0][2:4] = [21, 5.0]
    alternate = copy.deepcopy(tree)
    alternate["trees"][0][1][4] = 200.0
    state = dict(
        schema=4,
        architecture="guarded_sparse_mlp_offset",
        target_schema="critical_end_after_all_rank_start.v1",
        provenance=dict(general_prefill=True, packed_checkpoint=packed),
        checkpoint=dict(
            models=[dict(direct=dict(kind="tree", state=s)) for s in [tree, alternate]],
            weights=[1.0, 0.0],
            excluded_feature_indices=[21],
            support_guard=dict(
                indices=[0],
                minimum=[0],
                maximum=[32 if fallback else 100],
                fallback_model_index=1,
            ),
        ),
    )
    monkeypatch.setenv("VLLM_SEAL_PACKED_CHECKPOINT", "0")
    model = FrozenSealPredictor(state)
    assert model.state_dict() == state
    exported = model.state_dict()
    exported["checkpoint"]["weights"][0] = 0.0
    assert model.state_dict() == state
    if packed:
        assert "checkpoint" not in model._state
        assert "trees" not in model.neural.models[0].direct.tree._state
    data = dict(
        step=1,
        request_shapes=[[64, 64]],
        load_state=[0, 0],
        predictor_general_prefill=True,
        prefill_only=True,
        dispatch_ns=1_000_000,
        forecast_load_features=[0] * 10,
        execution_snapshot=[[1, 1_000_000, 2_000_000, 2_000_000, 0, 1, 0, 0, 0]] * 8,
    )
    expected = 200_000_000 if fallback else 100_000_000
    for elapsed_ms in [0, 10, 100]:
        query = dict(data, forecast_ns=2_000_000 + elapsed_ms * 1_000_000)
        assert model.predict(query) == expected


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


@pytest.mark.parametrize("packed", [False, True])
def test_composed_surface_restores_signed_residual_and_keeps_state_private(
    packed, monkeypatch
):
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
    monkeypatch.setenv("VLLM_SEAL_PACKED_CHECKPOINT", str(int(packed)))
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
