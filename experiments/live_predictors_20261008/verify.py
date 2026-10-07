# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check frozen serving outputs against research inference on recorded data."""

import copy
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))
for package in ["vllm", "vllm.v1", "vllm.v1.kv_offload"]:
    module = types.ModuleType(package)
    module.__path__ = [str(SOURCE / package.replace(".", "/"))]
    sys.modules[package] = module
RESEARCH = Path("/mnt/localnvme/vllm_glm53_envelope_blend_20261007")
sys.path.insert(0, str(RESEARCH / "experiments/predictor_launch_envelope_20261007"))
import envelope_blend as blend  # noqa: E402

from vllm.v1.kv_offload.frozen_seal_predictor import FrozenSealPredictor  # noqa: E402
from vllm.v1.kv_offload.seal_predictor import ComputeEstimator  # noqa: E402

ROOT = Path("/mnt/localnvme/glm53_live_predictors_20261008")
CHECKPOINTS = {
    "guarded_sparse_mlp": Path(
        "/mnt/localnvme/glm53_predictor_expanded_20261007/exports/improvement/guarded_sparse_mlp/seed73.json"
    ),
    "envelope_sparse_mlp": Path(
        "/mnt/localnvme/glm53_envelope_blend_20261007/exports/envelope_sparse_mlp/seed73.json"
    ),
}


def surface_state(name):
    path = CHECKPOINTS[name]
    return dict(
        schema=4,
        architecture=name,
        checkpoint=json.loads(path.read_text()),
        target_schema="critical_end_after_all_rank_start.v1",
        provenance=dict(
            general_prefill=True,
            seed=73,
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
    )


def main():
    os.environ["VLLM_DEFERRAL_FAST_FOREST"] = (
        "/mnt/localnvme/glm53_predictor_overnight_20261006/fast_forest/build/libpredictor_forest_walk.so"
    )
    pool = joblib.load(
        "/mnt/localnvme/glm53_predictor_expanded_20261007/benchmark_pool.joblib"
    )
    rows = pool["forecasts"]
    x = blend.e.features(rows)
    report = {}
    with threadpool_limits(limits=1):
        for name in CHECKPOINTS:
            state = surface_state(name)
            model = FrozenSealPredictor(state)
            reference = (
                blend.Frozen(state["checkpoint"], True)
                if name == "envelope_sparse_mlp"
                else blend.e.improve.FrozenCandidate(state["checkpoint"], True)
            )
            differences = []
            for row, features in zip(rows, x):
                data = dict(
                    row,
                    predictor_general_prefill=True,
                    prefill_only=True,
                    forecast_ns=row["forecast_time_ns"],
                    forecast_load_features=row["forecast_features"][:10],
                    execution_snapshot=row["execution_ranks"],
                    native_predictor=True,
                )
                extracted = model.input_features(data)
                np.testing.assert_allclose(extracted, features, rtol=0, atol=1e-9)
                expected = reference.predict_one(
                    features
                    if name == "envelope_sparse_mlp"
                    else np.r_[features[:23], np.zeros(5), features[23:26]]
                )
                actual = model.predict(data) / 1e6
                differences.append(abs(expected - actual))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
            estimator = ComputeEstimator()
            estimator.coefficients = np.ones(4)
            estimator.latency_surface = model
            snapshot = estimator.state_dict({"model": "test"})
            restored = ComputeEstimator.from_state_dict(snapshot, snapshot["context"])
            assert restored.state_dict(snapshot["context"]) == snapshot
            # Partial and future rank snapshots must decline to arm a deadline.
            for bad in [
                dict(data, execution_snapshot=data["execution_snapshot"][:-1]),
                dict(data, forecast_ns=1),
            ]:
                assert restored.latency_surface.predict(bad) is None
            mutated = copy.deepcopy(state)
            mutated["checkpoint"] = {}
            assert model.state_dict() == state
            malformed = copy.deepcopy(state)
            target = malformed["checkpoint"].get("neural", malformed["checkpoint"])
            target["models"][0]["direct"]["x_scale"][0] = 0
            try:
                FrozenSealPredictor(malformed)
            except ValueError:
                pass
            else:
                raise AssertionError("accepted zero normalization scale")
            report[name] = dict(
                rows=len(rows),
                maximum_prediction_difference_ms=max(differences),
                feature_equivalence=True,
                round_trip=True,
                causality_checks=True,
            )
    ROOT.mkdir(exist_ok=True)
    (ROOT / "cpu_equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
