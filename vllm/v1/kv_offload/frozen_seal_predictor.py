# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen experimental MLP and submission-envelope inference without trainers."""

import copy

import numpy as np

from vllm.v1.kv_offload.latency_surface import (
    ENVELOPE_NAMES,
    FEATURE_NAMES,
    FORECAST_NAMES,
    SPARSE_NAMES,
    LatencySurface,
    packed_checkpoint_enabled,
    restored_checkpoint,
    retain_checkpoint,
)

CORE_INDEX = [0, 1, 8, 23, 24, 25, 11, 12]
CORE_SCALE = np.array([1000, 16, 1000, 1e8, 1e8, 1e7, 8, 1e6])
HOST_INDEX = [1, 8, 11, 12, 13, 15, 16, 18]
HOST_SCALE = np.array([16, 1000, 8, 1e6, 1000, 1000, 8, 8])


def finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite predictor checkpoint")
    return array


class Component:
    def __init__(self, state, packed=False):
        self.state = state
        if state["kind"] == "tree":
            self.tree = LatencySurface(state["state"], packed=packed)
            self.shift = float(state.get("shift", 0))
            if not np.isfinite(self.shift):
                raise ValueError("invalid tree residual shift")
        elif state["kind"] == "mlp":
            self.mean = finite_array(state["x_mean"])
            self.scale = finite_array(state["x_scale"])
            if self.mean.ndim != 1 or self.scale.shape != self.mean.shape:
                raise ValueError("invalid MLP normalization shape")
            if (self.scale <= 0).any():
                raise ValueError("invalid MLP normalization scale")
            if state["config"]["input"] not in ("standard", "log1p"):
                raise ValueError("unsupported MLP input transform")
            if not 1 <= len(state["weights"]) == len(state["biases"]) <= 4:
                raise ValueError("invalid MLP layer count")
            self.layers = []
            width = len(self.mean)
            for w, b in zip(state["weights"], state["biases"]):
                w, b = finite_array(w), finite_array(b)
                if w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],):
                    raise ValueError("invalid MLP layer shape")
                if w.size > 1_000_000:
                    raise ValueError("MLP layer exceeds capacity")
                self.layers.append((w, b))
                width = len(b)
            if width != 1:
                raise ValueError("invalid MLP output shape")
            self.output = finite_array([state["y_scale"], state["y_mean"]])
        else:
            raise ValueError("unsupported predictor component")
        if packed:
            self.state = {"kind": state["kind"], "config": state.get("config")}

    def predict(self, x, native):
        if self.state["kind"] == "tree":
            return (
                self.tree._predict_features(x, {"native_predictor": native}) / 1e6
                - self.shift
            )
        if self.state["config"]["input"] == "log1p":
            x = np.sign(x) * np.log1p(np.abs(x))
        value = (x - self.mean) / self.scale
        for i, (weight, bias) in enumerate(self.layers):
            value = value @ weight + bias
            if i < len(self.layers) - 1:
                value = np.maximum(value, 0)
        return float(value[0] * self.output[0] + self.output[1])


class Architecture:
    def __init__(self, state, packed=False):
        self.indices = state.get("input_indices")
        if self.indices is not None and any(
            type(i) is not int or not 0 <= i < 31 for i in self.indices
        ):
            raise ValueError("invalid predictor input projection")
        self.direct = (
            Component(state["direct"], packed) if state.get("direct") else None
        )
        self.base = Component(state["base"], packed) if state.get("base") else None
        self.correction = (
            Component(state["correction"], packed) if state.get("correction") else None
        )
        if self.direct is None and self.base is None:
            raise ValueError("predictor has no base")

    def predict(self, x, native):
        x = x[:23] if self.indices is None else x[self.indices]
        if self.direct is not None:
            return max(1, self.direct.predict(x, native))
        assert self.base is not None
        value = self.base.predict(x[:11], native)
        if self.correction is not None:
            value += self.correction.predict(np.r_[x, value], native)
        return max(1, value)


class Support:
    def __init__(self, state, width):
        self.indices = state["indices"]
        self.minimum = finite_array(state["minimum"])
        self.maximum = finite_array(state["maximum"])
        if (
            any(type(i) is not int or not 0 <= i < width for i in self.indices)
            or self.minimum.shape != (len(self.indices),)
            or self.maximum.shape != self.minimum.shape
            or (self.minimum > self.maximum).any()
        ):
            raise ValueError("invalid predictor support bounds")

    def outside(self, x):
        values = x[self.indices]
        return bool(((values < self.minimum) | (values > self.maximum)).any())


class NeuralCandidate:
    def __init__(self, state, packed=False):
        self.models = [Architecture(s, packed) for s in state["models"]]
        self.weights = finite_array(state["weights"])
        self.support = Support(state["support_guard"], 31)
        self.fallback = state["support_guard"]["fallback_model_index"]
        if (
            self.weights.shape != (len(self.models),)
            or len(self.models) != 2
            or type(self.fallback) is not int
            or not 0 <= self.fallback < len(self.models)
            or (self.weights < 0).any()
            or not np.isclose(self.weights.sum(), 1)
        ):
            raise ValueError("invalid guarded predictor weights")

    def predict(self, x, native):
        if self.support.outside(x):
            return self.models[self.fallback].predict(x, native)
        return max(
            1,
            sum(
                w * m.predict(x, native) for w, m in zip(self.weights, self.models) if w
            ),
        )


class FrozenSealPredictor:
    requires_forecast = True
    requires_load_progress = True
    target_schema = "critical_end_after_all_rank_start.v1"
    backend = "numpy_mlp+native_forest"
    base = None

    def __init__(self, state):
        if (
            not isinstance(state, dict)
            or set(state)
            != {"schema", "architecture", "checkpoint", "provenance", "target_schema"}
            or state["schema"] != 4
            or state["architecture"]
            not in (
                "guarded_sparse_mlp",
                "guarded_sparse_mlp_offset",
                "envelope_sparse_mlp",
            )
            or state["target_schema"] != self.target_schema
            or state["provenance"].get("general_prefill") is not True
        ):
            raise ValueError("incompatible frozen predictor snapshot")
        self._state = copy.deepcopy(state)
        self._packed_state = None
        packed = state["provenance"].get(
            "packed_checkpoint", packed_checkpoint_enabled()
        )
        if type(packed) is not bool:
            raise ValueError("invalid packed checkpoint option")
        self.architecture = state["architecture"]
        self.excludes_elapsed = self.architecture == "guarded_sparse_mlp_offset"
        self.feature_names = (
            FEATURE_NAMES + FORECAST_NAMES + SPARSE_NAMES + ENVELOPE_NAMES
        )
        checkpoint = self._state["checkpoint"]
        self.duration_calibration = None
        if "duration_calibration" in checkpoint:
            from vllm.v1.kv_offload.seal_calibration import BatchShapeCalibration

            self.duration_calibration = BatchShapeCalibration(
                checkpoint["duration_calibration"]
            )
        if self.excludes_elapsed and checkpoint.get("excluded_feature_indices") != [21]:
            raise ValueError("offset predictor must exclude elapsed execution time")
        self.neural = NeuralCandidate(checkpoint.get("neural", checkpoint), packed)
        if all(
            component["kind"] == "tree"
            for model in checkpoint.get("neural", checkpoint)["models"]
            for key in ("direct", "base", "correction")
            if (component := model.get(key)) is not None
        ):
            self.backend = "native_forest"
        if self.architecture == "envelope_sparse_mlp":
            if checkpoint.get("weights") != [0.5, 0.5]:
                raise ValueError("unsupported envelope blend weights")
            envelope = checkpoint["envelope"]
            self.support = Support(envelope["support"], 27)
            self.fallback = Architecture(envelope["fallback"], packed)
            base = envelope["envelope"]["base"]
            self.core = finite_array(base["core_coefficients"])
            self.host = finite_array(base["host_coefficients"])
            if self.core.shape != (9,) or self.host.shape != (9,):
                raise ValueError("invalid envelope coefficients")
            self.correction = Component(envelope["envelope"]["correction"], packed)
        # Reuse the existing causal feature validator, independent of model fitting.
        self.extractor = LatencySurface(
            dict(
                schema=2,
                feature_names=FEATURE_NAMES + FORECAST_NAMES,
                target_schema=self.target_schema,
                provenance={},
                trees=[[[-1, -1, -2, 0.0, 1.0]]],
            )
        )
        if packed:
            retain_checkpoint(self, state, {"checkpoint"}, packed=True)

    def state_dict(self):
        return restored_checkpoint(self)

    def input_features(self, data):
        x = self.extractor.input_features(data)
        if x is None:
            return None
        records = data["execution_snapshot"]
        if any(not 0 < r[1] <= r[3] for r in records):
            return None
        prefix = causal = sparse = 0
        for q, c in data["request_shapes"]:
            prefix += q * c
            causal += q * (q + 1) / 2
            n = min(q, max(0, 2048 - c))
            sparse += n * c + n * (n + 1) / 2 + (q - n) * 2048
        lead = (max(r[3] for r in records) - max(r[1] for r in records)) / 1e6
        return np.r_[x, prefix, causal, sparse, lead]

    def predict_features_ms(self, x, native=True):
        if self.excludes_elapsed:
            x = x.copy()
            x[21] = 0
        neural_x = np.r_[x[:23], np.zeros(5), x[23:26]]
        neural = self.neural.predict(neural_x, native)
        if self.architecture in ("guarded_sparse_mlp", "guarded_sparse_mlp_offset"):
            return neural
        if self.support.outside(x):
            envelope = self.fallback.predict(x[:23], native)
        else:
            core = np.r_[1, x[CORE_INDEX] / CORE_SCALE] @ self.core
            host = np.r_[1, x[HOST_INDEX] / HOST_SCALE] @ self.host - x[-1]
            envelope = max(1, float(core), float(host))
            envelope = max(
                1, envelope + self.correction.predict(np.r_[x, envelope], native)
            )
        return (neural + envelope) / 2

    def predict_with_features(self, data):
        x = self.input_features(data)
        if x is None:
            return None, None
        value = self.predict_features_ms(x, data.get("native_predictor", False))
        if not np.isfinite(value):
            raise RuntimeError("nonfinite frozen predictor result")
        return max(1_000_000, int(value * 1e6)), x

    def predict(self, data):
        return self.predict_with_features(data)[0]
