# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online compute duration estimates for experimental late sealing."""

import math
from collections import deque

import numpy as np


class ComputeEstimator:
    """Fit positive costs for tokens, attention work, and request count."""

    def __init__(self):
        self.frozen = False
        self.calibration_mode = "legacy"
        self.latency_surface = None
        self.refinement_surface = None
        self.samples: deque[tuple[np.ndarray, float]] = deque(maxlen=128)
        self.coefficients = None
        self.errors: deque[float] = deque(maxlen=32)
        self.preparation_ns: deque[int] = deque(maxlen=64)
        self.preparation_shapes: deque[tuple[np.ndarray, int]] = deque(maxlen=128)
        self.recent_preparation: deque[tuple[np.ndarray, int, str]] = deque(maxlen=128)
        self.calibration_samples: deque[tuple[np.ndarray, np.ndarray, str, float]] = (
            deque(maxlen=128)
        )

    def observe_preparation(self, records, data=None):
        if self.frozen:
            return
        if not records or any(
            not r
            or len(r) != 9
            or r[0] != records[0][0]
            or r[7] != 1
            or not 0 < r[1] <= r[2]
            for r in records
        ):
            return
        duration = max(r[2] - r[1] for r in records)
        self.preparation_ns.append(duration)
        shape = self.preparation_shape(data) if data else None
        if shape is not None:
            self.preparation_shapes.append((shape, duration))
            self.recent_preparation.append(
                (shape, duration, self.preparation_regime(data))
            )

    @staticmethod
    def preparation_shape(data):
        shapes = data.get("request_shapes", [])
        if not shapes or any(n <= 0 or context < 0 for n, context in shapes):
            return None
        return np.array(
            [
                sum(n for n, _ in shapes),
                sum(context for _, context in shapes),
                len(shapes),
            ],
            dtype=float,
        )

    def preparation_reserve(self, extra_ns, upcoming=None):
        if upcoming is not None:
            query = self.preparation_shape(upcoming)
            if query is None:
                return extra_ns
            if self.preparation_shapes:
                distances = [
                    float(np.linalg.norm(np.log1p(query) - np.log1p(shape)))
                    for shape, _ in self.preparation_shapes
                ]
                order = np.argsort(distances)[:8]
                near = [i for i in order if distances[i] <= 1.5]
                if near:
                    return extra_ns + int(
                        np.quantile([self.preparation_shapes[i][1] for i in near], 0.9)
                    )
                # Extrapolate per-token preparation from the closest shape.
                index = int(order[0])
                shape, duration = self.preparation_shapes[index]
                return extra_ns + int(duration * query[0] / shape[0])
            return extra_ns
        observed = (
            int(np.quantile(self.preparation_ns, 0.9)) if self.preparation_ns else 0
        )
        return observed + extra_ns

    @staticmethod
    def preparation_regime(data):
        contexts = [context > 0 for _, context in data.get("request_shapes", [])]
        return (
            "cached"
            if contexts and all(contexts)
            else ("cold" if not any(contexts) else "mixed")
        )

    def recent_preparation_reserve(self, extra_ns, upcoming):
        """Use recent nearby shapes from the same cold/cache regime."""
        query = self.preparation_shape(upcoming)
        if query is None:
            return extra_ns
        regime = self.preparation_regime(upcoming)
        candidates = [
            (shape, duration)
            for shape, duration, observed_regime in self.recent_preparation
            if observed_regime == regime
        ][-32:]
        distances = [
            float(np.linalg.norm(np.log1p(query) - np.log1p(shape)))
            for shape, _ in candidates
        ]
        near = [i for i in np.argsort(distances)[:8] if distances[i] <= 0.75]
        if not near:
            return extra_ns
        return extra_ns + int(np.quantile([candidates[i][1] for i in near], 0.75))

    @staticmethod
    def features(data):
        shapes = data.get("request_shapes", [])
        if not shapes or any(n <= 0 or context < 0 for n, context in shapes):
            return None
        return np.array(
            [
                1.0,
                sum(n for n, _ in shapes) / 8192,
                sum(n * (context + n / 2) for n, context in shapes) / 100_000_000,
                len(shapes) / 8,
            ]
        )

    def predict(self, data):
        if self.latency_surface is not None:
            surface_data = data
            if data.get("predictor_general_prefill") and not getattr(
                self.latency_surface, "_state", {}
            ).get("provenance", {}).get("general_prefill", False):
                surface_data = dict(data, predictor_general_prefill=False)
            duration = self.latency_surface.predict(surface_data)
            if duration is not None:
                return duration
        features = self.features(data)
        if self.coefficients is None or features is None:
            return None
        duration = float(features @ self.coefficients)
        correction = self.calibration_factor(data)
        return max(1_000_000, int(duration * correction * 1_000_000))

    @staticmethod
    def compute_regime(data):
        shapes = data.get("request_shapes", [])
        if shapes and all(n == 1 for n, _ in shapes):
            return "decode"
        return ComputeEstimator.preparation_regime(data)

    def calibration_factor(self, data):
        """Calibrate raw fitted costs, optionally using nearby execution shapes."""
        if self.calibration_mode == "legacy":
            return (
                max(1.0, float(np.quantile(self.errors, 0.75))) if self.errors else 1.0
            )
        samples = list(self.calibration_samples)
        if not samples or self.coefficients is None:
            return 1.0
        if self.calibration_mode == "shape":
            regime = self.compute_regime(data)
            samples = [s for s in samples if s[2] == regime]
            query = self.preparation_shape(data)
            if query is None or not samples:
                return 1.0
            distances = [
                float(np.linalg.norm(np.log1p(query) - np.log1p(s[1]))) for s in samples
            ]
            near = [i for i in np.argsort(distances)[:8] if distances[i] <= 0.5]
            if len(near) >= 2:
                samples = [samples[i] for i in near]
        ratios = [
            duration / raw
            for features, _, _, duration in samples
            if (raw := float(features @ self.coefficients)) > 0
        ]
        return float(np.quantile(ratios, 0.75)) if ratios else 1.0

    def observe(self, data, execution):
        if self.frozen:
            return
        features = self.features(data)
        if (
            features is None
            or not execution
            or any(
                not r or r[0] != data["step"] or r[5] != 1 or not r[6] or not r[8]
                for r in execution
            )
        ):
            return
        duration_ms = max(r[6] for r in execution) / 1_000_000
        estimate = None
        if self.coefficients is not None:
            correction = (
                max(1.0, float(np.quantile(self.errors, 0.75))) if self.errors else 1.0
            )
            estimate = max(
                1_000_000, int(float(features @ self.coefficients) * correction * 1e6)
            )
        if estimate:
            self.errors.append(duration_ms / (estimate / 1_000_000))
        self.samples.append((features, duration_ms))
        self.calibration_samples.append(
            (
                features,
                self.preparation_shape(data),
                self.compute_regime(data),
                duration_ms,
            )
        )
        if len(self.samples) < 8:
            return
        x = np.vstack([sample[0] for sample in self.samples])
        y = np.array([sample[1] for sample in self.samples])
        coefficients = (
            np.zeros(4) if self.coefficients is None else self.coefficients.copy()
        )
        # Coordinate descent solves nonnegative least squares on four features.
        for _ in range(40):
            for column in range(4):
                residual = y - x @ coefficients + x[:, column] * coefficients[column]
                coefficients[column] = max(
                    0.0,
                    float(x[:, column] @ residual)
                    / max(float(x[:, column] @ x[:, column]), 1e-12),
                )
        self.coefficients = coefficients

    def state_dict(self, context):
        """Export bounded training state as JSON data without runtime flags."""
        if self.coefficients is None:
            raise ValueError("seal estimator has no trained coefficients")
        state = dict(
            schema=1,
            feature_schema="tokens8192_attention1e8_requests8.v1",
            context=context,
            coefficients=self.coefficients.tolist(),
            samples=[[f.tolist(), y] for f, y in self.samples],
            errors=list(self.errors),
            calibration_samples=[
                [f.tolist(), shape.tolist(), regime, y]
                for f, shape, regime, y in self.calibration_samples
            ],
            preparation_ns=list(self.preparation_ns),
            preparation_shapes=[[s.tolist(), n] for s, n in self.preparation_shapes],
            recent_preparation=[
                [s.tolist(), n, regime] for s, n, regime in self.recent_preparation
            ],
        )
        if self.latency_surface is not None:
            state.update(schema=2, latency_surface=self.latency_surface.state_dict())
        if self.refinement_surface is not None:
            if self.latency_surface is None:
                raise ValueError("refinement requires an initial latency surface")
            state.update(
                schema=3, refinement_surface=self.refinement_surface.state_dict()
            )
        return state

    @classmethod
    def from_state_dict(cls, state, context):
        """Validate completely before publishing a restored estimator."""
        keys = {
            "schema",
            "feature_schema",
            "context",
            "coefficients",
            "samples",
            "errors",
            "calibration_samples",
            "preparation_ns",
            "preparation_shapes",
            "recent_preparation",
        }
        if isinstance(state, dict) and state.get("schema") in (2, 3):
            keys.add("latency_surface")
        if isinstance(state, dict) and state.get("schema") == 3:
            keys.add("refinement_surface")
        if (
            not isinstance(state, dict)
            or set(state) != keys
            or type(state["schema"]) is not int
            or state["schema"] not in (1, 2, 3)
            or state["feature_schema"] != "tokens8192_attention1e8_requests8.v1"
            or state["context"] != context
        ):
            raise ValueError("incompatible seal estimator snapshot")

        def number(value, *, positive=False):
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or value < 0 or (positive and value == 0):
                raise ValueError("invalid seal estimator numeric value")
            return value

        def vector(value, size):
            if not isinstance(value, list) or len(value) != size:
                raise ValueError("invalid seal estimator vector")
            return np.array([number(v) for v in value], dtype=float)

        def rows(key, bound, size=None):
            values = state[key]
            if (
                not isinstance(values, list)
                or len(values) > bound
                or (
                    size is not None
                    and any(not isinstance(v, list) or len(v) != size for v in values)
                )
            ):
                raise ValueError("invalid seal estimator history")
            return values

        def duration(value):
            if type(value) is not int:
                raise ValueError("invalid seal estimator preparation duration")
            return number(value)

        def regime(value):
            if value not in ("cached", "cold", "mixed", "decode"):
                raise ValueError("invalid seal estimator regime")
            return value

        estimator = cls()
        estimator.coefficients = vector(state["coefficients"], 4)
        for features, y in rows("samples", 128, 2):
            estimator.samples.append((vector(features, 4), number(y, positive=True)))
        for error in rows("errors", 32):
            estimator.errors.append(number(error, positive=True))
        for features, shape, r, y in rows("calibration_samples", 128, 4):
            estimator.calibration_samples.append(
                (
                    vector(features, 4),
                    vector(shape, 3),
                    regime(r),
                    number(y, positive=True),
                )
            )
        for n in rows("preparation_ns", 64):
            estimator.preparation_ns.append(duration(n))
        for shape, n in rows("preparation_shapes", 128, 2):
            estimator.preparation_shapes.append((vector(shape, 3), duration(n)))
        for shape, n, r in rows("recent_preparation", 128, 3):
            estimator.recent_preparation.append(
                (vector(shape, 3), duration(n), regime(r))
            )
        if state["schema"] in (2, 3):
            from vllm.v1.kv_offload.latency_surface import LatencySurface

            estimator.latency_surface = LatencySurface(state["latency_surface"])
            if state["schema"] == 3:
                estimator.refinement_surface = LatencySurface(
                    state["refinement_surface"]
                )
                model = estimator.refinement_surface
                names = (
                    model.residual.feature_names
                    if model.base is not None
                    else model.feature_names
                )
                if (
                    "cpu_submitted_ranks" not in names
                    or model.target_schema != "critical_end_after_all_rank_start.v1"
                ):
                    raise ValueError("incompatible refinement surface")
        return estimator


def estimator_context(config):
    """Identify the model and execution environment for snapshot compatibility."""
    import torch

    from vllm.platforms import current_platform

    model = config.model_config
    parallel = config.parallel_config
    scheduler = config.scheduler_config
    extra = config.kv_transfer_config.kv_connector_extra_config
    capability = current_platform.get_device_capability()
    if capability is None:
        raise ValueError("invalid seal estimator GPU capability")
    return dict(
        model=model.model,
        revision=model.revision,
        dtype=str(model.dtype),
        quantization=model.quantization,
        kv_dtype=config.cache_config.cache_dtype,
        block_size=config.cache_config.block_size,
        tensor_parallel=parallel.tensor_parallel_size,
        pipeline_parallel=parallel.pipeline_parallel_size,
        expert_parallel=parallel.enable_expert_parallel,
        max_batch_tokens=scheduler.max_num_batched_tokens,
        max_sequences=scheduler.max_num_seqs,
        chunked_prefill=scheduler.enable_chunked_prefill,
        enforce_eager=model.enforce_eager,
        cudagraph_mode=str(config.compilation_config.cudagraph_mode),
        torch_version=str(torch.__version__),
        gpu_name=current_platform.get_device_name(),
        gpu_capability=list(capability),
        async_h2d=extra.get("async_h2d_submission", False),
        h2d_chunk_bytes=extra.get("async_h2d_chunk_bytes", 0),
        execution_schema="cuda_event_start_to_finish.v1",
    )
