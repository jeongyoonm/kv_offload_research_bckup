# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check frozen exports and measure inference cost before serving tests."""

import argparse
import gc
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from vllm.v1.kv_offload.seal_predictor import ComputeEstimator


def features(row):
    from vllm.v1.kv_offload.latency_surface import batch_features

    prefix = causal = sparse = 0
    for q, c in row["request_shapes"]:
        prefix += q * c
        causal += q * (q + 1) / 2
        n = min(q, max(0, 2048 - c))
        sparse += n * c + n * (n + 1) / 2 + (q - n) * 2048
    ranks = row["execution_ranks"]
    lead = (max(r[3] for r in ranks) - max(r[1] for r in ranks)) / 1e6
    return np.r_[
        batch_features(row), row["forecast_features"], prefix, causal, sparse, lead
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--state", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bias-us", action="append", default=[], metavar="NAME=US")
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve previous proof artifacts"
    pool = json.loads(gzip.decompress(args.pool.read_bytes()))
    x = np.stack([features(row) for row in pool["forecasts"]])
    biases = {
        name: int(value) / 1000
        for name, value in (item.split("=", 1) for item in args.bias_us)
    }
    results = {}
    for item in args.state:
        name, path = item.split("=", 1)
        state = json.loads(Path(path).read_text())
        gc.collect()
        before = len(gc.get_objects())
        estimator = ComputeEstimator.from_state_dict(state, state["context"])
        assert estimator.state_dict(state["context"]) == state
        model = estimator.latency_surface
        tracked = len(gc.get_objects()) - before
        reference = np.array([model.predict_features_ms(v, native=False) for v in x])
        native = np.array([model.predict_features_ms(v, native=True) for v in x])
        error = float(np.max(np.abs(reference - native)))
        assert error < 2e-6, dict(name=name, native_error_ms=error)
        changed = x.copy()
        changed[:, 21] += 1000
        elapsed = np.array([model.predict_features_ms(v, native=True) for v in changed])
        assert np.array_equal(native, elapsed), "Elapsed must remain an external offset"
        subset = x[np.linspace(0, len(x) - 1, min(256, len(x)), dtype=int)]
        for v in subset:
            model.predict_features_ms(v, native=True)
        timings = []
        enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(5):
                started = time.perf_counter_ns()
                for v in subset:
                    model.predict_features_ms(v, native=True)
                timings.append((time.perf_counter_ns() - started) / len(subset) / 1000)
        finally:
            if enabled:
                gc.enable()
        truth = np.array(
            [
                (
                    max(r[3] + r[6] for r in row["execution_ranks"])
                    - max(r[3] for r in row["execution_ranks"])
                )
                / 1e6
                for row in pool["forecasts"]
            ]
        )
        holdout = np.array([row.get("split") == "final" for row in pool["forecasts"]])
        neural_fallback = np.array(
            [
                model.neural.support.outside(np.r_[v[:23], np.zeros(5), v[23:26]])
                for v in x
            ]
        )
        envelope_fallback = np.array(
            [
                model.support.outside(v) if hasattr(model, "support") else False
                for v in x
            ]
        )
        results[name] = dict(
            path=path,
            sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            frames=len(x),
            max_native_error_ms=error,
            elapsed_invariant=True,
            roundtrip_exact=True,
            backend=model.backend,
            tracked_objects_added=tracked,
            inference_us=timings,
            median_inference_us=float(np.median(timings)),
            timing_scope="Warm native predict_features_ms, feature vector built; "
            "GC disabled only during this isolated CPU timing, serving GC unchanged",
            final_samples=int(holdout.sum()),
            neural_guard_fallback_frames=int(neural_fallback.sum()),
            final_neural_guard_fallback_frames=int(neural_fallback[holdout].sum()),
            envelope_guard_fallback_frames=int(envelope_fallback.sum()),
            final_envelope_guard_fallback_frames=int(envelope_fallback[holdout].sum()),
            applied_target_bias_ms=biases.get(name, 0),
            final_mae_ms=float(
                np.mean(
                    np.abs(
                        np.maximum(1, native[holdout] - biases.get(name, 0))
                        - truth[holdout]
                    )
                )
            )
            if holdout.any()
            else None,
        )
        print(json.dumps(results[name]), flush=True)
        del estimator, model
        gc.collect()
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
