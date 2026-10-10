# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify packed checkpoints on every observed live predictor feature frame."""

import argparse
import copy
import gzip
import hashlib
import json
import os

import numpy as np

from vllm.v1.kv_offload.seal_predictor import ComputeEstimator


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    environment = json.loads((args.root / "server/environment.json").read_text())
    os.environ["VLLM_DEFERRAL_FAST_FOREST"] = environment["VLLM_DEFERRAL_FAST_FOREST"]
    os.environ["VLLM_SEAL_PACKED_CHECKPOINT"] = "0"
    state = json.loads((args.root / "loaded_predictor.json").read_text())
    state["latency_surface"]["checkpoint"]["duration_calibration"] = json.loads(
        args.fit.read_text()
    )["calibration"]
    packed = copy.deepcopy(state)
    packed["latency_surface"]["provenance"]["packed_checkpoint"] = True
    models = [
        ComputeEstimator.from_state_dict(s, s["context"]) for s in (state, packed)
    ]
    for model, original in zip(models, (state, packed)):
        assert model.state_dict(original["context"]) == original
    frames = json.loads(
        gzip.decompress((args.root / "prediction_rows.json.gz").read_bytes())
    )
    maximum = {"numpy": 0.0, "native": 0.0}
    for frame in frames:
        x = np.array(frame["features"], dtype=float)
        for name, native in (("numpy", False), ("native", True)):
            values = [m.latency_surface.predict_features_ms(x, native) for m in models]
            difference = abs(values[0] - values[1])
            maximum[name] = max(maximum[name], difference)
            assert values[0] == values[1]
            for m, value in zip(models, values):
                raw = round(value * 1e6)
                correction = m.latency_surface.duration_calibration.bias_ns(
                    raw, x, 90_000_000
                )
                assert correction == models[
                    0
                ].latency_surface.duration_calibration.bias_ns(raw, x, 90_000_000)
    result = dict(
        frames=len(frames),
        max_absolute_difference_ms=maximum,
        checkpoint_sha256=hashlib.sha256(
            (args.root / "loaded_predictor.json").read_bytes()
        ).hexdigest(),
        scope="Exact NumPy/native prediction equality on observed frames, "
        "exact private state roundtrip, same shape calibration. No serving "
        "throughput or numerical model-output claim.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
