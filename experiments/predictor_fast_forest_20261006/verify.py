# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify exact frozen-model predictions and benchmark both CPU evaluators."""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "surface", SOURCE / "vllm/v1/kv_offload/latency_surface.py"
)
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.corpus.read_text().splitlines()]
    reports = []
    for path in args.models:
        state = json.loads(path.read_text())["state"]["latency_surface"]
        os.environ.pop("VLLM_DEFERRAL_FAST_FOREST", None)
        ordinary = production.LatencySurface(state)
        os.environ["VLLM_DEFERRAL_FAST_FOREST"] = str(args.library)
        native = production.LatencySurface(state)
        difference, cost = [], {"numpy": [], "native": []}
        for index in np.random.default_rng(17).permutation(len(rows)):
            row = rows[index]
            data = dict(
                row,
                execution_snapshot=row["execution_ranks"],
                forecast_ns=row["forecast_time_ns"],
                forecast_load_features=row["forecast_features"][:10],
            )
            values = []
            for name, model in [("numpy", ordinary), ("native", native)]:
                start = time.perf_counter_ns()
                value = model.predict(dict(data, native_predictor=name == "native"))
                cost[name].append(time.perf_counter_ns() - start)
                if value is None:
                    raise ValueError("all equivalence queries must be supported")
                values.append(value)
            difference.append(abs(values[0] - values[1]))
        assert max(difference) <= 3, max(difference)
        reports.append(
            dict(
                model=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                samples=len(rows),
                max_prediction_difference_ns=max(difference),
                model_state_unchanged=native.state_dict() == ordinary.state_dict(),
                latency_us={
                    name: dict(
                        mean=float(np.mean(v) / 1000),
                        median=float(np.median(v) / 1000),
                        p95=float(np.quantile(v, 0.95) / 1000),
                        p99=float(np.quantile(v, 0.99) / 1000),
                        maximum=float(max(v) / 1000),
                    )
                    for name, v in cost.items()
                },
            )
        )
        print(json.dumps(reports[-1], indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                scope="CPU-only exact evaluator comparison; no GPU benefit claim",
                reports=reports,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
