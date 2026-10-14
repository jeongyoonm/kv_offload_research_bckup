# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare prediction plus trace inputs with the equivalent single pass."""

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
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["VLLM_DEFERRAL_FAST_FOREST"] = str(args.library)
    rows = [json.loads(s) for s in args.corpus.read_text().splitlines()]
    reports = []
    for path in args.models:
        model = production.LatencySurface(
            json.loads(path.read_text())["state"]["latency_surface"]
        )
        before = model.state_dict()
        for native in [False, True]:
            costs = {"separate": [], "single": []}
            error, features_equal = [], True
            for index in np.random.default_rng(17).permutation(len(rows)):
                r = rows[index]
                data = dict(
                    r,
                    execution_snapshot=r["execution_ranks"],
                    forecast_ns=r["forecast_time_ns"],
                    forecast_load_features=r["forecast_features"][:10],
                    native_predictor=native,
                )
                values = {}
                order = ["separate", "single"] if index % 2 else ["single", "separate"]
                for name in order:
                    started = time.perf_counter_ns()
                    values[name] = (
                        (model.predict(data), model.input_features(data))
                        if name == "separate"
                        else model.predict_with_features(data)
                    )
                    costs[name].append(time.perf_counter_ns() - started)
                separate, single = values["separate"], values["single"]
                if single[0] is None or separate[0] is None:
                    raise ValueError("verification corpus contains unsupported inputs")
                error.append(abs(single[0] - separate[0]))
                features_equal &= np.array_equal(single[1], separate[1])
            assert max(error) == 0 and features_equal
            assert before == model.state_dict()
            reports.append(
                dict(
                    model=str(path),
                    model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    backend="native" if native else "numpy",
                    samples=len(rows),
                    max_prediction_difference_ns=max(error),
                    trace_features_exactly_equal=features_equal,
                    state_unchanged=True,
                    latency_us={
                        name: dict(
                            mean=float(np.mean(v) / 1000),
                            median=float(np.median(v) / 1000),
                            p95=float(np.quantile(v, 0.95) / 1000),
                        )
                        for name, v in costs.items()
                    },
                )
            )
            print(json.dumps(reports[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
