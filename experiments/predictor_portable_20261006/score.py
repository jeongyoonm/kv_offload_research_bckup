# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score frozen predictors on identical causal forecasts and physical end labels."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "experiments/predictor_corpus_20261006"))
from corpus import metrics  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "latency_surface", SOURCE / "vllm/v1/kv_offload/latency_surface.py"
)
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    rows = [
        r
        for r in rows
        if all(
            0 < e[2] <= e[3] <= r["forecast_time_ns"] and e[3] - e[2] <= 2_000_000
            for e in r["execution_ranks"]
        )
    ]
    targets, raw, prior = [], [], []
    for r in rows:
        ranks = r["execution_ranks"]
        anchor = max(e[3] for e in ranks)
        targets.append(max(e[3] + e[6] for e in ranks) / 1e6 - anchor / 1e6)
        raw.append(r["duration_ms"])
        prior.append(r["prior_prediction_ms"])
    targets, raw, prior = map(np.asarray, (targets, raw, prior))
    forecasts = []
    for row in rows:
        data = dict(row)
        data.update(
            execution_snapshot=row["execution_ranks"],
            forecast_ns=row["forecast_time_ns"],
            forecast_load_features=row["forecast_features"][:10],
        )
        forecasts.append(data)
    workloads = np.array([r["workload"] for r in rows])
    predictions = {"issued_prior": prior}
    for path in args.models:
        state = json.loads(path.read_text())["state"]["latency_surface"]
        model = production.LatencySurface(state)
        predictions[path.stem] = np.array([model.predict(r) / 1e6 for r in forecasts])
    report = {
        "samples": len(rows),
        "episodes": len({r["episode"] for r in rows}),
        "target": "max_r(start_upper_r + CUDA_span_r) - max_r(start_upper_r)",
        "models": {},
    }
    for name, prediction in predictions.items():
        report["models"][name] = dict(
            physical_end=metrics(prediction - targets),
            cuda_span=metrics(prediction - raw),
            per_workload={
                w: metrics((prediction - targets)[workloads == w])
                for w in sorted(set(workloads))
            },
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
