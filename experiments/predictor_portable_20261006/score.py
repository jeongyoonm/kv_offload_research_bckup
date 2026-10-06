# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score frozen predictors on identical causal forecasts and physical end labels."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "experiments/predictor_corpus_20261006"))
from corpus import metrics  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "latency_surface", SOURCE / "vllm/v1/kv_offload/latency_surface.py"
)
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)
spec_estimator = importlib.util.spec_from_file_location(
    "seal_predictor", SOURCE / "vllm/v1/kv_offload/seal_predictor.py"
)
estimation = importlib.util.module_from_spec(spec_estimator)
spec_estimator.loader.exec_module(estimation)


def overlap(state, rows):
    provenance = (state.get("latency_surface") or {}).get("provenance", {})
    corpora = dict(provenance.get("corpora", {}))
    if "corpus" in provenance:
        corpora[provenance["corpus"]] = provenance["corpus_sha256"]
    training = set()
    for name, digest in corpora.items():
        path = Path(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"training corpus changed: {path}")
        training.update(
            (r["source_file"], r["step"])
            for line in path.read_text().splitlines()
            if (r := json.loads(line))
        )
    return len(training.intersection((r["source_file"], r["step"]) for r in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-training-overlap", action="store_true")
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
    if not rows:
        raise ValueError("evaluation corpus contains no valid all-rank forecasts")
    targets, raw, prior = [], [], []
    for r in rows:
        ranks = r["execution_ranks"]
        anchor = max(e[3] for e in ranks)
        targets.append((max(e[3] + e[6] for e in ranks) - anchor) / 1e6)
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
    provenance = {}
    for path in args.models:
        state = json.loads(path.read_text())["state"]
        training_overlap = overlap(state, rows)
        if training_overlap and not args.allow_training_overlap:
            raise ValueError(
                f"{path}: {training_overlap} evaluation batches used in training"
            )
        legacy_state = dict(state)
        surface = legacy_state.pop("latency_surface", None)
        legacy_state["schema"] = 1
        model = estimation.ComputeEstimator.from_state_dict(
            legacy_state, state["context"]
        )
        if surface is not None:
            model.latency_surface = production.LatencySurface(surface)
        model.frozen = True
        name = f"{path.parent.name}/{path.stem}"
        if name in predictions:
            raise ValueError(f"duplicate model label: {name}")
        predictions[name] = np.array([model.predict(r) / 1e6 for r in forecasts])
        provenance[name] = dict(
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            training=(state.get("latency_surface") or {}).get("provenance", {}),
            training_overlap=training_overlap,
        )
    report = {
        "samples": len(rows),
        "episodes": len({r["episode"] for r in rows}),
        "target": "max_r(start_upper_r + CUDA_span_r) - max_r(start_upper_r)",
        "models": {},
        "model_provenance": provenance,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
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
