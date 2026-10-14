# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay all frozen predictors at identical issued live forecast snapshots."""

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_live_predictors_20261008")
os.environ["VLLM_DEFERRAL_FAST_FOREST"] = (
    "/mnt/localnvme/glm53_predictor_overnight_20261006/fast_forest/build/"
    "libpredictor_forest_walk.so"
)
for name in ["vllm", "vllm.v1", "vllm.v1.kv_offload"]:
    package = types.ModuleType(name)
    package.__path__ = [str(SOURCE / name.replace(".", "/"))]
    sys.modules[name] = package
from vllm.v1.kv_offload.seal_predictor import ComputeEstimator  # noqa: E402


def main():
    models = {}
    for p in (ROOT / "models").glob("*.json"):
        state = json.loads(p.read_text())
        models[p.stem] = ComputeEstimator.from_state_dict(state, state["context"])
    predictions = []
    with threadpool_limits(limits=1):
        for p in sorted((ROOT / "live").glob("*/r[0-3]_*/result.json")):
            result = json.loads(p.read_text())
            events = [
                json.loads(line)
                for line in (p.parent / "engine.jsonl").read_text().splitlines()
            ]
            done = {
                e["step"]: e
                for e in events
                if e["event"] == "step_completion"
                and e.get("valid")
                and e.get("selected")
            }
            late = json.loads((p.parent / "late.json").read_text())["records"]
            for point in late:
                if (
                    point["event"] != "seal_prediction"
                    or point.get("predictor_input") is None
                ):
                    continue
                data = done.get(point["predecessor_step"])
                if data is None:
                    continue
                ranks = data["execution_ranks"]
                actual_ms = (
                    max(r[3] + r[6] for r in ranks) - max(r[3] for r in ranks)
                ) / 1e6
                snapshot = dict(
                    data,
                    native_predictor=True,
                    execution_snapshot=ranks,
                    forecast_ns=point["forecast_ns"],
                    forecast_load_features=point["predictor_input"][11:21],
                )
                values = {}
                for name, model in models.items():
                    value, features = model.latency_surface.predict_with_features(
                        snapshot
                    )
                    if value is None:
                        raise RuntimeError(
                            f"Model declined a shared eligible point: {name}"
                        )
                    np.testing.assert_allclose(
                        features[:23], point["predictor_input"][:23], rtol=0, atol=1e-8
                    )
                    values[name] = value / 1e6
                predictions.append(
                    dict(
                        cell=str(p.parent),
                        workload=result["workload"],
                        issued_by=result["label"],
                        predecessor_step=data["step"],
                        actual_ms=actual_ms,
                        elapsed_ms=(point["forecast_ns"] - max(r[3] for r in ranks))
                        / 1e6,
                        predictions_ms=values,
                    )
                )
    metrics = []
    for workload in ["familiar60k", "short16k", "heterogeneous"]:
        rows = [r for r in predictions if r["workload"] == workload]
        if not rows:
            continue
        y = np.array([r["actual_ms"] for r in rows])
        for name in models:
            error = np.array([r["predictions_ms"][name] for r in rows]) - y
            metrics.append(
                dict(
                    workload=workload,
                    predictor=name,
                    count=len(rows),
                    mae_ms=float(np.mean(np.abs(error))),
                    bias_ms=float(np.mean(error)),
                    mape_percent=float(np.mean(np.abs(error) / y) * 100),
                    p95_absolute_ms=float(np.quantile(np.abs(error), 0.95)),
                    overpredicted_by_more_than_deadline_lead=int(sum(error > 5)),
                    underpredicted_by_more_than_50ms=int(sum(error < -50)),
                )
            )
    (ROOT / "common_predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in predictions)
    )
    (ROOT / "common_accuracy.json").write_text(
        json.dumps(
            dict(
                metrics=metrics,
                snapshots=len(predictions),
                causal_features=True,
                scope=(
                    "Same snapshots for all predictors, sampled by the union of "
                    "three live policies; final execution records provide labels "
                    "and start-marker-only features."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
