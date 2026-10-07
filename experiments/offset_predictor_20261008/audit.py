# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit live inference, labels, decision timing, and offline coverage on CPU."""

import argparse
import hashlib
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VLLM_DEFERRAL_FAST_FOREST"] = (
    "/mnt/localnvme/glm53_predictor_overnight_20261006/fast_forest/build/"
    "libpredictor_forest_walk.so"
)
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).resolve().parents[2]
for name in ["vllm", "vllm.v1", "vllm.v1.kv_offload"]:
    package = types.ModuleType(name)
    package.__path__ = [str(SOURCE / name.replace(".", "/"))]
    sys.modules[name] = package
from vllm.v1.kv_offload.seal_predictor import ComputeEstimator  # noqa: E402


def stats(values):
    if not values:
        return None
    return dict(
        count=len(values),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        maximum=float(np.max(values)),
    )


def metric(y, p):
    e = np.array(p) - np.array(y)
    return dict(
        samples=len(e),
        mae_ms=float(np.mean(np.abs(e))),
        bias_ms=float(np.mean(e)),
        mape_percent=float(np.mean(np.abs(e) / y) * 100),
        p95_ms=float(np.quantile(np.abs(e), 0.95)),
    )


def signature(row):
    return tuple(sorted(tuple(qc) for qc in row["request_shapes"]))


def snapshot(row, forecast, loads, records=None):
    return dict(
        row,
        predictor_general_prefill=True,
        prefill_only=True,
        native_predictor=True,
        execution_snapshot=records if records is not None else row["execution_ranks"],
        forecast_ns=forecast,
        forecast_load_features=list(loads),
    )


def target(row):
    ranks = row["execution_ranks"]
    return (max(r[3] + r[6] for r in ranks) - max(r[3] for r in ranks)) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = {}
    for path in (args.root / "models").glob("*.json"):
        state = json.loads(path.read_text())
        models[path.stem] = ComputeEstimator.from_state_dict(state, state["context"])
    train = joblib.load("/mnt/localnvme/glm53_predictor_expanded_20261007/pool.joblib")[
        "forecasts"
    ]
    benchmark = joblib.load(
        "/mnt/localnvme/glm53_predictor_expanded_20261007/locked_benchmark.joblib"
    )
    geometries = defaultdict(list)
    for row in train:
        geometries[signature(row)].append(target(row))
    forecasts, checks = [], []
    with threadpool_limits(limits=1):
        for path in sorted((args.root / "live").glob("*/r[0-3]_*/result.json")):
            result = json.loads(path.read_text())
            state = json.loads(
                (args.root / "models" / f"{result['label']}.json").read_text()
            )
            assert (
                result["checkpoint_sha256"]
                == hashlib.sha256(
                    json.dumps(state, sort_keys=True).encode()
                ).hexdigest()
            )
            events = [
                json.loads(line)
                for line in (path.parent / "engine.jsonl").read_text().splitlines()
            ]
            done = {
                e["step"]: e
                for e in events
                if e["event"] == "step_completion"
                and e.get("valid")
                and e.get("selected")
            }
            late = json.loads((path.parent / "late.json").read_text())["records"]
            for point in late:
                if (
                    point["event"] != "seal_prediction"
                    or point.get("predictor_input") is None
                ):
                    continue
                row = done[point["predecessor_step"]]
                ranks = row["execution_ranks"]
                observed = point.get("execution_snapshot", ranks)
                data = snapshot(
                    row, point["forecast_ns"], point["predictor_input"][11:21], observed
                )
                issued_model = models[result["label"]].latency_surface
                predicted, features = issued_model.predict_with_features(data)
                assert predicted is not None
                delta = abs(predicted - point["predicted_duration_ns"])
                feature_delta = float(
                    np.max(np.abs(np.array(features) - point["predictor_input"]))
                )
                assert delta <= 2, (path, point, delta)
                assert feature_delta < 1e-8, (path, feature_delta)
                start = max(r[3] for r in ranks)
                actual = target(row)
                assert point["gpu_start_upper_ns"] == start
                assert (
                    point["deadline_ns"]
                    == start
                    + point["predicted_duration_ns"]
                    - point["preparation_reserve_ns"]
                )
                predictions = {}
                sensitivity = {}
                for name, estimator in models.items():
                    model = estimator.latency_surface
                    value = model.predict(data)
                    assert value is not None
                    predictions[name] = value / 1e6
                    # Hold every load/geometry feature fixed; vary only query time.
                    alternatives = [
                        model.predict(dict(data, forecast_ns=start + int(ms * 1e6)))
                        / 1e6
                        for ms in [0, 5, 25, 50, 100]
                    ]
                    sensitivity[name] = dict(
                        fixed_state_prediction_range_ms=max(alternatives)
                        - min(alternatives),
                        at_zero_elapsed_ms=alternatives[0],
                        at_100_elapsed_ms=alternatives[-1],
                    )
                matches = geometries.get(signature(row), [])
                forecasts.append(
                    dict(
                        workload=result["workload"],
                        cell=str(path.parent),
                        issued_by=result["label"],
                        predecessor_step=row["step"],
                        actual_ms=actual,
                        elapsed_ms=(point["forecast_ns"] - start) / 1e6,
                        queue_ms=(start - row["dispatch_ns"]) / 1e6,
                        current_load_count=point["predictor_input"][11],
                        current_load_tokens=point["predictor_input"][12],
                        prediction_cost_us=point["prediction_cost_ns"] / 1e3,
                        predictions_ms=predictions,
                        sensitivity=sensitivity,
                        same_geometry_training_count=len(matches),
                        same_geometry_training_duration_ms=stats(matches),
                        label_start_uncertainty_ms=max(r[3] - r[2] for r in ranks)
                        / 1e6,
                        label_end_observation_window_ms=(
                            max(r[8] for r in ranks) - max(r[7] for r in ranks)
                        )
                        / 1e6,
                    )
                )
                checks.append(
                    dict(replay_difference_ns=delta, feature_difference=feature_delta)
                )
        offline = []
        offline_rows = []
        for name, estimator in models.items():
            y, p = [], []
            for row in benchmark:
                data = snapshot(
                    row, row["forecast_time_ns"], row["forecast_features"][:10]
                )
                v = estimator.latency_surface.predict(data)
                assert v is not None
                y.append(target(row))
                p.append(v / 1e6)
                offline_rows.append(
                    dict(
                        predictor=name,
                        actual_ms=y[-1],
                        prediction_ms=p[-1],
                        episode=row["episode"],
                        workload=row["workload"],
                        current_load_count=row["forecast_features"][0],
                    )
                )
            offline.append(
                dict(
                    predictor=name,
                    metrics=metric(y, p),
                    scope="Reused 192-row benchmark, not an independent new test",
                )
            )
    summaries = []
    for workload in sorted({r["workload"] for r in forecasts}):
        rows = [r for r in forecasts if r["workload"] == workload]
        summaries.append(
            dict(
                workload=workload,
                count=len(rows),
                timing={
                    k: stats([r[k] for r in rows])
                    for k in [
                        "actual_ms",
                        "elapsed_ms",
                        "queue_ms",
                        "prediction_cost_us",
                        "label_start_uncertainty_ms",
                        "label_end_observation_window_ms",
                    ]
                },
                exact_geometry_covered=sum(
                    r["same_geometry_training_count"] > 0 for r in rows
                ),
                models={
                    name: dict(
                        accuracy=metric(
                            [r["actual_ms"] for r in rows],
                            [r["predictions_ms"][name] for r in rows],
                        ),
                        zero_elapsed_counterfactual_accuracy=metric(
                            [r["actual_ms"] for r in rows],
                            [
                                r["sensitivity"][name]["at_zero_elapsed_ms"]
                                for r in rows
                            ],
                        ),
                        elapsed_sensitivity=stats(
                            [
                                r["sensitivity"][name][
                                    "fixed_state_prediction_range_ms"
                                ]
                                for r in rows
                            ]
                        ),
                        matched_geometry_accuracy=metric(
                            [
                                r["actual_ms"]
                                for r in rows
                                if r["same_geometry_training_count"]
                            ],
                            [
                                r["predictions_ms"][name]
                                for r in rows
                                if r["same_geometry_training_count"]
                            ],
                        )
                        if any(r["same_geometry_training_count"] for r in rows)
                        else None,
                    )
                    for name in models
                },
            )
        )

    def distribution(rows):
        return {
            key: stats(values)
            for key, values in dict(
                duration_ms=[target(r) for r in rows],
                elapsed_ms=[
                    (r["forecast_time_ns"] - max(a[3] for a in r["execution_ranks"]))
                    / 1e6
                    for r in rows
                ],
                queue_ms=[
                    (max(a[3] for a in r["execution_ranks"]) - r["dispatch_ns"]) / 1e6
                    for r in rows
                ],
            ).items()
        }

    report = dict(
        source=str(args.root),
        forecasts=len(forecasts),
        issued_replay_max_difference_ns=max(c["replay_difference_ns"] for c in checks),
        issued_feature_max_difference=max(c["feature_difference"] for c in checks),
        issued_label_and_deadline_checks=True,
        offline_benchmark=offline,
        source_distributions={
            name: distribution(rows)
            for name, rows in [("training", train), ("reused_benchmark", benchmark)]
        },
        live=summaries,
        limits=(
            "Sensitivity changes one feature with loads held fixed; it is not "
            "a causal intervention on GPU execution. Final records provide labels; "
            "start/enqueue fields only are used as predictor inputs."
        ),
    )
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "forecasts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in forecasts)
    )
    (args.output / "offline_predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in offline_rows)
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
