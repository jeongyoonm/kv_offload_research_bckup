# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check reconstruction timestamps against saved feature-capture timestamps."""

import json
from collections import Counter
from functools import cache
from pathlib import Path

import joblib
import numpy as np

ROOT = Path("/mnt/localnvme/glm53_offset_predictor_20261008")


@cache
def forecasts(source):
    path = Path(source)
    engine = [json.loads(line) for line in path.read_text().splitlines()]
    late = path.with_name("late.json")
    records = json.loads(late.read_text()).get("records", []) if late.exists() else []
    result = {}
    for event in engine + records:
        if event["event"] in ("dataset_forecast", "seal_prediction"):
            for timestamp in {event["time_ns"], event.get("forecast_ns")} - {None}:
                result[(event["predecessor_step"], timestamp)] = event
    return result


def stats(values):
    return (
        dict(
            count=len(values),
            median=float(np.median(values)),
            p95=float(np.quantile(values, 0.95)),
            maximum=float(np.max(values)),
        )
        if values
        else None
    )


def main():
    base = Path("/mnt/localnvme/glm53_predictor_expanded_20261007")
    data = {
        "training": joblib.load(base / "pool.joblib")["forecasts"],
        "benchmark": joblib.load(base / "locked_benchmark.joblib"),
    }
    summaries = {}
    for source, rows in data.items():
        kinds = Counter()
        timestamp_delta, input_delta, count_delta = [], [], []
        for row in rows:
            point = forecasts(row["source_file"])[
                (row["step"], row["forecast_time_ns"])
            ]
            kinds[point["event"]] += 1
            if "forecast_ns" in point:
                timestamp_delta.append((point["time_ns"] - point["forecast_ns"]) / 1e6)
                kinds[
                    "capture_timestamp"
                    if point["forecast_ns"] == row["forecast_time_ns"]
                    else "log_timestamp_with_known_capture"
                ] += 1
            if point.get("predictor_input") is not None:
                original = point["predictor_input"][11:23]
                if len(original) == len(row["forecast_features"]):
                    input_delta.append(
                        max(
                            abs(a - b)
                            for a, b in zip(original, row["forecast_features"])
                        )
                    )
                    count_delta.append(abs(original[0] - row["forecast_features"][0]))
        summaries[source] = dict(
            rows=len(rows),
            event_types=dict(kinds),
            known_capture_to_record_lag_ms=stats(timestamp_delta),
            raw_feature_max_abs_difference=stats(input_delta),
            raw_load_count_difference=stats(count_delta),
        )
    report = dict(
        sources=summaries,
        limitation=(
            "Older seal events often omit forecast_ns and raw native features, "
            "so their evaluation lag cannot be recovered exactly. Shadow "
            "dataset events record the capture time directly and do not evaluate "
            "a model. Feature maximum differences combine different units."
        ),
    )
    (ROOT / "timestamp_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
