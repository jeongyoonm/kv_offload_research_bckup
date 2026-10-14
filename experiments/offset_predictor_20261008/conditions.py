# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare host submission and transfer conditions in offline and live labels."""

import argparse
import json
from collections import defaultdict
from functools import cache
from pathlib import Path

import audit
import joblib
import numpy as np
from scipy.stats import spearmanr


@cache
def events(path):
    return {
        e["step"]: e
        for line in Path(path).read_text().splitlines()
        if (e := json.loads(line))["event"] == "step_completion"
        and e.get("valid")
        and e.get("selected")
    }


def conditions(row, forecast, loads):
    ranks = row["execution_ranks"]
    start = max(r[3] for r in ranks)
    primary = row.get("ranks", [])
    submit = None
    if len(primary) == 8 and all(
        len(r) == 9 and r[7] == 1 and 0 < r[1] <= r[2] <= r[3] for r in primary
    ):
        submit = max(r[3] for r in primary)
    return dict(
        actual_ms=audit.target(row),
        elapsed_ms=(forecast - start) / 1e6,
        queue_ms=(start - row["dispatch_ns"]) / 1e6,
        marker_enqueue_headstart_ms=(start - max(r[1] for r in ranks)) / 1e6,
        current_load_count=loads[0],
        current_load_tokens=loads[1],
        load_max_age_ms=loads[3],
        completed_old_tokens=loads[9],
        cpu_submit_end_after_gpu_start_ms=(submit - start) / 1e6 if submit else None,
        cpu_submit_remaining_at_forecast_ms=max(0, submit - forecast) / 1e6
        if submit
        else None,
    )


def summarize(rows):
    return {k: audit.stats([r[k] for r in rows if r[k] is not None]) for k in rows[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = joblib.load("/mnt/localnvme/glm53_predictor_expanded_20261007/pool.joblib")
    benchmark = joblib.load(
        "/mnt/localnvme/glm53_predictor_expanded_20261007/locked_benchmark.joblib"
    )
    offline = defaultdict(list)
    geometry = defaultdict(list)
    for source, rows in [("training", pool["forecasts"]), ("benchmark", benchmark)]:
        for row in rows:
            data = events(row["source_file"]).get(row["step"])
            if data is None:
                data = row
            record = conditions(data, row["forecast_time_ns"], row["forecast_features"])
            offline[source].append(record)
            if source == "training":
                geometry[audit.signature(row)].append(record)
    live = defaultdict(list)
    joined = []
    forecasts = [
        json.loads(line)
        for line in (args.output / "forecasts.jsonl").read_text().splitlines()
    ]
    for record in forecasts:
        data = events(str(Path(record["cell"]) / "engine.jsonl"))[
            record["predecessor_step"]
        ]
        late = json.loads((Path(record["cell"]) / "late.json").read_text())["records"]
        point = next(
            p
            for p in late
            if p["event"] == "seal_prediction"
            and p["predecessor_step"] == record["predecessor_step"]
        )
        info = conditions(data, point["forecast_ns"], point["predictor_input"][11:21])
        live[record["workload"]].append(info)
        matched = geometry.get(audit.signature(data), [])
        joined.append(
            dict(
                record,
                conditions=info,
                matched_training_conditions=summarize(matched) if matched else None,
            )
        )
    analyses = []
    for workload in sorted(live):
        rows = [r for r in joined if r["workload"] == workload]
        correlations = {}
        for model in rows[0]["predictions_ms"]:
            error = np.array(
                [r["predictions_ms"][model] - r["actual_ms"] for r in rows]
            )
            correlations[model] = {}
            for key in rows[0]["conditions"]:
                x = np.array(
                    [
                        r["conditions"][key]
                        if r["conditions"][key] is not None
                        else np.nan
                        for r in rows
                    ]
                )
                mask = np.isfinite(x)
                if mask.sum() >= 5 and np.std(x[mask]) > 0:
                    correlation = spearmanr(x[mask], error[mask])
                    correlations[model][key] = dict(
                        rho=float(correlation.statistic),
                        pvalue=float(correlation.pvalue),
                        count=int(mask.sum()),
                    )
        analyses.append(
            dict(
                workload=workload,
                distribution=summarize(live[workload]),
                signed_error_spearman=correlations,
            )
        )
    result = dict(
        offline={source: summarize(rows) for source, rows in offline.items()},
        live=analyses,
        label_scope=(
            "CUDA event span includes gaps during host submission and joins "
            "required output copies; it is not isolated kernel runtime."
        ),
        scope=(
            "CPU submission end is a diagnostic label and is never passed to "
            "the predictor when it lies after the forecast."
        ),
        causal_limit=(
            "Correlations and matched-geometry comparisons do not isolate "
            "causal contributions."
        ),
    )
    (args.output / "conditions.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "conditions_forecasts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in joined)
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
