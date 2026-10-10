# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU study of A's causally observed host preparation as a B setup feature."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from experiments.setup_prediction_20261011.train import metrics

ROOT = Path("/mnt/localnvme/jeongyoon")
CAMPAIGN = ROOT / "setup_prediction_20261011"
NAMES = [
    "A_cpu_to_forward_max_ms",
    "A_cpu_to_forward_mean_ms",
    "A_cpu_to_prepared_max_ms",
    "A_cpu_prepared_to_forward_max_ms",
    "A_cpu_observed_fraction",
]


def enrich(rows, live_root=None):
    cells = {}
    enriched = []
    for row in rows:
        root = live_root or ROOT / row["source"]
        key = root, row["workload"], row["arm"], row["repeat"]
        if key not in cells:
            path = root / "live" / row["workload"] / f"r{row['repeat']}_{row['arm']}"
            records = json.loads((path / "late.json").read_text())["records"]
            steps = {}
            for point in records:
                if "step" in point:
                    steps.setdefault(point["step"], []).append(point)
            cells[key] = steps
        points = cells[key][row["step"]]
        forecast = next(p for p in points if p["event"] == "seal_prediction")
        assert forecast["forecast_ns"] <= row["forecast_ns"]
        primary = None
        for point in points:
            if point["time_ns"] > row["forecast_ns"]:
                continue
            if point["event"] == "predictor_submission_progress":
                primary = point["primary_snapshot"]
        spans = []
        if primary is not None:
            for r, e in zip(primary, forecast["execution_snapshot"]):
                if not r or not e or not (0 < r[1] <= r[2] <= e[1]):
                    continue
                assert r[0] == e[0] == forecast["predecessor_step"]
                assert e[1] <= forecast["forecast_ns"]
                spans.append(
                    [(e[1] - r[1]) / 1e6, (r[2] - r[1]) / 1e6, (e[1] - r[2]) / 1e6]
                )
        values = (
            [
                max(s[0] for s in spans),
                np.mean([s[0] for s in spans]),
                max(s[1] for s in spans),
                max(s[2] for s in spans),
                len(spans) / 8,
            ]
            if spans
            else [0] * 5
        )
        base = row.get("features", row.get("inputs"))
        assert base is not None and len(base) == 23
        enriched.append(dict(row, features=list(base) + values))
    return enriched


def score(rows, model, history):
    x = np.asarray([r["features"][: 28 if history else 23] for r in rows], np.float32)
    result = metrics(rows, model.predict(x), "target_ready_ms")
    result["rows"] = len(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate", type=Path)
    args = parser.parse_args()
    if args.evaluate:
        summary = json.loads((args.evaluate / "summary.json").read_text())
        assert summary["complete"] and summary["stopped"]
        rows = json.loads(
            gzip.decompress((args.evaluate / "setup_rows.json.gz").read_bytes())
        )["frames"]
        counts = Counter(
            (r["workload"], r["arm"], r["repeat"], r["step"]) for r in rows
        )
        for row in rows:
            row["weight"] = (
                1 / counts[row["workload"], row["arm"], row["repeat"], row["step"]]
            )
        rows = enrich(rows, args.evaluate)
        models = joblib.load(args.output / "models.joblib")
        scores = {}
        for workload in sorted({r["workload"] for r in rows}):
            part = [r for r in rows if r["workload"] == workload]
            scores[workload] = {
                key: score(part, model, key.startswith("history"))
                for key, model in models.items()
            }
        (args.output / (args.evaluate.name + "_evaluation.json")).write_text(
            json.dumps(scores, indent=2) + "\n"
        )
        print(json.dumps(scores, indent=2), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "design.json").exists()
    (args.output / "design.json").write_text(
        json.dumps(
            dict(
                features=NAMES,
                target="target_ready_ms",
                rule="Only primary snapshots already recorded by forecast time and "
                "the forecast's CPU enqueue timestamps; no A GPU duration/end or "
                "future B membership as input.",
                train="Original cs10/dl10 train; cs10 repeat1 dev for model selection",
                external_holdout="sr11 allocation, unavailable when design frozen",
                scope="CPU feature study; joblib models are not deployed in serving.",
            ),
            indent=2,
        )
        + "\n"
    )
    rows = enrich(
        json.loads(gzip.decompress((CAMPAIGN / "data/pool.json.gz").read_bytes()))
    )
    (args.output / "pool.json.gz").write_bytes(
        gzip.compress(json.dumps(rows).encode(), mtime=0)
    )
    splits = {
        s: [r for r in rows if r["split"] == s] for s in ["train", "dev", "final"]
    }
    models, development = {}, {}
    with threadpool_limits(limits=4):
        for history in [False, True]:
            for family in ["extra", "boost"]:
                key = ("history" if history else "base") + "_" + family
                model = (
                    ExtraTreesRegressor(
                        n_estimators=80,
                        max_depth=12,
                        min_samples_leaf=20,
                        random_state=171,
                        n_jobs=4,
                    )
                    if family == "extra"
                    else HistGradientBoostingRegressor(
                        max_iter=120,
                        max_leaf_nodes=15,
                        min_samples_leaf=30,
                        random_state=171,
                        early_stopping=False,
                    )
                )
                x = np.asarray(
                    [r["features"][: 28 if history else 23] for r in splits["train"]],
                    np.float32,
                )
                model.fit(
                    x,
                    [r["target_ready_ms"] for r in splits["train"]],
                    sample_weight=[r["weight"] for r in splits["train"]],
                )
                models[key] = model
                development[key] = score(splits["dev"], model, history)
        chosen = min(
            (k for k in models if k.startswith("history")),
            key=lambda k: development[k]["mae_ms"],
        )
        (args.output / "selection.json").write_text(
            json.dumps(dict(selected=chosen, dev=development), indent=2) + "\n"
        )
        joblib.dump(models, args.output / "models.joblib")
        final = {
            key: score(splits["final"], model, key.startswith("history"))
            for key, model in models.items()
        }
        (args.output / "final.json").write_text(json.dumps(final, indent=2) + "\n")
        print(json.dumps(dict(dev=development, final=final), indent=2), flush=True)


if __name__ == "__main__":
    main()
