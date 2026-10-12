# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short frozen-predictor audit, excluded from trace throughput comparisons."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.agentic_1m_20261012.common import DATA, save
from experiments.agentic_1m_20261012.prepare import Writer, finish
from experiments.agentic_1m_20261012.replay import cell


def prepare():
    name = "predictor_audit"
    if (DATA / (name + ".json")).exists():
        return name, prepare_prefixes(name)
    writer = Writer(name)
    rng = np.random.default_rng(20261015)
    sessions = []
    for i in range(16):
        prefix = (65536, 98304, 131072, 196608)[i % 4]
        suffix = (128, 512, 2048, 4096)[i // 4]
        tokens = rng.integers(128, 31700, prefix + suffix, dtype=np.uint32)
        row = writer.row(tokens, 16, session=i, index=0)
        sessions.append(dict(session=i, requests=[row]))
    finish(
        name,
        sessions,
        writer,
        adapter=(
            "Disjoint synthetic diagnostic,16 output tokens. Prime full CPU KV "
            "then reset GPU only; not a trace performance cell or training data."
        ),
    )
    return name, prepare_prefixes(name)


def prepare_prefixes(name):
    """Populate only the prefix, leaving real uncached work for restoration."""
    prefix_name = name + "_prefixes"
    if (DATA / (prefix_name + ".json")).exists():
        return prefix_name
    data = json.loads((DATA / (name + ".json")).read_text())
    blob = np.memmap(data["token_file"], dtype="<u4", mode="r")
    writer = Writer(prefix_name)
    sessions = []
    for i, session in enumerate(data["sessions"]):
        prefix = (65536, 98304, 131072, 196608)[i % 4]
        offset = session["requests"][0]["offset"]
        row = writer.row(blob[offset : offset + prefix], 16, session=i, index=0)
        sessions.append(dict(session=i, requests=[row]))
    finish(
        prefix_name,
        sessions,
        writer,
        adapter="Only the cached prefix of the disjoint predictor diagnostic.",
    )
    return prefix_name


def analyze(directory):
    records = json.loads((directory / "prediction_records.json").read_text())
    events = [json.loads(s) for s in (directory / "engine.jsonl").open()]
    completed = {
        e["step"]: e
        for e in events
        if e["event"] == "step_completion" and e.get("selected")
    }
    rows, missing = [], Counter()
    for point in records:
        if point["event"] != "seal_prediction":
            continue
        previous = completed.get(point["predecessor_step"])
        if previous is None or not previous.get("execution_ranks"):
            missing["preceding_execution"] += 1
            continue
        ranks = previous["execution_ranks"]
        if len(ranks) != 8 or any(not r[3] or not r[6] for r in ranks):
            missing["incomplete_execution"] += 1
            continue
        actual_end = max(r[3] + r[6] for r in ranks)
        predicted_end = point["gpu_start_upper_ns"] + point["predicted_duration_ns"]
        error = (predicted_end - actual_end) / 1e6
        rows.append(
            dict(
                step=point["step"],
                preceding_step=point["predecessor_step"],
                forecast_ns=point["forecast_ns"],
                error_ms=error,
                predicted_end_ns=predicted_end,
                actual_end_ns=actual_end,
                actual_remaining_ms=(actual_end - point["forecast_ns"]) / 1e6,
                predictor_input=point.get("predictor_input"),
            )
        )
    absolute = [abs(r["error_ms"]) for r in rows]
    save(directory / "forecast_audit_rows.json", rows)
    summary = dict(
        forecasts=len(rows),
        missing=dict(missing),
        mae_ms=float(np.mean(absolute)) if absolute else None,
        p95_absolute_error_ms=float(np.quantile(absolute, 0.95)) if absolute else None,
        mean_signed_error_ms=float(np.mean([r["error_ms"] for r in rows]))
        if rows
        else None,
        target=(
            "A end=max(all-rank observed start+elapsed); predicted end uses the "
            "live anchor+predicted duration. Positive error predicts too late."
        ),
        scope=(
            "Short, instrumented synthetic CPU-restoration diagnostic; not "
            "representative trace accuracy, model retraining, or throughput."
        ),
    )
    save(directory / "forecast_audit.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    name, prefix_name = prepare()
    spec = dict(
        id="predictor_audit_prime",
        dataset=prefix_name,
        concurrency=8,
        measurement="diagnostic",
    )
    assert cell(args.root, "full", spec)["accepted"]
    restore = dict(
        spec, id="predictor_audit_restore", dataset=name, reset_external=False
    )
    assert cell(args.root, "full", restore)["accepted"]
    analyze(args.root / "cells" / restore["id"])


if __name__ == "__main__":
    main()
