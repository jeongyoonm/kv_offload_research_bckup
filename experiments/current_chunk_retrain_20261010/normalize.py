# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply the historical measurement guards to fresh controlled cohorts."""

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    allocation = json.loads((root / "allocation.json").read_text())
    assert allocation["server_exit_code"] is not None, "Stop server before extraction"
    sys.path.insert(0, str(SOURCE / "experiments/dataset_coverage_20261007"))
    audit = load("audit", "experiments/dataset_coverage_20261007/audit.py")
    audit.ROOT = root
    audit.audit()
    corpus = load("corpus", "experiments/predictor_corpus_20261006/corpus.py")
    rows, rejected = [], Counter()
    for path in sorted((root / "cells").rglob("result.json")):
        meta = json.loads((path.parent / "family.json").read_text())
        guard_path = path.parent / "measurement_audit.json"
        guard = json.loads(guard_path.read_text())
        if meta["split"] == "pilot" or (
            meta["split"] == "development" and meta["seed"] == 11201
        ):
            guard.update(usable=False, reason="first_use_warmup")
            guard_path.write_text(json.dumps(guard, indent=2) + "\n")
        if not guard["usable"]:
            rejected[guard["reason"]] += 1
            continue
        for line in (path.parent / "engine.jsonl").read_text().splitlines():
            event = json.loads(line)
            row, reason = corpus.extract(event, cached_only=False)
            if row is None:
                if event["event"] == "step_completion":
                    rejected[reason] += 1
                continue
            if any(
                not 0 < r[2] <= r[3] or r[3] - r[2] > 2000000
                for r in row["execution_ranks"]
            ):
                rejected["rank_clock"] += 1
                continue
            row.update(
                episode=str(path.parent),
                source_file=str(path.parent / "engine.jsonl"),
                workload=meta["id"],
                block=meta["block"],
                split=meta["split"],
                _source="controlled_current",
                cpu_cached_prefix_tokens=event.get("cpu_cached_prefix_tokens"),
            )
            rows.append(row)
    from experiments.native_scheduler_ab_20261010 import run as baseline

    sys.modules["run"] = baseline.collector
    baseline.collector.ROOT = root
    cold = load("cold", "experiments/dataset_coverage_20261007/cold.py")
    cold_rows = cold.rows(corpus)
    for row in cold_rows:
        row["_source"] = "controlled_current"
    rows.extend(cold_rows)
    data = root / "data"
    data.mkdir(exist_ok=True)
    (data / "batches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run(
        [
            sys.executable,
            str(SOURCE / "experiments/predictor_runtime_20261006/forecast_corpus.py"),
            "--corpus",
            str(data / "batches.jsonl"),
            "--output",
            str(data),
        ],
        check=True,
    )
    (data / "batch_inventory.json").write_text(
        json.dumps(
            dict(
                samples=len(rows),
                cold_samples=len(cold_rows),
                rejected=dict(rejected),
                splits=dict(Counter(r["split"] for r in rows)),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
