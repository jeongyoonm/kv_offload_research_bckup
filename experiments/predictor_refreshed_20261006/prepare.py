# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze expanded representative cached training data after held-out scoring."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/mnt/localnvme/glm53_predictor_overnight_20261006")


def legacy_cache_lineage(directory):
    result = json.loads((directory / "result.json").read_text())
    distribution = result.get("cached_distribution", {})
    if result.get("gpu_hits") != 0 or not distribution:
        return False, {}
    if all(int(tokens) > 0 for tokens in distribution):
        return True, {}
    usage_path = directory / "usage.json"
    if not usage_path.exists():
        return False, {}
    usage = json.loads(usage_path.read_text())
    cached = {}
    for line in (directory / "engine.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event"] != "step_completion" or not event.get("selected"):
            continue
        values = []
        for request in event["selected"]:
            matches = [
                row["cached"]
                for row in usage
                if request == row["request_id"]
                or request.startswith("cmpl-" + row["request_id"] + "-")
            ]
            if len(matches) != 1:
                values = []
                break
            values.append(matches[0])
        cached[event["step"]] = bool(values) and min(values) > 0
    return False, cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = dict(
        batches=[
            ROOT / "expanded/batches.jsonl",
            ROOT / "live_comparison/frozen_evaluation/corpus/corpus.jsonl",
            ROOT / "micro_unprofiled_corpus/corpus.jsonl",
        ],
        forecasts=[
            ROOT / "expanded/forecasts.jsonl",
            ROOT / "live_comparison/frozen_evaluation/forecasts/forecasts.jsonl",
        ],
    )
    manifest = dict(
        protocol=(
            "The completed first live comparison becomes development data only "
            "after all previously frozen models were scored. Submission-progress "
            "and subsequent live phases remain held out from this checkpoint."
        ),
        features="Cached request shapes and causal load state; no preparation mode",
        inputs={},
        datasets={},
    )
    legacy = {}
    for kind, paths in inputs.items():
        rows, seen = [], set()
        rejected = Counter()
        for path in paths:
            manifest["inputs"][str(path)] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key = row["source_file"], row["step"]
                if key in seen:
                    rejected["duplicate"] += 1
                    continue
                if (Path(row["episode"]) / "profile_capture.json").exists():
                    raise ValueError("profiled labels cannot enter cached training")
                cached = row.get("cpu_cached_prefix_tokens")
                if cached is None:
                    directory = Path(row["episode"])
                    if directory not in legacy:
                        legacy[directory] = legacy_cache_lineage(directory)
                    all_cached, steps = legacy[directory]
                    if not all_cached and not steps.get(row["step"], False):
                        rejected["legacy_cold_or_unknown_cache"] += 1
                        continue
                    row["cache_lineage_source"] = "accepted_usage_with_zero_gpu_hits"
                if cached is not None and (
                    len(cached) != len(row["request_shapes"])
                    or any(type(v) is not int or v <= 0 for v in cached)
                ):
                    raise ValueError("cold or unknown per-request cache lineage")
                if any(q <= 1 or c <= 0 for q, c in row["request_shapes"]):
                    raise ValueError("only cached prefills belong in this model")
                if not all(
                    0 < e[2] <= e[3] and e[3] - e[2] <= 2_000_000
                    for e in row["execution_ranks"]
                ):
                    rejected["invalid_rank_clock"] += 1
                    continue
                seen.add(key)
                rows.append(row)
        with (args.output / f"{kind}.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        manifest["datasets"][kind] = dict(
            samples=len(rows),
            episodes=len({r["episode"] for r in rows}),
            workloads=dict(Counter(r["workload"] for r in rows)),
            rejected=dict(rejected),
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
