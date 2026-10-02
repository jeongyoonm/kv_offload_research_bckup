# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Describe replay batch shapes without attributing GPU time to pure kernels."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze import stats


def describe(cell):
    usage = json.loads((cell / "usage.json").read_text())
    ids = {row["request_id"] for row in usage}
    batches, invalid, missing_timing = [], [], []
    for line in (cell / "engine.jsonl").read_text().splitlines():
        row = json.loads(line)
        selected = row.get("selected", {})
        if row["event"] != "step_completion" or not any(
            request in req for req in selected for request in ids
        ):
            continue
        ranks = row.get("execution_ranks", [])
        shapes = row.get("request_shapes", [])
        if not shapes or sum(n for n, _ in shapes) != sum(selected.values()):
            invalid.append(row["step"])
            continue
        complete_timing = len(ranks) == 8 and all(r and r[6] for r in ranks)
        if not complete_timing:
            missing_timing.append(row["step"])
        batches.append(
            dict(
                step=row["step"],
                requests=len(shapes),
                uncached_tokens=sum(n for n, _ in shapes),
                attention_work=sum(n * (context + n / 2) for n, context in shapes),
                gpu_wall_ms=max(r[6] for r in ranks) / 1e6 if complete_timing else None,
                primary_completion_valid=row.get("valid", False),
            )
        )
    grouped = defaultdict(list)
    for batch in batches:
        if batch["gpu_wall_ms"] is not None:
            grouped[batch["requests"]].append(batch["gpu_wall_ms"])
    return dict(
        path=str(cell),
        invalid_steps=invalid,
        missing_execution_timing_steps=missing_timing,
        compute_batches=len(batches),
        total_uncached_tokens=sum(batch["uncached_tokens"] for batch in batches),
        requests_per_batch=stats([batch["requests"] for batch in batches]),
        batch_request_histogram=dict(Counter(batch["requests"] for batch in batches)),
        gpu_wall_by_request_count={n: stats(values) for n, values in grouped.items()},
        batches=batches,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=Path)
    args = parser.parse_args()
    results = [
        describe(path.parent)
        for path in sorted(args.phase.glob("*/*/result.json"))
        if json.loads(path.read_text()).get("accepted")
    ]
    (args.phase / "batch_shapes.json").write_text(
        json.dumps(
            dict(
                cells=results,
                limitation=(
                    "Token work and batch counts include completed batches with "
                    "missing GPU timing; their GPU wall time is null. "
                    "GPU wall time includes required output completion and host "
                    "submission stalls. Batch-shape differences are descriptive; "
                    "they do not isolate kernel cost or transfer interference."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    for row in results:
        print(
            Path(row["path"]).parent.name,
            Path(row["path"]).name,
            "batches",
            row["compute_batches"],
            "tokens",
            row["total_uncached_tokens"],
            "requests/batch",
            row["requests_per_batch"].get("mean"),
            "invalid",
            row["invalid_steps"],
            "missing timing",
            row["missing_execution_timing_steps"],
        )


if __name__ == "__main__":
    main()
