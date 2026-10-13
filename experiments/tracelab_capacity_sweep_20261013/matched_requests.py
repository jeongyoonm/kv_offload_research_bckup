# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Supplement primary window results with shared-request latency comparisons."""

import argparse
import json
import time

from experiments.tracelab_capacity_sweep_20261013.analyze import ROOT, distribution


def cohort(result):
    directory = ROOT / "episodes" / result["episode"] / "cells" / result["arm"]
    begin = result["replay_start_timestamp"] + result["warmup_seconds"]
    end = begin + result["measurement_seconds"]
    rows = [json.loads(line) for line in (directory / "rounds.jsonl").open()]
    return {
        row["request_id"]: row for row in rows if begin <= row["post_timestamp"] < end
    }


def write():
    path = ROOT / "comparisons.json"
    pairs = json.loads(path.read_text()) if path.exists() else []
    results = []
    for pair in pairs:
        baseline, full = cohort(pair["baseline"]), cohort(pair["full"])
        common = sorted(set(baseline) & set(full))
        checks = {
            field: all(baseline[key][field] == full[key][field] for key in common)
            for field in (
                "prompt_len",
                "prefix_len",
                "input_len",
                "output_len_target",
                "tool_wait_after_ms",
            )
        }
        assert all(checks.values()), checks
        latencies = {}
        for arm, rows in (("baseline", baseline), ("full", full)):
            selected = [rows[key] for key in common]
            latencies[arm] = dict(
                http_ttft_ms=distribution([row["first_token_ms"] for row in selected]),
                ready_ttft_ms=distribution(
                    [
                        (row["post_timestamp"] - row["ready_timestamp"]) * 1000
                        + row["first_token_ms"]
                        for row in selected
                    ]
                ),
                cached_input_fraction=sum(
                    row["server_cached_prompt_tokens"] for row in selected
                )
                / sum(row["prompt_len"] for row in selected)
                if selected
                else None,
            )
        delta = [
            full[key]["first_token_ms"] - baseline[key]["first_token_ms"]
            for key in common
        ]
        results.append(
            dict(
                rate=pair["rate"],
                concurrency=pair["concurrency"],
                repeat=pair["repeat"],
                common_requests=len(common),
                baseline_only_requests=len(baseline) - len(common),
                full_only_requests=len(full) - len(common),
                length_and_tool_wait_checks=checks,
                latency=latencies,
                paired_http_ttft_delta_ms=distribution(delta),
                fraction_faster_in_full=sum(value < 0 for value in delta) / len(delta)
                if delta
                else None,
                request_ids=common,
            )
        )
    output = dict(
        pairs=results,
        scope=(
            "Intersection of measurement POST request IDs in both arms. Lengths "
            "and recorded waits verified per ID. HTTP TTFT excludes the client "
            "slot wait; ready TTFT adds that wait and excludes preceding tool time. "
            "Supplemental conditioning selects requests that both systems posted "
            "inside their windows. It does not replace primary metrics or estimate "
            "throughput, eliminate cache-state differences, or establish causal "
            "latency effects. Generated token contents can differ across arms."
        ),
    )
    (ROOT / "MATCHED_REQUESTS.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    stamp = None
    while True:
        path = ROOT / "comparisons.json"
        current = path.stat().st_mtime_ns if path.exists() else None
        if current != stamp:
            summary = write()
            print(f"Shared-request pairs analyzed: {len(summary['pairs'])}", flush=True)
            stamp = current
        terminal = any(
            (ROOT / name).exists() for name in ("completed.json", "failure.json")
        )
        if not args.watch or terminal:
            write()
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
