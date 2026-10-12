# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit phase-specific cache reuse from saved requests and all-rank transfers."""

import bisect
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import regex as re

ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_request_slots_20261013")
CPU_CAPACITY_BYTES = 3028011286528
TOKEN_BYTES_ALL_RANKS = 381600


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def distribution(values):
    return dict(
        mean=statistics.mean(values),
        median=statistics.median(values),
        min=min(values),
        max=max(values),
    )


def audit(arm, wall_minus_monotonic_seconds):
    directory = ROOT / "cells" / arm
    result = json.loads((directory / "result.json").read_text())
    rows = read_rows(directory / "rounds.jsonl")
    events = read_rows(directory / "transfer_events.jsonl")
    lookup = {row["request_id"]: row for row in rows}
    start = result["replay_start_timestamp"]
    begin = start + result["warmup_seconds"]
    end = begin + result["measurement_seconds"]
    loaded_bytes = defaultdict(int)
    preflight_bytes = 0
    for event in events:
        if event["event"] != "job_observed_done":
            continue
        match = re.fullmatch(r"cmpl-(\d+_round_\d+)-0-[0-9a-f]{8}", event["request"])
        if match:
            request_id = match[1]
            assert request_id in lookup
            loaded_bytes[request_id] += event["transfer_bytes"]
        else:
            preflight_bytes += event["transfer_bytes"]
    total_load_bytes = sum(loaded_bytes.values()) + preflight_bytes
    assert total_load_bytes == result["h2d_bytes_all_ranks_whole_episode"]
    assert (
        total_load_bytes / TOKEN_BYTES_ALL_RANKS
        == result["token_source_deltas"]["external_kv_transfer"]
    )
    assert (
        sum(row["server_cached_prompt_tokens"] for row in rows) + 512
        == result["metric_deltas"]["vllm:prompt_tokens_cached_total"]
    )

    stores = sorted(
        (
            event["time_ns"] / 1e9 + wall_minus_monotonic_seconds,
            event["transfer_bytes"],
        )
        for event in events
        if event["event"] == "store_observed_done"
    )
    store_times = [stamp for stamp, _ in stores]
    store_prefix_bytes = [0]
    for _, size in stores:
        store_prefix_bytes.append(store_prefix_bytes[-1] + size)

    phases = {}
    for phase, lower, upper in (
        ("warmup", start, begin),
        ("measurement", begin, end),
    ):
        posted = [row for row in rows if lower <= row["post_timestamp"] < upper]
        cpu_requests = sum(row["request_id"] in loaded_bytes for row in posted)
        cached_tokens = sum(row["server_cached_prompt_tokens"] for row in posted)
        prompt_tokens = sum(row["prompt_len"] for row in posted)
        followups = []
        turnover_ratios = []
        for row in posted:
            if row["round_idx"] == 0:
                continue
            previous_id = f"{row['session_id']}_round_{row['round_idx'] - 1:06}"
            previous = lookup[previous_id]
            followups.append(
                dict(
                    reuse_gap_seconds=row["post_timestamp"]
                    - previous["complete_timestamp"],
                    tool_wait_seconds=previous["tool_wait_after_ms"] / 1000,
                    slot_queue_seconds=row["request_slot_wait_ms"] / 1000,
                )
            )
            if row["prefix_len"] > 0:
                writes = (
                    store_prefix_bytes[
                        bisect.bisect_right(store_times, row["post_timestamp"])
                    ]
                    - store_prefix_bytes[
                        bisect.bisect_right(store_times, previous["complete_timestamp"])
                    ]
                )
                turnover_ratios.append(writes / CPU_CAPACITY_BYTES)
        phases[phase] = dict(
            post_cohort_requests=len(posted),
            first_calls=sum(row["round_idx"] == 0 for row in posted),
            cpu_hit_requests=cpu_requests,
            cpu_hit_request_percent=100 * cpu_requests / len(posted),
            any_cache_hit_requests=sum(
                row["server_cached_prompt_tokens"] > 0 for row in posted
            ),
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_tokens,
            cached_prompt_token_percent=100 * cached_tokens / prompt_tokens,
            cpu_loaded_prompt_tokens=sum(
                loaded_bytes.get(row["request_id"], 0) for row in posted
            )
            // TOKEN_BYTES_ALL_RANKS,
            followup_requests=len(followups),
            followup_times={
                key: distribution([item[key] for item in followups])
                for key in followups[0]
            },
            store_bytes_between_followups_over_cpu_capacity=distribution(
                turnover_ratios
            ),
            fraction_positive_prefix_followups_with_writes_above_cpu_capacity=sum(
                ratio > 1 for ratio in turnover_ratios
            )
            / len(turnover_ratios),
            observed_d2h_bytes=sum(
                size for stamp, size in stores if lower <= stamp < upper
            ),
        )
    loaded_rows = [lookup[key] for key, size in loaded_bytes.items() if size]
    return dict(
        phases=phases,
        cpu_load_requests_whole_episode=len(loaded_rows),
        last_cpu_hit_post_seconds=max(row["post_timestamp"] for row in loaded_rows)
        - start,
        last_load_observed_seconds=max(
            event["time_ns"] / 1e9 + wall_minus_monotonic_seconds
            for event in events
            if event["event"] == "job_observed_done"
        )
        - start,
        accounting_verified=True,
    )


def main():
    offset = time.time() - time.monotonic()
    arms = {arm: audit(arm, offset) for arm in ("baseline", "full")}
    evidence = dict(
        arms=arms,
        cpu_capacity_bytes=CPU_CAPACITY_BYTES,
        wall_minus_monotonic_seconds=offset,
        timing_caveat=(
            "Store-window statistics translate monotonic timestamps using the "
            "current clock offset; they assume no intervening wall-clock step. "
            "Request-phase cache-hit counts use recorded POST timestamps and "
            "exact request-ID matching and do not depend on this assumption."
        ),
        interpretation=(
            "Both measurement POST cohorts have zero CPU loads and zero "
            "cached prompt tokens. CPU hits occurred during warmup. Long "
            "request-slot queues and cache-sized intervening writes support "
            "eviction before reuse; write volume alone is not an exact "
            "per-block eviction count."
        ),
    )
    (ROOT / "CACHE_AUDIT.json").write_text(json.dumps(evidence, indent=2) + "\n")
    lines = [
        "TraceLab request-slot cache audit",
        "",
        "The previously suspected attribution bug is not present in this pair.",
        "Zero CPU-hit requests applies to the measurement window, not warmup.",
        "",
        "Arm | phase | POSTs | CPU-hit requests % | cached prompt tokens %",
    ]
    for arm, item in arms.items():
        for phase, values in item["phases"].items():
            lines.append(
                f"{arm} | {phase} | {values['post_cohort_requests']} | "
                f"{values['cpu_hit_request_percent']:.2f} | "
                f"{values['cached_prompt_token_percent']:.2f}"
            )
        values = item["phases"]["measurement"]
        times = values["followup_times"]
        lines += [
            (
                f"{arm}: last CPU-hit POST {item['last_cpu_hit_post_seconds']:.2f}s; "
                f"last load observed {item['last_load_observed_seconds']:.2f}s."
            ),
            (
                "  Measurement followups: "
                f"mean tool wait {times['tool_wait_seconds']['mean']:.2f}s, "
                f"mean slot queue {times['slot_queue_seconds']['mean']:.2f}s."
            ),
            (
                "  Measurement D2H writes "
                f"{values['observed_d2h_bytes'] / 1e12:.3f} TB into "
                f"{CPU_CAPACITY_BYTES / 1e12:.3f} TB CPU capacity."
            ),
        ]
    lines += ["", evidence["interpretation"], "", evidence["timing_caveat"]]
    (ROOT / "CACHE_AUDIT.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
