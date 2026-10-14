# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit trace mix, physical transfer accounting, and available execution envelopes."""

import json
import statistics
from collections import Counter
from pathlib import Path

from experiments.tracelab_arrival_intensity_20261013 import campaign as base

ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_capacity_sweep_20261013")
PREVIOUS = Path("/mnt/localnvme/jeongyoon/tracelab_arrival_intensity_20261013")
TOKEN_BYTES_ALL_RANKS = 381600


def distribution(values):
    if not values:
        return dict(count=0)
    ordered = sorted(values)
    return dict(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        p95=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        min=ordered[0],
        max=ordered[-1],
        sum=sum(values),
    )


def union_size(intervals):
    cursor, size = None, 0
    for start, end in sorted(intervals):
        size += max(0, end - max(start, cursor)) if cursor is not None else end - start
        cursor = max(cursor, end) if cursor is not None else end
    return size


def audit(directory, result, engine_paths, offsets, clock_offsets):
    rows = [json.loads(line) for line in (directory / "rounds.jsonl").open()]
    events = [json.loads(line) for line in (directory / "transfer_events.jsonl").open()]
    start = result["replay_start_timestamp"]
    begin = start + result["warmup_seconds"]
    end = begin + result["measurement_seconds"]
    posted = [row for row in rows if begin <= row["post_timestamp"] < end]
    mapping = json.loads((PREVIOUS / "session_metadata.json").read_text())
    load_bytes = sum(
        event["transfer_bytes"]
        for event in events
        if event["event"] == "job_observed_done"
    )
    store_bytes = sum(
        event["transfer_bytes"]
        for event in events
        if event["event"] == "store_observed_done"
    )
    checks = dict(
        physical_load_bytes=load_bytes
        == result["metric_deltas"].get("vllm:kv_offload_load_bytes_total", 0),
        physical_store_bytes=store_bytes
        == result["metric_deltas"].get("vllm:kv_offload_store_bytes_total", 0),
        initial_cache_tokens=sum(row["server_cached_prompt_tokens"] for row in rows)
        + 512
        == result["metric_deltas"]["vllm:prompt_tokens_cached_total"],
    )
    profile = dict(
        requests=len(posted),
        sessions=len({row["session_id"] for row in posted}),
        providers=dict(
            Counter(mapping[row["session_id"]]["provider"] for row in posted)
        ),
        lengths={
            name: distribution([row[field] for row in posted])
            for name, field in (
                ("input_tokens", "prompt_len"),
                ("recorded_prefix_tokens", "prefix_len"),
                ("recorded_append_tokens", "input_len"),
                ("computed_input_tokens", "server_uncached_prompt_tokens"),
                ("output_tokens", "output_len_target"),
                ("tool_wait_after_ms", "tool_wait_after_ms"),
            )
        },
        physical_load_bytes_whole_episode=load_bytes,
        physical_store_bytes_whole_episode=store_bytes,
        initial_prefill_source_tokens=result["token_source_deltas"],
        bytes_per_cached_token_all_ranks=TOKEN_BYTES_ALL_RANKS,
        accounting_checks=checks,
    )
    offset = statistics.mean(clock_offsets)
    drift_ms = abs(clock_offsets[-1] - clock_offsets[0]) * 1000
    lower, upper = int((begin - offset) * 1e9), int((end - offset) * 1e9)
    intervals, shapes, admissions = [], Counter(), Counter()
    missing = Counter()
    selected_sizes, prefill_sizes = [], []
    for path in engine_paths:
        with path.open("rb") as stream:
            stream.seek(offsets.get(str(path), 0))
            for line in stream:
                event = json.loads(line)
                if not lower <= event["time_ns"] < upper:
                    continue
                if event["event"] in ("dispatch", "sealed_dispatch"):
                    admissions[event["event"]] += 1
                if event["event"] != "step_completion" or not event.get("selected"):
                    continue
                shapes[(event.get("shape") or ["unknown"])[0]] += 1
                selected_sizes.append(len(event["selected"]))
                if event.get("prefill_only"):
                    prefill_sizes.append(len(event["selected"]))
                ranks = event.get("execution_ranks") or []
                if len(ranks) != 8 or any(not r[3] or not r[6] for r in ranks):
                    missing["incomplete_execution_ranks"] += 1
                    continue
                a = max(rank[3] for rank in ranks)
                b = max(rank[3] + rank[6] for rank in ranks)
                if a < upper and b > lower:
                    intervals.append((max(a, lower), min(b, upper)))
    intervals.sort()
    gaps = [max(0, b[0] - a[1]) / 1e6 for a, b in zip(intervals, intervals[1:])]
    profile["execution_envelopes"] = dict(
        available=bool(intervals),
        clock_offset_drift_ms=drift_ms,
        clock_alignment_accepted=drift_ms < 2,
        measured_compute_rounds=len(selected_sizes),
        shapes=dict(shapes),
        selected_requests_per_round=distribution(selected_sizes),
        selected_requests_per_prefill_round=distribution(prefill_sizes),
        observed_envelopes=len(intervals),
        gaps_ms=distribution(gaps),
        envelope_union_seconds=union_size(intervals) / 1e9,
        window_outside_envelopes_seconds=result["measurement_seconds"]
        - union_size(intervals) / 1e9
        if intervals
        else None,
        missing=dict(missing),
        dispatch_counts=dict(admissions),
        scope=(
            "Full prototype native all-rank start+elapsed execution envelopes. "
            "These include submission within a forward and are not kernel-level "
            "GPU utilization. Baseline disables this instrumentation; no baseline "
            "physical-gap estimate is fabricated. Wall/monotonic alignment assumes "
            "no clock step; exclude aligned gap comparisons if drift >=2ms."
        ),
    )
    base.save(directory / "audit.json", profile)
    assert all(checks.values()), checks
    return profile


def results():
    rows = [
        json.loads(path.read_text())
        for path in sorted((ROOT / "results").glob("*.json"))
    ]
    return [row for row in rows if row.get("capacity_sweep_finalized")]


def write_report():
    rows = results()
    groups = {}
    for result in rows:
        key = (
            result["arrival_rate_sessions_per_second"],
            result["measurement_request_slots"],
            result["repeat"],
        )
        groups.setdefault(key, {})[result["arm"]] = result
    pairs = []
    for (rate, limit, repeat), pair in sorted(groups.items()):
        if set(pair) != {"baseline", "full"}:
            continue
        b, f = pair["baseline"], pair["full"]
        pairs.append(
            dict(
                rate=rate,
                concurrency=limit,
                repeat=repeat,
                baseline=b,
                full=f,
                throughput_gain_pct=(f["throughput_rps"] / b["throughput_rps"] - 1)
                * 100,
                mean_ttft_reduction_pct=(
                    1 - f["ttft_ms"]["mean"] / b["ttft_ms"]["mean"]
                )
                * 100,
                p95_ttft_reduction_pct=(1 - f["ttft_ms"]["p95"] / b["ttft_ms"]["p95"])
                * 100,
            )
        )
    base.save(ROOT / "comparisons.json", pairs)
    status = "running"
    for marker in ("completed", "failure"):
        if (ROOT / f"{marker}.json").exists():
            status = marker
    lines = [
        "TraceLab capacity and arrival-rate sweep",
        f"Status: {status}; accepted cells: {len(rows)}; matched pairs: {len(pairs)}",
        "",
        (
            "sessions/s | calls | repeat | arm | req/s | HTTP TTFT mean/p95 ms | "
            "CPU-load requests % | GPU % | queue mean s | preemptions | late admissions"
        ),
    ]
    for r in sorted(
        rows,
        key=lambda r: (
            r["arrival_rate_sessions_per_second"],
            r["measurement_request_slots"],
            r["repeat"],
            r["arm"],
        ),
    ):
        gpu = r["cache_profile"]["gpu_utilization_mean_percent"]
        gpu_text = "N/A" if gpu is None else f"{gpu:.2f}"
        lines.append(
            f"{r['arrival_rate_sessions_per_second']:.2f} | "
            f"{r['measurement_request_slots']} | {r['repeat']} | {r['arm']} | "
            f"{r['throughput_rps']:.5f} | "
            f"{r['ttft_ms']['mean']:.2f}/{r['ttft_ms']['p95']:.2f} | "
            f"{r['cache_profile']['measurement']['cpu_hit_request_percent']:.2f} | "
            f"{gpu_text} | {r['slot_queue_ms']['mean'] / 1000:.2f} | "
            f"{r['metric_deltas'].get('vllm:num_preemptions_total', 0):.0f} | "
            f"{r['protocol_counts'].get('membership_sealed_activated', 0)}"
        )
    lines += [
        "",
        (
            "sessions/s | calls | repeat | throughput gain % | "
            "mean TTFT reduction % | p95 TTFT reduction %"
        ),
    ]
    for pair in pairs:
        lines.append(
            f"{pair['rate']:.2f} | {pair['concurrency']} | {pair['repeat']} | "
            f"{pair['throughput_gain_pct']:+.2f} | "
            f"{pair['mean_ttft_reduction_pct']:+.2f} | "
            f"{pair['p95_ttft_reduction_pct']:+.2f}"
        )
    lines += [
        "",
        "Every cell: 600s warmup + 900s measurement; logical CPU/GPU caches reset.",
        "Reuse servers within each arm group to amortize population/registration.",
        "Preserve the pinned full 8058-session/665453-call seed0 catalogue, native",
        "prompt construction, recorded outputs and tool waits. Only synthetic",
        "arrival pacing and in-flight call limits vary. No agent-wide session cap.",
        "GLM5.3 NVFP4 TP8, 1M context, same maximum validated CPU budget and",
        "96GiB/rank GPU KV, frozen predictor, original sparse selector and policies.",
        "One-second client idle expiry, no retries. Every submitted call drains.",
        "Throughput uses completion window; TTFT/cache/mix use the POST cohort.",
        "Transfers, admissions and preemptions above cover warmup+measurement+drain.",
        "CPU-load requests include preemption reloads; initial-prefill token counts",
        "and physical transfer bytes are audited separately. GPU-only outcomes are",
        "requests with cached tokens and no observed CPU load, not GPU token shares.",
        "Execution-envelope gaps are available for full prototype only; no baseline",
        "physical-gap estimate is inferred from GPU utilization or HTTP timestamps.",
        "A growing queue is not indefinite steady-state stability. Faster closed-loop",
        "progress changes finite-window request mix; audit.json records that mix.",
        "All positive and negative results remain. Individual pairs are point",
        "estimates; compare repeated 32 cells and optional replication separately.",
    ]
    (ROOT / "REPORT.txt").write_text("\n".join(lines) + "\n")
    return pairs
