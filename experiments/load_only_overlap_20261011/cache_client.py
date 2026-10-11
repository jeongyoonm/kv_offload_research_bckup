# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-outcome serving measurements with identical deterministic inputs."""

import concurrent.futures
import json
import statistics
import time
from collections import Counter, defaultdict

import numpy as np
import regex as re

from experiments.load_only_overlap_20261011.live import stream
from experiments.native_scheduler_ab_20261010.run import collector as run


def metrics():
    values = defaultdict(float)
    for line in run.textcall("/metrics").splitlines():
        if line.startswith("#") or not line:
            continue
        name, value = line.split()[:2]
        key = name.split("{")[0]
        if key.endswith("_total") or key.endswith("_count") or key.endswith("_sum"):
            values[key] += float(value)
    return dict(values)


def flush_timeline(root):
    directory = root / "server/load_timeline"
    directory.mkdir(parents=True, exist_ok=True)
    before = set(directory.glob("*.jsonl"))
    trigger = directory / "flush.request"
    trigger.touch()
    stamp = trigger.stat().st_mtime_ns
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if len(list(directory.glob(f"*.{stamp}.jsonl"))) == 9:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("timeline flush requires engine and all eight ranks")
    paths = sorted(set(directory.glob("*.jsonl")) - before)
    return [
        json.loads(line) for path in paths for line in path.read_text().splitlines()
    ]


def issue(requests, label, concurrency, *, initial_barrier=True):
    """Start with one engine-visible cohort, then allow normal replenishment."""
    if initial_barrier:
        run.call("/pause?mode=keep&clear_cache=false", {})
    responses = []
    started = time.monotonic_ns()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(stream, r["prefix"] + r["tail"], f"{label}-{r['index']}")
                for r in requests
            ]
            if initial_barrier:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    status = run.call("/experimental/deferral-cohort")
                    if (
                        status["paused"]
                        and not status["queued_steps"]
                        and status["requests"] == min(concurrency, len(requests))
                    ):
                        break
                    time.sleep(0.005)
                else:
                    raise RuntimeError("initial cohort did not arrive")
                started = time.monotonic_ns()
                run.call("/resume", {})
            responses = [f.result() for f in futures]
            ended = time.monotonic_ns()
    finally:
        if initial_barrier:
            run.call("/resume", {})
    return responses, started, ended


def prime(requests, label):
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for start in range(0, len(requests), 16):
            futures = [
                pool.submit(
                    run.completion, r["prefix"] + [31700] * 64, f"{label}-{r['index']}"
                )
                for r in requests[start : start + 16]
            ]
            for future in futures:
                future.result()


def summarize(responses, windows, events, engine_events, *, expected=None):
    ttfts = [
        (r["first_ns"] - max(r["start_ns"], start)) / 1e6 for r, start in responses
    ]
    observed = [
        r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        for r, _ in responses
    ]
    jobs = defaultdict(dict)
    requests_loaded = set()
    for event in events:
        if event["event"] == "job_observed_done":
            jobs[event["job_id"]][event["pid"]] = event
            requests_loaded.add(event["request"])
    stores = defaultdict(dict)
    for event in events:
        if event["event"] == "store_observed_done":
            stores[event["job_id"]][event["pid"]] = event
    counts = Counter()
    for response, _ in responses:
        name = response["request"]
        loaded = any(
            re.search(r"(?:^|-)" + re.escape(name) + r"(?:$|-)", req)
            for req in requests_loaded
        )
        cached = (
            response["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        )
        counts["cpu_hit" if loaded else "gpu_hit" if cached else "miss"] += 1
    compute = [
        e
        for e in engine_events
        if e["event"] == "step_completion" and e.get("selected") and e.get("valid")
    ]
    intervals = sorted(
        (
            max(r[3] for r in e["execution_ranks"]),
            max(r[3] + r[6] for r in e["execution_ranks"]),
        )
        for e in compute
    )
    gaps = sum(
        max(0, later[0] - earlier[1])
        for start, end in windows
        for measured in [[x for x in intervals if start <= x[0] <= end]]
        for earlier, later in zip(measured, measured[1:])
    )
    seconds = sum(end - start for start, end in windows) / 1e9
    accepted = (
        all(r["usage"]["completion_tokens"] == 1 for r, _ in responses)
        and all(len(rank_events) == 8 for rank_events in jobs.values())
        and (expected is None or observed == expected)
    )
    return dict(
        accepted=accepted,
        requests=len(responses),
        seconds=seconds,
        throughput_rps=len(responses) / seconds,
        mean_ttft_ms=statistics.mean(ttfts),
        p50_ttft_ms=float(np.quantile(ttfts, 0.5)),
        p95_ttft_ms=float(np.quantile(ttfts, 0.95)),
        p99_ttft_ms=float(np.quantile(ttfts, 0.99)),
        realized_outcomes=dict(counts),
        outcome_definition="CPU hit if any CPU load, otherwise GPU hit if cached "
        "tokens >0, otherwise miss; partial/mixed-tier requests "
        "are included in CPU hits.",
        cached_tokens=observed,
        load_jobs=len(jobs),
        h2d_bytes_all_ranks=sum(
            e["transfer_bytes"] for ranks in jobs.values() for e in ranks.values()
        ),
        h2d_active_ms_all_ranks=sum(
            (e.get("transfer_time_seconds") or 0) * 1000
            for ranks in jobs.values()
            for e in ranks.values()
        ),
        h2d_job_slowest_rank_ms=[
            max((e.get("transfer_time_seconds") or 0) * 1000 for e in ranks.values())
            for ranks in jobs.values()
        ],
        d2h_bytes_all_ranks=sum(
            (e.get("transfer_bytes") or 0)
            for ranks in stores.values()
            for e in ranks.values()
        ),
        store_jobs_observed_done=len(stores),
        store_rank_coverage=[len(ranks) for ranks in stores.values()],
        compute_batch_requests=[len(e["selected"]) for e in compute],
        compute_batch_tokens=[sum(e["selected"].values()) for e in compute],
        late_admissions=sum(
            len(e.get("admitted", []))
            for e in engine_events
            if e["event"] == "late_admission_sealed"
        ),
        compute_rounds=len(compute) if engine_events else None,
        gpu_envelope_ms=sum(end - start for start, end in intervals) / 1e6
        if intervals
        else None,
        inter_compute_gaps_ms=gaps / 1e6 if intervals else None,
        responses=[r for r, _ in responses],
    )
