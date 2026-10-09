# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched streaming confirmation of frozen predictors under one seal policy."""

import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import run


def stream(tokens, name):
    start = time.monotonic_ns()
    request = urllib.request.Request(
        run.BASE + "/v1/completions",
        data=json.dumps(
            dict(
                model="glm53-compat",
                prompt=tokens,
                request_id=name,
                max_tokens=1,
                temperature=0,
                seed=0,
                ignore_eos=True,
                stream=True,
                stream_options=dict(include_usage=True),
            )
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    first = None
    usage = None
    output = ""
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            for choice in event.get("choices", []):
                first = first or time.monotonic_ns()
                if choice.get("text"):
                    output += choice["text"]
            if event.get("usage"):
                usage = event["usage"]
    end = time.monotonic_ns()
    if first is None or usage is None:
        raise RuntimeError("stream omitted first token or usage")
    return dict(
        request=name,
        start_ns=start,
        first_ns=first,
        end_ns=end,
        ttft_ms=(first - start) / 1e6,
        usage=usage,
        output=output,
    )


def cell(
    label,
    state,
    workload,
    seed,
    repeat,
    requests_count=75,
    *,
    require_preparation=True,
    prefix_working_set=None,
):
    directory = run.ROOT / "live" / workload / f"r{repeat}_{label}"
    if (directory / "result.json").exists():
        return json.loads((directory / "result.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    run.reset(True)
    homogeneous = workload in ("familiar60k", "short16k", "mid32k")
    if prefix_working_set is not None:
        assert homogeneous and 0 < prefix_working_set <= requests_count
    context, suffix = {
        "familiar60k": (58368, 3072),
        "short16k": (15552, 832),
        "mid32k": (30720, 2048),
        "heterogeneous": (24576, 8192),
    }[workload]
    requests, primes, expected = [], [], []
    shared_prefixes = {}
    for i in range(requests_count):
        c = context if homogeneous or i % 2 else 4096
        s = suffix if homogeneous or i % 3 else 2048
        prefix, tail = run.prompt(c, s, seed * 100000 + i)
        if prefix_working_set is not None:
            slot = i % prefix_working_set
            if slot not in shared_prefixes:
                shared_prefixes[slot] = prefix
            prefix = shared_prefixes[slot]
        hit = homogeneous or i % 2 == 0
        expected.append(c if hit else 0)
        requests.append(prefix + tail)
        if hit and (prefix_working_set is None or i < prefix_working_set):
            primes.append(prefix + [31700] * 64)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        run.configure()
        for i in range(0, len(primes), 32):
            futures = [
                pool.submit(run.completion, p, f"live-prime-{seed}-{j}")
                for j, p in enumerate(primes[i : i + 32], start=i)
            ]
            for f in futures:
                f.result()
        run.reset(False)
        run.save(directory / "policy.json", run.configure(True, state))
        before = run.counters()
        offsets = run.snapshots()
        late_before = set((run.ROOT / "server/timing/late_admission").glob("*.json"))
        start = time.monotonic_ns()
        futures = [
            pool.submit(stream, p, f"live-main-{seed}-{i}")
            for i, p in enumerate(requests)
        ]
        responses = [f.result() for f in futures]
        elapsed = (time.monotonic_ns() - start) / 1e9
    events = run.traces(offsets, directory)
    engine_records, late_sources, worker_exports, engine_exports = [], [], [], []
    for path in sorted(
        set((run.ROOT / "server/timing/late_admission").glob("*.json")) - late_before
    ):
        data = json.loads(path.read_text())
        if data.get("role") == "engine":
            if data.get("overflow"):
                raise RuntimeError("live seal trace overflowed")
            engine_records.extend(data["records"])
            engine_exports.append(data)
            late_sources.append(str(path))
        elif data.get("role") == "worker":
            worker_exports.append(dict(source=str(path), **data))
    run.save(
        directory / "late.json",
        dict(
            records=engine_records,
            sources=late_sources,
            overflow=False,
            generations=[e["generation"] for e in engine_exports],
            worker_sources=[e["source"] for e in worker_exports],
        ),
    )
    after = run.counters()
    exported = run.call(
        "/experimental/kv-load-scheduling",
        dict(
            early_load_completion=True,
            experiment_options=run.options(True),
            seal_estimator=dict(action="export"),
        ),
    )["seal_estimator"]["state"]
    frozen = exported == state
    restored = [
        r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        for r in responses
    ]
    accepted = (
        frozen
        and restored == expected
        and all(r["usage"]["completion_tokens"] == 1 for r in responses)
        and after.get("vllm:num_preemptions_total", 0)
        == before.get("vllm:num_preemptions_total", 0)
        and after.get("vllm:prefix_cache_hits_total", 0)
        == before.get("vllm:prefix_cache_hits_total", 0)
    )
    ttft = [r["ttft_ms"] for r in responses]
    activated = {
        req
        for e in engine_records
        if e["event"] == "membership_sealed"
        for req in e.get("activated", {})
    }
    completed_steps = [
        e
        for e in events
        if e["event"] == "step_completion" and e.get("selected") and e.get("valid")
    ]
    intervals = [
        (
            max(r[3] for r in e["execution_ranks"]),
            max(r[3] + r[6] for r in e["execution_ranks"]),
        )
        for e in completed_steps
    ]
    jobs = {
        e["job_id"]: e["request"] for e in events if e["event"] == "kv_load_dispatched"
    }
    ready_during_compute = {
        jobs[e["job_id"]]
        for e in events
        if e["event"] == "kv_all_rank_completion"
        and e["job_id"] in jobs
        and any(a <= e["completion_ns"] < b for a, b in intervals)
    }
    worker_coverage = {}
    compute_steps = {e["step"] for e in completed_steps}
    for export in worker_exports:
        entries = [
            e
            for e in export["records"]
            if e["event"] == "worker_entry" and e.get("step") in compute_steps
        ]
        if export.get("overflow"):
            raise RuntimeError("worker trace overflowed")
        totals = worker_coverage.setdefault(
            str(export["rank"]), dict(entries=0, compact=0, pinned=0)
        )
        totals["entries"] += len(entries)
        totals["compact"] += sum(bool(e.get("compact_tokens")) for e in entries)
        totals["pinned"] += sum(bool(e.get("pinned_tokens")) for e in entries)
    preparation_verified = set(worker_coverage) == {str(r) for r in range(8)} and all(
        coverage["compact"] > 0 and coverage["pinned"] > 0
        for coverage in worker_coverage.values()
    )
    accepted = accepted and (
        preparation_verified if require_preparation else not activated
    )
    result = dict(
        accepted=accepted,
        label=label,
        workload=workload,
        seed=seed,
        repeat=repeat,
        concurrency=32,
        requests=requests_count,
        prefix_working_set=prefix_working_set,
        seconds=elapsed,
        throughput_rps=requests_count / elapsed,
        mean_ttft_ms=statistics.mean(ttft),
        p50_ttft_ms=float(np.quantile(ttft, 0.5)),
        p95_ttft_ms=float(np.quantile(ttft, 0.95)),
        frozen_unchanged=frozen,
        checkpoint_sha256=hashlib.sha256(
            json.dumps(state, sort_keys=True).encode()
        ).hexdigest(),
        compute_rounds=sum(
            e["event"] == "step_completion" and bool(e.get("selected")) for e in events
        ),
        expected_cached=expected,
        observed_cached=restored,
        late_admissions=len(activated),
        kv_ready_during_compute=len(ready_during_compute),
        recovered_compute_opportunities=len(activated & ready_during_compute),
        worker_preparation_coverage=worker_coverage,
        optimized_preparation_verified=preparation_verified,
        preparation_required=require_preparation,
        responses=responses,
        metrics_before=before,
        metrics_after=after,
    )
    run.save(directory / "result.json", result)
    run.log(
        "live_cell",
        label=label,
        workload=workload,
        repeat=repeat,
        accepted=accepted,
        throughput_rps=result["throughput_rps"],
    )
    if not accepted:
        raise RuntimeError(f"live cell failed validation: {directory}")
    return result


def main():
    selection = json.loads((run.ROOT / "study/selection.json").read_text())
    models = {
        "historical": json.loads(
            (run.HIST / "boosted_residual/model.json").read_text()
        )["state"],
        "selected": json.loads(Path(selection["checkpoint"]).read_text())["state"],
    }
    allocation = json.loads((run.ROOT / "allocation.json").read_text())
    used = allocation["total_seconds"]
    if used + 900 > 14400:
        raise RuntimeError("insufficient GPU budget for a fresh live server")
    proc = None
    started = time.monotonic()
    watchdog = None
    results = []
    paired_seconds = []
    try:
        proc = run.launch()
        watchdog = threading.Timer(max(1, 14400 - used - 60), run.stop_server, (proc,))
        watchdog.daemon = True
        watchdog.start()
        while time.monotonic() - started < 1500:
            if proc.poll() is not None:
                raise RuntimeError("live server exited during startup")
            try:
                run.call("/health", timeout=2)
                break
            except (OSError, TimeoutError):
                time.sleep(2)
        else:
            raise RuntimeError("live server startup exceeded budget")
        # ABBA ordering within each workload, with identical prompt seeds per pair.
        for repeat in range(4):
            for workload in ["familiar60k", "heterogeneous"]:
                reserve = max(
                    600 if not paired_seconds else 300,
                    1.5 * max(paired_seconds, default=0) + 60,
                )
                if used + time.monotonic() - started > 14400 - reserve:
                    run.log(
                        "live_budget_stop",
                        completed_cells=len(results),
                        reserve_seconds=reserve,
                    )
                    return
                order = (
                    ["historical", "selected"]
                    if repeat in [0, 3]
                    else ["selected", "historical"]
                )
                pair_start = time.monotonic()
                for label in order:
                    results.append(
                        cell(label, models[label], workload, 1001 + repeat, repeat)
                    )
                paired_seconds.append(time.monotonic() - pair_start)
        run.log("live_complete", cells=len(results))
    finally:
        if watchdog:
            watchdog.cancel()
        if proc:
            run.stop_server(proc)
        run.save(
            run.ROOT / "allocation.json",
            dict(
                seconds=time.monotonic() - started,
                total_seconds=used + time.monotonic() - started,
                limit_seconds=14400,
                server_stopped=proc is None or proc.poll() is not None,
            ),
        )
        run.save(
            run.ROOT / "live/summary.json",
            dict(cells=results, complete=len(results) == 16),
        )


if __name__ == "__main__":
    main()
