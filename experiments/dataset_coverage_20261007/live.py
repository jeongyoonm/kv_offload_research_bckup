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
                if choice.get("text"):
                    first = first or time.monotonic_ns()
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


def cell(label, state, workload, seed, repeat):
    directory = run.ROOT / "live" / workload / f"r{repeat}_{label}"
    if (directory / "result.json").exists():
        return json.loads((directory / "result.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    run.reset(True)
    context = 58368 if workload == "familiar60k" else 24576
    suffix = 3072 if workload == "familiar60k" else 8192
    requests, primes, expected = [], [], []
    for i in range(75):
        c = context if workload == "familiar60k" or i % 2 else 4096
        s = suffix if workload == "familiar60k" or i % 3 else 2048
        prefix, tail = run.prompt(c, s, seed * 100000 + i)
        hit = workload == "familiar60k" or i % 2 == 0
        expected.append(c if hit else 0)
        requests.append(prefix + tail)
        if hit:
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
        start = time.monotonic_ns()
        futures = [
            pool.submit(stream, p, f"live-main-{seed}-{i}")
            for i, p in enumerate(requests)
        ]
        responses = [f.result() for f in futures]
        elapsed = (time.monotonic_ns() - start) / 1e9
    events = run.traces(offsets, directory)
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
    result = dict(
        accepted=accepted,
        label=label,
        workload=workload,
        seed=seed,
        repeat=repeat,
        concurrency=32,
        requests=75,
        seconds=elapsed,
        throughput_rps=75 / elapsed,
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
                if used + time.monotonic() - started > 14400 - 900:
                    run.log("live_budget_stop", completed_cells=len(results))
                    return
                order = (
                    ["historical", "selected"]
                    if repeat in [0, 3]
                    else ["selected", "historical"]
                )
                for label in order:
                    results.append(
                        cell(label, models[label], workload, 1001 + repeat, repeat)
                    )
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
