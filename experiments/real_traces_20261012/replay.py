# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Natural evolving-cache trace replay and full-length streaming measurement."""

import argparse
import asyncio
import copy
import gzip
import hashlib
import json
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import aiohttp
import numpy as np
import regex as re

from experiments.load_only_overlap_20261011 import cache_client as client
from experiments.load_only_overlap_20261011 import evaluate as old
from experiments.real_traces_20261012.prepare import OUT, digest
from vllm.v1.kv_offload.early_readiness import validate_experiment_options

CAMPAIGN = Path("/mnt/localnvme/jeongyoon/real_traces_20261012")
STATE_PATH = CAMPAIGN / "frozen_predictor.json"
STATE_SHA256 = "f159f32d226894457fb56b3195c47d4ff6e2a5d982d8d600b1970ea000294f96"
BASE = "http://127.0.0.1:18472"


def flush_timeline(root):
    """Require every worker; an unused baseline engine emits no timeline."""
    directory = root / "server/load_timeline"
    pids = set(
        subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).split()
    )
    assert len(pids) == 8, pids
    trigger = directory / "flush.request"
    trigger.touch()
    stamp = trigger.stat().st_mtime_ns
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        paths = list(directory.glob(f"*.{stamp}.jsonl"))
        observed = {p.name.split(".")[0] for p in paths}
        complete = {
            pid
            for pid in observed
            if (directory / f"{pid}.flushed").exists()
            and (directory / f"{pid}.flushed").read_text() == str(stamp)
        }
        if pids <= complete and observed <= complete:
            time.sleep(0.2)
            paths = [
                path
                for path in directory.glob(f"*.{stamp}.jsonl")
                if (directory / f"{path.name.split('.')[0]}.flushed").exists()
                and (directory / f"{path.name.split('.')[0]}.flushed").read_text()
                == str(stamp)
            ]
            return [
                json.loads(line)
                for path in paths
                for line in path.read_text().splitlines()
            ]
        time.sleep(0.05)
    raise RuntimeError("transfer timeline omitted one or more TP ranks")


def configure(root, arm):
    run = old.base.collector
    old.base.module.ROOT = old.base.module.driver.ROOT = run.ROOT = root
    run.BASE = BASE
    assert digest(STATE_PATH) == STATE_SHA256
    state = json.loads(STATE_PATH.read_text())
    if arm == "baseline":
        # Preserve the baseline's ordinary connector and scheduler defaults.
        return run, state, validate_experiment_options({})
    policy = old.policy("full", 4)
    # Per-token decode creates many records; counters remain enabled. Sparse
    # diagnostic cells can turn these markers on without altering admission.
    policy.update(late_admission_markers=False, zero_token_stage_markers=False)
    policy = validate_experiment_options(policy)
    run.options = lambda active=False: copy.deepcopy(policy)
    run.configure(True, state)
    return run, state, policy


async def request(session, sem, row, name, scheduled, sink):
    result = dict(
        request=name,
        index=row["index"],
        session=row.get("session"),
        scheduled_ns=scheduled,
        input_tokens=len(row["prompt"]),
        target_output_tokens=row["output_tokens"],
    )
    try:
        async with sem:
            result["send_ns"] = time.monotonic_ns()
            payload = dict(
                model="glm53-compat",
                prompt=row["prompt"],
                request_id=name,
                max_tokens=row["output_tokens"],
                min_tokens=row["output_tokens"],
                temperature=0,
                seed=0,
                ignore_eos=True,
                stream=True,
                stream_options=dict(include_usage=True),
            )
            output_hash = hashlib.sha256()
            chunks = 0
            async with session.post(BASE + "/v1/completions", json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"HTTP {response.status}: {(await response.text())[:1000]}"
                    )
                async for raw in response.content:
                    if not raw.startswith(b"data: "):
                        continue
                    data = raw[6:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    for choice in event.get("choices", []):
                        result.setdefault("first_ns", time.monotonic_ns())
                        if choice.get("finish_reason") is None:
                            chunks += 1
                        output_hash.update(choice.get("text", "").encode())
                        if choice.get("finish_reason"):
                            result["finish_reason"] = choice["finish_reason"]
                    if event.get("usage"):
                        result["usage"] = event["usage"]
            result["end_ns"] = time.monotonic_ns()
            result["output_sha256"] = output_hash.hexdigest()
            result["stream_chunks"] = chunks
            if result.get("usage", {}).get("prompt_tokens") != len(row["prompt"]):
                raise RuntimeError("input token count mismatch")
            if result.get("usage", {}).get("completion_tokens") != row["output_tokens"]:
                raise RuntimeError("output token count mismatch")
            if "first_ns" not in result:
                raise RuntimeError("missing first token marker")
            result["accepted"] = True
    except Exception as error:
        result.update(accepted=False, error=repr(error), end_ns=time.monotonic_ns())
    sink.write(json.dumps(result) + "\n")
    sink.flush()
    return result


async def issue(data, spec, directory):
    timeout = aiohttp.ClientTimeout(total=1800, sock_read=600)
    sem = asyncio.Semaphore(spec["concurrency"])
    start = time.monotonic_ns()
    with (directory / "responses.jsonl").open("w") as sink:
        async with aiohttp.ClientSession(
            timeout=timeout, connector=aiohttp.TCPConnector(limit=0)
        ) as session:
            if data["kind"] == "mooncake":
                rows = data["requests"][: spec["requests"]]
                origin = rows[0]["arrival_ms"]
                if spec.get("saturated"):
                    arrivals = [start] * len(rows)
                else:
                    arrivals = [
                        start
                        + int((r["arrival_ms"] - origin) / spec["time_scale"] * 1e6)
                        for r in rows
                    ]

                async def paced(row, arrival):
                    wait = (arrival - time.monotonic_ns()) / 1e9
                    if wait > 0:
                        await asyncio.sleep(wait)
                    return await request(
                        session,
                        sem,
                        row,
                        f"{directory.name}-{row['index']}",
                        arrival,
                        sink,
                    )

                results = await asyncio.gather(
                    *(paced(row, arrival) for row, arrival in zip(rows, arrivals))
                )
                expected = len(rows)
                offered_end = max(arrivals)
            else:
                sessions = data["sessions"][: spec["sessions"]]

                async def conversation(item):
                    results = []
                    for row in item["requests"][: spec.get("turns", 1000)]:
                        ready = time.monotonic_ns()
                        response = await request(
                            session,
                            sem,
                            row,
                            f"{directory.name}-s{item['session']}-t{row['index']}",
                            ready,
                            sink,
                        )
                        results.append(response)
                        if not response["accepted"]:
                            break
                    return results

                groups = await asyncio.gather(
                    *(conversation(item) for item in sessions)
                )
                results = [r for group in groups for r in group]
                expected = sum(
                    len(s["requests"][: spec.get("turns", 1000)]) for s in sessions
                )
                offered_end = None
    return results, start, time.monotonic_ns(), expected, offered_end


def distribution(values):
    return (
        dict(
            mean=statistics.mean(values),
            p50=float(np.quantile(values, 0.5)),
            p95=float(np.quantile(values, 0.95)),
            p99=float(np.quantile(values, 0.99)),
            min=min(values),
            max=max(values),
        )
        if values
        else None
    )


def token_sources(run):
    result = {}
    for line in run.textcall("/metrics").splitlines():
        if line.startswith("vllm:prompt_tokens_by_source_total{"):
            source = re.search(r'source="([^"]+)"', line).group(1)
            result[source] = result.get(source, 0) + float(line.split()[-1])
    return result


def summarize(responses, start, end, expected, offered_end, events):
    valid = [r for r in responses if r["accepted"]]
    seconds = (end - start) / 1e9
    jobs = defaultdict(dict)
    stores = defaultdict(dict)
    for e in events:
        if e["event"] == "job_observed_done":
            jobs[e["job_id"]][e["pid"]] = e
        if e["event"] == "store_observed_done":
            stores[e["job_id"]][e["pid"]] = e
    loaded = {e.get("request", "") for ranks in jobs.values() for e in ranks.values()}
    outcomes = Counter()
    for r in valid:
        # vLLM wraps the provided ID with a completion prefix and unique suffix.
        pattern = re.compile(r"(?:^|-)" + re.escape(r["request"]) + r"(?:$|-)")
        cpu = any(pattern.search(name) for name in loaded)
        cached = r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        outcomes["cpu_hit" if cpu else "gpu_hit" if cached else "miss"] += 1
    input_total = sum(r["usage"]["prompt_tokens"] for r in valid)
    output_total = sum(r["usage"]["completion_tokens"] for r in valid)
    cached_total = sum(
        r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        for r in valid
    )
    return dict(
        accepted=len(valid) == expected
        and all(len(ranks) == 8 for ranks in jobs.values()),
        requests=len(valid),
        expected_requests=expected,
        errors=[r for r in responses if not r["accepted"]],
        seconds=seconds,
        throughput_rps=len(valid) / seconds,
        output_tokens_per_second=output_total / seconds,
        input_tokens=input_total,
        output_tokens=output_total,
        cached_tokens=cached_total,
        cached_input_fraction=cached_total / input_total if input_total else 0,
        ttft_ms=distribution([(r["first_ns"] - r["send_ns"]) / 1e6 for r in valid]),
        scheduled_ttft_ms=distribution(
            [(r["first_ns"] - r["scheduled_ns"]) / 1e6 for r in valid]
        ),
        driver_queue_ms=distribution(
            [(r["send_ns"] - r["scheduled_ns"]) / 1e6 for r in valid]
        ),
        latency_ms=distribution([(r["end_ns"] - r["send_ns"]) / 1e6 for r in valid]),
        realized_outcomes=dict(outcomes),
        outcome_definition=(
            "Any CPU load: CPU hit (may also have GPU prefix). "
            "Otherwise cached>0: GPU hit. Otherwise miss."
        ),
        h2d_bytes_all_ranks=sum(
            e["transfer_bytes"] for ranks in jobs.values() for e in ranks.values()
        ),
        d2h_bytes_all_ranks=sum(
            e.get("transfer_bytes") or 0
            for ranks in stores.values()
            for e in ranks.values()
        ),
        h2d_jobs=len(jobs),
        h2d_rank_coverage=dict(Counter(len(ranks) for ranks in jobs.values())),
        d2h_jobs=len(stores),
        d2h_rank_coverage=dict(Counter(len(ranks) for ranks in stores.values())),
        offered_window_seconds=(offered_end - start) / 1e9 if offered_end else None,
        offered_rps=(expected / ((offered_end - start) / 1e9))
        if offered_end and offered_end > start
        else None,
        drain_after_last_arrival_seconds=max(0, end - offered_end) / 1e9
        if offered_end
        else None,
        completion_within_arrival_window=sum(r["end_ns"] <= offered_end for r in valid)
        if offered_end
        else None,
        start_ns=start,
        end_ns=end,
    )


def cell(root, arm, spec):
    directory = root / "cells" / spec["id"]
    directory.mkdir(parents=True, exist_ok=False)
    data_path = OUT / (spec["dataset"] + ".json.gz")
    with gzip.open(data_path, "rt") as f:
        data = json.load(f)
    run, state, policy = configure(root, arm)
    client.drain()
    run.reset(True)
    flush_timeline(root)
    before = client.metrics()
    sources_before = token_sources(run)
    trace_before = set((root / "server/timing/late_admission").glob("*.json"))
    run.save(
        directory / "spec.json",
        dict(
            spec,
            arm=arm,
            dataset_sha256=digest(data_path),
            policy=policy,
            predictor_sha256=digest(STATE_PATH),
        ),
    )
    run.log("trace_cell_started", arm=arm, spec=spec)
    responses, start, end, expected, offered_end = asyncio.run(
        issue(data, spec, directory)
    )
    client.drain()
    # Export policy diagnostics only after the request/transfer stream is idle.
    frozen = None
    if arm == "full":
        run.configure(True)
        exported = run.call(
            "/experimental/kv-load-scheduling",
            dict(
                early_load_completion=True,
                experiment_options=policy,
                seal_estimator=dict(action="export"),
            ),
        )["seal_estimator"]["state"]
        frozen = exported == state
    events = flush_timeline(root)
    after = client.metrics()
    sources_after = token_sources(run)
    result = summarize(responses, start, end, expected, offered_end, events)
    protocol_counts = Counter()
    for path in (
        set((root / "server/timing/late_admission").glob("*.json")) - trace_before
    ):
        obj = json.loads(path.read_text())
        if obj.get("role") == "engine":
            protocol_counts.update(obj.get("counts", {}))
        assert not obj.get("overflow"), path
    result.update(
        arm=arm,
        spec=spec,
        dataset_sha256=digest(data_path),
        predictor_unchanged=frozen,
        protocol_counts=dict(protocol_counts),
        metric_deltas={
            k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)
        },
        token_source_deltas={
            k: sources_after.get(k, 0) - sources_before.get(k, 0)
            for k in set(sources_before) | set(sources_after)
        },
        gpu_cache_bytes_per_rank=96 * 1024**3,
        cpu_cache_bytes_server=2130300354560,
        natural_cache_evolution=True,
        cache_reset_only_at_episode_start=True,
    )
    if arm == "full":
        result["accepted"] &= frozen
    run.save(directory / "result.json", result)
    with (directory / "transfer_events.jsonl").open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    run.log(
        "trace_cell_finished",
        arm=arm,
        id=spec["id"],
        accepted=result["accepted"],
        throughput_rps=result["throughput_rps"],
        ttft=result["ttft_ms"],
        outcomes=result["realized_outcomes"],
    )
    if not result["accepted"]:
        raise RuntimeError(f"Rejected trace cell: {directory}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--arm", choices=["baseline", "full"], required=True)
    p.add_argument("--specs", type=Path, required=True)
    args = p.parse_args()
    specs = json.loads(args.specs.read_text())
    for spec in specs:
        if (args.root / "cells" / spec["id"] / "result.json").exists():
            continue
        cell(args.root, args.arm, spec)


if __name__ == "__main__":
    main()
