# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact immutable prompt replay with causal sessions and explicit pacing."""

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import aiohttp
import numpy as np

from experiments.agentic_1m_20261012.common import (
    BASE,
    DATA,
    LIMIT,
    ROOT,
    STATE,
    STATE_SHA,
    digest,
    log,
    save,
)
from experiments.load_only_overlap_20261011 import cache_client as client
from experiments.real_traces_20261012 import replay as prior


class NoLimit:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def setup():
    prior.BASE = BASE
    prior.CAMPAIGN = ROOT
    prior.STATE_PATH = STATE
    prior.STATE_SHA256 = STATE_SHA


async def issue(data, spec, directory):
    """Keep request content fixed; pacing adds only an external interval."""
    blob = np.memmap(data["token_file"], dtype="<u4", mode="r")
    sessions = data["sessions"][: spec.get("sessions", len(data["sessions"]))]
    sem = asyncio.Semaphore(spec["concurrency"])
    timeout = aiohttp.ClientTimeout(total=3600, sock_read=1800)
    interval = float(spec.get("min_interval_seconds", 0))
    start = time.monotonic_ns()
    count = sum(len(s["requests"][: spec.get("turns", 100000)]) for s in sessions)
    completed = 0
    with (directory / "responses.jsonl").open("w") as sink:
        async with aiohttp.ClientSession(
            timeout=timeout, connector=aiohttp.TCPConnector(limit=0)
        ) as http:

            async def conversation(item):
                nonlocal completed
                responses = []
                for metadata in item["requests"][: spec.get("turns", 100000)]:
                    if interval:
                        await asyncio.sleep(interval)
                    ready = time.monotonic_ns()
                    async with sem:
                        prepare_start = time.monotonic_ns()
                        offset = metadata["offset"]
                        length = metadata["input_tokens"]
                        prompt = blob[offset : offset + length].tolist()
                        row = dict(metadata, prompt=prompt)
                        response = await prior.request(
                            http,
                            NoLimit(),
                            row,
                            f"{directory.name}-s{item['session']}-t{metadata['index']}",
                            ready,
                            sink,
                        )
                        response["client_materialization_ms"] = (
                            time.monotonic_ns() - prepare_start
                        ) / 1e6 - (response["end_ns"] - response["send_ns"]) / 1e6
                        del row, prompt
                    completed += 1
                    responses.append(response)
                    if completed % 64 == 0:
                        save(
                            directory / "progress.json",
                            dict(
                                completed=completed,
                                expected=count,
                                elapsed_seconds=(time.monotonic_ns() - start) / 1e9,
                            ),
                        )
                    if not response["accepted"]:
                        break
                return responses

            groups = await asyncio.gather(*(conversation(s) for s in sessions))
    return [r for group in groups for r in group], start, time.monotonic_ns(), count


def cell(root, arm, spec):
    setup()
    directory = root / "cells" / spec["id"]
    directory.mkdir(parents=True, exist_ok=False)
    path = DATA / (spec["dataset"] + ".json")
    data = json.loads(path.read_text())
    assert data["max_model_len"] == LIMIT
    run, state, policy = prior.configure(root, arm)
    client.drain()
    run.reset(spec.get("reset_external", True))
    prior.flush_timeline(root)
    before = client.metrics()
    sources_before = prior.token_sources(run)
    trace_before = set((root / "server/timing/late_admission").glob("*.json"))
    save(
        directory / "spec.json",
        dict(
            spec,
            arm=arm,
            policy=policy,
            dataset_sha256=digest(path),
            token_sha256=data["token_sha256"],
            predictor_sha256=digest(STATE),
            max_model_len=LIMIT,
        ),
    )
    log("cell_started", arm=arm, root=str(root), spec=spec)
    responses, start, end, expected = asyncio.run(issue(data, spec, directory))
    client.drain()
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
    events = prior.flush_timeline(root)
    after = client.metrics()
    sources_after = prior.token_sources(run)
    counts = Counter()
    for p in set((root / "server/timing/late_admission").glob("*.json")) - trace_before:
        obj = json.loads(p.read_text())
        assert not obj.get("overflow"), p
        if obj.get("role") == "engine":
            counts.update(obj.get("counts", {}))
    result = prior.summarize(responses, start, end, expected, None, events)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
    checks = dict(
        input_accounting=delta.get("vllm:request_prompt_tokens_sum")
        == result["input_tokens"],
        output_accounting=delta.get("vllm:generation_tokens_total")
        == result["output_tokens"],
        exactly_once=delta.get("vllm:request_success_total") == expected,
        h2d_all_ranks=all(int(k) == 8 for k in result["h2d_rank_coverage"]),
        d2h_all_ranks=all(int(k) == 8 for k in result["d2h_rank_coverage"]),
        predictor_frozen=arm == "baseline" or frozen,
    )
    result.update(
        arm=arm,
        spec=spec,
        protocol_counts=dict(counts),
        checks=checks,
        predictor_unchanged=frozen,
        metric_deltas=delta,
        token_source_deltas={
            k: sources_after.get(k, 0) - sources_before.get(k, 0)
            for k in set(sources_after) | set(sources_before)
        },
        dataset_sha256=digest(path),
        token_sha256=data["token_sha256"],
        max_model_len=LIMIT,
        logical_sessions=spec.get("sessions", len(data["sessions"])),
        gpu_cache_bytes_per_rank=96 * 1024**3,
        cpu_cache_bytes_server=2130300354560,
        natural_cache_evolution=True,
        client_materialization_ms=prior.distribution(
            [r["client_materialization_ms"] for r in responses if r["accepted"]]
        ),
    )
    result["accepted"] &= all(checks.values())
    save(directory / "result.json", result)
    with (directory / "transfer_events.jsonl").open("w") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")
    log(
        "cell_finished",
        arm=arm,
        id=spec["id"],
        accepted=result["accepted"],
        seconds=result["seconds"],
        rps=result["throughput_rps"],
        outcomes=result["realized_outcomes"],
        checks=checks,
    )
    if not result["accepted"]:
        raise RuntimeError(f"Cell rejected: {directory}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=["baseline", "full"], required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    cell(args.root, args.arm, json.loads(args.spec.read_text()))


if __name__ == "__main__":
    main()
