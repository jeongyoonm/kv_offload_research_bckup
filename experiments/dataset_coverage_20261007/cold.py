# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse recorded cold priming executions for training geometry coverage."""

import bisect
import json
import time
from collections import defaultdict

import audit
import run


def rows(extractor):
    cells = []
    for path in (run.ROOT / "cells").rglob("result.json"):
        result = json.loads(path.read_text())
        responses = result.get("responses", [])
        if responses:
            cells.append(
                (
                    min(r["started_ns"] for r in responses),
                    max(r["ended_ns"] for r in responses),
                    path,
                    result,
                    json.loads((path.parent / "family.json").read_text()),
                )
            )
    cells.sort(key=lambda x: x[0])
    starts = [c[0] for c in cells]
    all_events = []
    for path in (run.ROOT / "server/timing").glob("engine-*.jsonl"):
        all_events.extend(json.loads(line) for line in path.read_text().splitlines())
    selected, metadata = defaultdict(list), {}
    for event in all_events:
        ids = event.get("selected", {})
        if event["event"] != "step_completion" or not ids:
            continue
        if not all("coverage-prime-" in key for key in ids):
            continue
        dispatch = event["dispatch_ns"]
        i = bisect.bisect_right(starts, dispatch)
        if i >= len(cells):
            continue
        begin, end, path, result, meta = cells[i]
        lower = cells[i - 1][1] if i else 0
        if (
            meta["split"] != "train"
            or meta["block"] != "geometry"
            or meta["repeat"] == 0
            or not result["accepted"]
            or dispatch <= lower
            or not all(
                f"coverage-prime-{meta['seed']}-{meta['repeat']}-" in key for key in ids
            )
        ):
            continue
        cached = event.get("cpu_cached_prefix_tokens")
        if cached is None or any(cached) or event.get("load_state") != [0, 0]:
            continue
        if len({tuple(shape) for shape in event.get("request_shapes", [])}) != 1:
            continue
        selected[path.parent].append(event)
        metadata[path.parent] = (meta, lower, begin)
    jit = audit.inference_jit_events()
    offset = time.time() - time.monotonic()
    result_rows = []
    audit_records = []
    for directory, events in selected.items():
        meta, lower, upper = metadata[directory]
        compile_events = [
            line
            for t, line in jit
            if lower / 1e9 + offset - 1 <= t <= upper / 1e9 + offset + 1
        ]
        if compile_events:
            audit_records.append(
                dict(
                    episode=str(directory),
                    rejected="priming_JIT",
                    events=compile_events,
                )
            )
            continue
        steps = {e["step"] for e in events}
        trace = [
            e
            for e in all_events
            if (
                (e["event"] == "step_completion" and e.get("step") in steps)
                or (
                    e["event"] == "dataset_forecast"
                    and e.get("predecessor_step") in steps
                )
                or (
                    e["event"] in ("kv_load_dispatched", "kv_all_rank_completion")
                    and lower < e.get("dispatch_ns", e["time_ns"]) < upper
                )
            )
        ]
        source = directory / "prime-engine.jsonl"
        source.write_text("".join(json.dumps(e) + "\n" for e in trace))
        for e in events:
            row, reason = extractor.extract(e, cached_only=False)
            if row is None:
                continue
            if any(
                not 0 < r[2] <= r[3] or r[3] - r[2] > 2000000
                for r in row["execution_ranks"]
            ):
                continue
            row.update(
                episode=str(directory),
                source_file=str(source),
                workload="cold_" + meta["id"],
                block="geometry",
                split="train",
                collection_phase="cold_priming",
                cpu_hit_percent=0,
            )
            result_rows.append(row)
    run.save(
        run.ROOT / "data/cold_priming_audit.json",
        dict(samples=len(result_rows), rejected=audit_records),
    )
    return result_rows
