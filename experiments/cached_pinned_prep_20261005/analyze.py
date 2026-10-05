# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize serving cells and verify all-rank cached/pinned path coverage."""

import importlib.util
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path(
    os.environ.get(
        "CACHED_PINNED_ARTIFACT_ROOT", "/mnt/localnvme/glm53_cached_pinned_20261005"
    )
)


def read(path):
    return json.loads(path.read_text())


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def summary(values):
    return (
        dict(
            count=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            min=min(values),
            max=max(values),
        )
        if values
        else dict(count=0)
    )


def main():
    analyzer = module(
        "prep_poll_analysis", "experiments/poll_prediction_20261003/analyze.py"
    )
    cpu = module(
        "prep_cpu_analysis", "experiments/coalesced_notify_20261004/analyze_ab.py"
    )
    cells = []
    workers = [
        read(p)
        for p in (ROOT / "canary/gpu/server/timing/late_admission").glob("*.json")
        if read(p).get("role") == "worker"
    ]
    stage_exports = [
        read(p) for p in (ROOT / "canary/gpu/server/timing/queue_stages").glob("*.json")
    ]
    for path in sorted((ROOT / "canary/gpu").glob("p*/*/result.json")):
        if not read(path).get("accepted"):
            continue
        directory = path.parent
        item = analyzer.analyze(directory)
        assert item["accepted"], directory
        events = [
            json.loads(line)
            for line in (directory / "engine.jsonl").read_text().splitlines()
        ]
        completed = {e["step"]: e for e in events if e["event"] == "step_completion"}
        dispatch = {
            e["step"]: e
            for e in events
            if e["event"] == "dispatch" and e.get("selected")
        }
        prefill = {
            s
            for s, e in dispatch.items()
            if any(n > 1 for n, _ in e.get("request_shapes", []))
        }
        setup, handoff, gpu_start = [], [], []
        for step in prefill:
            d = dispatch[step]
            c = completed.get(step, {})
            ranks, execution = c.get("ranks", []), c.get("execution_ranks", [])
            if len(ranks) == 8 and all(r and r[2] >= r[1] > 0 for r in ranks):
                setup.append(max(r[2] - r[1] for r in ranks) / 1e6)
                handoff.append((max(r[1] for r in ranks) - d["dispatch_ns"]) / 1e6)
                if len(execution) == 8 and all(r and r[3] for r in execution):
                    gpu_start.append(
                        (max(r[3] for r in execution) - d["dispatch_ns"]) / 1e6
                    )
        late = read(directory / "late.json")
        phase_events = defaultdict(dict)
        for export in stage_exports:
            if export["generation"] != late["generation"] or export["rank"] < 0:
                continue
            assert not export["overflow"]
            for event in export["events"]:
                if event.get("step") in prefill:
                    phase_events[(event["step"], export["rank"])][event["event"]] = (
                        event["time_ns"]
                    )
        phases = {}
        for name, begin, end in (
            ("request_state", "worker_entry", "request_state_updated"),
            ("inputs", "request_state_updated", "input_preparation_return"),
            ("attention", "input_preparation_return", "attention_metadata_ready"),
            ("pre_forward", "attention_metadata_ready", "forward_submission_begin"),
        ):
            values = []
            for step in prefill:
                spans = []
                for rank in range(8):
                    event = phase_events[(step, rank)]
                    if begin in event and end in event:
                        spans.append((event[end] - event[begin]) / 1e6)
                if len(spans) == 8:
                    values.append(max(spans))
            phases[name] = summary(values)
        generations = [w for w in workers if w["generation"] == late["generation"]]
        assert len(generations) == 8 and {w["rank"] for w in generations} == set(
            range(8)
        ), directory
        coverage = {
            w["rank"]: dict(
                cached=sum(
                    e.get("cached_requests", 0)
                    for e in w["records"]
                    if e["event"] == "worker_entry"
                ),
                pinned=sum(
                    e.get("pinned_tokens", 0)
                    for e in w["records"]
                    if e["event"] == "worker_entry"
                ),
                compact=sum(
                    e.get("compact_tokens", 0)
                    for e in w["records"]
                    if e["event"] == "worker_entry"
                ),
            )
            for w in generations
        }
        mode = directory.name.split("_", 1)[1].split("_warmretry", 1)[0]
        reference_count = late["counts"].get("cached_request_references_references", 0)
        if mode in ("references", "cached_pinned", "compact_pinned"):
            assert reference_count > 0 and all(
                v["cached"] >= reference_count for v in coverage.values()
            ), directory
        if mode in ("cached_pinned", "compact_pinned"):
            assert all(v["pinned"] > 0 for v in coverage.values()), directory
        else:
            assert all(v["pinned"] == 0 for v in coverage.values()), directory
        if mode == "compact_pinned":
            assert all(v["compact"] >= reference_count for v in coverage.values()), (
                directory
            )
        else:
            assert all(v["compact"] == 0 for v in coverage.values()), directory
        sampled = read(directory / "server_cpu.json")["samples"]
        cpu_stats = cpu.cpu_deltas(
            directory, sampled[0]["time_ns"], sampled[-1]["time_ns"]
        )
        item.update(
            workload=directory.parent.name,
            mode=mode,
            prefill_setup_ms=summary(setup),
            prefill_host_phases_max_rank_ms=phases,
            prefill_dispatch_to_entry_ms=summary(handoff),
            prefill_dispatch_to_gpu_ms=summary(gpu_start),
            reference_requests=reference_count,
            compute_rounds=len(dispatch),
            worker_coverage=coverage,
            server_cpu=cpu_stats,
        )
        cells.append(item)
    groups = defaultdict(list)
    for cell in cells:
        groups[(cell["workload"], cell["mode"])].append(cell)
    aggregates = []
    for (workload, mode), rows in sorted(groups.items()):
        aggregates.append(
            dict(
                workload=workload,
                mode=mode,
                repeats=len(rows),
                throughput=statistics.mean(r["throughput"] for r in rows),
                mean_ttft_ms=statistics.mean(r["mean_ttft_ms"] for r in rows),
                p95_ttft_ms=statistics.mean(r["p95_ttft_ms"] for r in rows),
                compute_rounds=statistics.mean(r["compute_rounds"] for r in rows),
                prefill_setup_median_ms=statistics.mean(
                    r["prefill_setup_ms"]["median"] for r in rows
                ),
                prefill_handoff_median_ms=statistics.mean(
                    r["prefill_dispatch_to_entry_ms"]["median"] for r in rows
                ),
                prefill_dispatch_to_gpu_median_ms=statistics.mean(
                    r["prefill_dispatch_to_gpu_ms"]["median"] for r in rows
                ),
                referenced_requests=sum(r["reference_requests"] for r in rows),
            )
        )
    (ROOT / "cells.json").write_text(json.dumps(cells, indent=2) + "\n")
    (ROOT / "aggregate.json").write_text(json.dumps(aggregates, indent=2) + "\n")
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
