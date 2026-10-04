# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched notification A/B: include all server processes and latency tails."""

import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_coalesced_notify_20261004")


def load(path):
    return json.loads(path.read_text())


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def stats(values):
    return (
        dict(
            count=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            p95=float(np.quantile(values, 0.95)),
            p99=float(np.quantile(values, 0.99)),
            min=min(values),
            max=max(values),
        )
        if values
        else dict(count=0)
    )


def cpu_deltas(directory, begin, end):
    data = load(directory / "server_cpu.json")
    samples = data["samples"]
    identities = data["identities"]
    deltas = {}
    for pid, identity in identities.items():
        points = [(s["time_ns"], s["cpu_seconds"][pid]) for s in samples]
        times, cpu = zip(*points)
        delta = float(np.interp(end, times, cpu) - np.interp(begin, times, cpu))
        text = identity["name"] + " " + " ".join(identity["cmdline"])
        role = (
            "worker"
            if "Worker" in text
            else ("engine" if "EngineCore" in text else "other")
        )
        deltas[pid] = dict(role=role, seconds=delta, **identity)
    return dict(
        begin_ns=begin,
        end_ns=end,
        duration_seconds=(end - begin) / 1e9,
        total_seconds=sum(d["seconds"] for d in deltas.values()),
        engine_seconds=sum(
            d["seconds"] for d in deltas.values() if d["role"] == "engine"
        ),
        worker_seconds=sum(
            d["seconds"] for d in deltas.values() if d["role"] == "worker"
        ),
        processes=deltas,
        errors=data["errors"],
        sample_interval_ms=data["interval_ms"],
    )


def latency(directory):
    late = load(directory / "late.json")["records"]
    eligibility = {e["step"]: e for e in late if e["event"] == "seal_eligibility"}
    acks = defaultdict(list)
    for event in late:
        if event["event"] == "preparation_ack_transition":
            acks[event["step"]].append(event)
    engine = [
        json.loads(line)
        for line in (directory / "engine.jsonl").read_text().splitlines()
    ]
    kv = {}
    for event in engine:
        if event["event"] in ("kv_all_rank_completion", "kv_ready_observed"):
            request = event.get("request", event.get("early_promoted_request"))
            if request:
                kv[request] = event.get("completion_ns", event["time_ns"])
    target_lags, candidate_lags, samples = [], [], []
    for event in late:
        if event["event"] != "membership_sealed":
            continue
        t = event.get("transport", {})
        if not t.get("base_ready_ns"):
            continue
        candidates = eligibility[event["step"]]["candidates"]
        ready = []
        for slot, request in enumerate(candidates):
            ack = next(
                (
                    e["time_ns"]
                    for e in acks[event["step"]]
                    if e["candidates"][slot] == [1] * 8
                ),
                math.inf,
            )
            ready.append(max(t["base_ready_ns"], kv.get(request, math.inf), ack))
            if request in event.get("activated", {}) and math.isfinite(ready[-1]):
                candidate_lags.append((t["seal_ns"] - ready[-1]) / 1e6)
        target = t.get("ready_target", 0)
        if (
            target
            and len(ready) >= target
            and sorted(ready)[target - 1] <= t["seal_ns"]
        ):
            lag = (t["seal_ns"] - sorted(ready)[target - 1]) / 1e6
            target_lags.append(lag)
            samples.append(
                dict(
                    step=event["step"],
                    reason=t.get("seal_reason"),
                    target_ready_to_seal_ms=lag,
                    final_wait_phase=(t.get("adaptive_waits") or [{}])[-1].get("phase"),
                )
            )
    return dict(
        target_ready_to_seal_ms=target_lags,
        admitted_candidate_ready_to_seal_ms=candidate_lags,
        samples=samples,
    )


def main():
    synthetic = load(ROOT / "cpu-comparison/cells.json")
    cpu_groups = []
    for skew in (0, 1, 2):
        for mode in ("bridge", "coalesced"):
            rows = [
                r for r in synthetic if r["rank_skew_ms"] == skew and r["mode"] == mode
            ]
            cpu_groups.append(
                dict(
                    rank_skew_ms=skew,
                    mode=mode,
                    **{
                        k: stats([r[k] for r in rows])
                        for k in (
                            "engine_cpu_ms",
                            "worker_cpu_ms",
                            "total_cpu_ms",
                            "ready_to_seal_ms",
                            "coarse_wakes",
                            "bridge_scans",
                        )
                    },
                )
            )
    analyzer = module(
        "poll_analysis", "experiments/poll_prediction_20261003/analyze.py"
    )
    diagnostics = module(
        "full_analysis", "experiments/full_predicted_wait_20261004/summarize.py"
    ).diagnostics
    cells = []
    for result in sorted((ROOT / "ab/gpu").glob("p*/*/result.json")):
        directory = result.parent
        a = analyzer.analyze(directory)
        events = [
            json.loads(line)
            for line in (directory / "engine.jsonl").read_text().splitlines()
        ]
        ids = [r["request_id"] for r in load(directory / "usage.json")]

        def matches(request, ids=ids):
            return any(r in request for r in ids)

        starts, steps = [], set()
        for event in events:
            if event["event"] == "kv_load_dispatched" and matches(event["request"]):
                starts.append(event["time_ns"])
            elif event["event"] == "dispatch" and any(
                matches(r) for r in event.get("selected", {})
            ):
                starts.append(event["schedule_begin_ns"])
                steps.add(event["step"])
        ends = [
            r[5]
            for event in events
            if event["event"] == "step_completion" and event["step"] in steps
            for r in event.get("ranks", [])
            if r and r[5]
        ]
        # Include response delivery and API cleanup in a bounded 25 ms margin.
        cpu = cpu_deltas(directory, min(starts), max(ends) + 25_000_000)
        samples = load(directory / "server_cpu.json")["samples"]
        envelope = cpu_deltas(directory, samples[0]["time_ns"], samples[-1]["time_ns"])
        item = dict(
            workload=directory.parent.name,
            mode=directory.name.split("_", 1)[1].split("_warmretry", 1)[0],
            repeat=int(directory.name.split("_", 1)[0][1:]),
            **a,
            **diagnostics(directory, a),
            latency=latency(directory),
            server_cpu=cpu,
            server_cpu_envelope=envelope,
        )
        item["server_cpu_ms_per_request"] = cpu["total_seconds"] * 1000 / len(ids)
        item["engine_cpu_ms_per_request"] = cpu["engine_seconds"] * 1000 / len(ids)
        item["worker_cpu_ms_per_request"] = cpu["worker_seconds"] * 1000 / len(ids)
        item["worker_processes"] = sum(
            d["role"] == "worker" for d in cpu["processes"].values()
        )
        assert item["worker_processes"] == 8, directory
        assert not cpu["errors"], directory
        cells.append(item)
    aggregate = []
    for workload in sorted({c["workload"] for c in cells}):
        for mode in ("bridge", "coalesced"):
            rows = [
                c
                for c in cells
                if c["workload"] == workload and c["mode"] == mode and c["accepted"]
            ]
            if not rows:
                continue
            item = dict(workload=workload, mode=mode, repeats=len(rows))
            for key in (
                "throughput",
                "mean_ttft_ms",
                "p95_ttft_ms",
                "p99_ttft_ms",
                "compute_calls",
                "late_admissions",
                "server_cpu_ms_per_request",
                "engine_cpu_ms_per_request",
                "worker_cpu_ms_per_request",
                "coarse_progress_wakes",
                "coarse_timer_wakes",
                "bridge_scans",
                "fine_waits",
            ):
                item[key] = stats([r[key] for r in rows])
            item["target_ready_to_seal_ms"] = stats(
                [lag for r in rows for lag in r["latency"]["target_ready_to_seal_ms"]]
            )
            item["target_ready_to_seal_by_final_wait_phase_ms"] = {
                phase: stats(
                    [
                        s["target_ready_to_seal_ms"]
                        for r in rows
                        for s in r["latency"]["samples"]
                        if s["final_wait_phase"] == phase
                    ]
                )
                for phase in ("coarse", "fine", None)
            }
            recovered = sum(r["recovered_compute_opportunities"] for r in rows)
            total = sum(r["kv_ready_during_compute"] for r in rows)
            item["admission_ratio"] = dict(
                recovered=recovered,
                total=total,
                percent=100 * recovered / total if total else None,
            )
            aggregate.append(item)
    paired = []
    for workload in sorted({c["workload"] for c in cells}):
        for repeat in range(3):
            pair = {
                c["mode"]: c
                for c in cells
                if c["accepted"] and c["workload"] == workload and c["repeat"] == repeat
            }
            if set(pair) != {"bridge", "coalesced"}:
                continue
            item = dict(workload=workload, repeat=repeat)
            for key in (
                "throughput",
                "mean_ttft_ms",
                "p95_ttft_ms",
                "server_cpu_ms_per_request",
                "coarse_progress_wakes",
            ):
                base, new = pair["bridge"][key], pair["coalesced"][key]
                item[key] = dict(
                    bridge=base,
                    coalesced=new,
                    change_percent=100 * (new / base - 1) if base else None,
                )
            paired.append(item)
    out = ROOT / "ab/summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cpu.json").write_text(json.dumps(cpu_groups, indent=2) + "\n")
    (out / "cells.json").write_text(json.dumps(cells, indent=2) + "\n")
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    (out / "paired.json").write_text(json.dumps(paired, indent=2) + "\n")
    print(json.dumps(dict(cpu=cpu_groups, gpu=aggregate), indent=2))


if __name__ == "__main__":
    main()
