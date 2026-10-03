# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize replay performance, bridge cost, and individual exposed waits."""

import argparse
import csv
import importlib.util
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_full_predicted_wait_20261004")
spec = importlib.util.spec_from_file_location(
    "poll_analyze", SOURCE / "experiments/poll_prediction_20261003/analyze.py"
)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def load(path):
    return json.loads(path.read_text())


def diagnostics(directory, analysis):
    records = load(directory / "late.json")["records"]
    seals = {e["step"]: e for e in records if e["event"] == "membership_sealed"}
    eligibility = {e["step"]: e for e in records if e["event"] == "seal_eligibility"}
    windows = {w["step"]: w for w in analysis["windows"]}
    acks = defaultdict(list)
    for event in records:
        if event["event"] == "preparation_ack_transition":
            acks[event["step"]].append(event)
    kv = {}
    for line in (directory / "engine.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event"] in ("kv_all_rank_completion", "kv_ready_observed"):
            request = event.get("request", event.get("early_promoted_request"))
            if request:
                kv[request] = event.get("completion_ns", event["time_ns"])
    result = dict(
        bridge_cpu_ms=0.0,
        bridge_scans=0,
        bridge_notifications=0,
        coarse_waits=0,
        coarse_progress_wakes=0,
        coarse_timer_wakes=0,
        fine_waits=0,
        max_requested_coarse_ms=0.0,
        max_actual_coarse_ms=0.0,
        response_lag_estimate_ms=0.0,
        sleep_after_stop_estimate_ms=0.0,
        coarse_sleep_after_stop_estimate_ms=0.0,
        fine_sleep_after_stop_estimate_ms=0.0,
        covered_stop_windows=0,
        unknown_stop_windows=[],
    )
    for step, event in seals.items():
        transport = event.get("transport", {})
        result["bridge_cpu_ms"] += transport.get("progress_bridge_cpu_ns", 0) / 1e6
        result["bridge_scans"] += transport.get("progress_bridge_scans", 0)
        result["bridge_notifications"] += transport.get(
            "progress_bridge_notifications", 0
        )
        waits = transport.get("adaptive_waits", [])
        for wait in waits:
            if wait["phase"] == "coarse":
                result["coarse_waits"] += 1
                result["coarse_" + wait["reason"] + "_wakes"] += 1
                result["max_requested_coarse_ms"] = max(
                    result["max_requested_coarse_ms"], wait["requested_ns"] / 1e6
                )
                result["max_actual_coarse_ms"] = max(
                    result["max_actual_coarse_ms"],
                    (wait["end_ns"] - wait["begin_ns"]) / 1e6,
                )
            else:
                result["fine_waits"] += 1
        if not waits:
            continue
        window = windows.get(step, {})
        if "exposed_lower_ms" not in window:
            result["unknown_stop_windows"].append(step)
            continue
        assert not transport.get("cancelled"), directory
        base = transport["base_ready_ns"]
        end = window.get("a_end_upper_ns", base)
        candidates = eligibility[step]["candidates"]
        target = transport["ready_target"]
        ready = []
        for slot, request in enumerate(candidates):
            ack = next(
                (e["time_ns"] for e in acks[step] if e["candidates"][slot] == [1] * 8),
                math.inf,
            )
            ready.append(max(base, kv.get(request, math.inf), ack))
        assert 0 < target <= len(ready)
        stops = [sorted(ready)[target - 1], base + 2_000_000_000]
        if transport["has_compute"]:
            stops.append(end)
        exposed_stop = max(base, end, min(stops))
        seal = transport["seal_ns"]
        result["response_lag_estimate_ms"] += max(0, seal - exposed_stop) / 1e6
        for wait in waits:
            overlap = (
                max(0, min(seal, wait["end_ns"]) - max(exposed_stop, wait["begin_ns"]))
                / 1e6
            )
            result["sleep_after_stop_estimate_ms"] += overlap
            result[wait["phase"] + "_sleep_after_stop_estimate_ms"] += overlap
        result["covered_stop_windows"] += 1
    result["nonwait_response_lag_estimate_ms"] = (
        result["response_lag_estimate_ms"] - result["sleep_after_stop_estimate_ms"]
    )
    result["accounted_polling_cpu_ms"] = (
        analysis["adaptive_poll_cpu_ms"] + result["bridge_cpu_ms"]
    )
    assert (
        result["sleep_after_stop_estimate_ms"]
        <= result["response_lag_estimate_ms"] + 1e-6
    )
    return result


def summarize(root):
    rows = []
    for path in sorted(root.glob("*/p*/*/result.json")):
        if "warmretry" in path.parent.name and not load(path)["accepted"]:
            continue
        analysis_path = path.parent / "analysis.json"
        if analysis_path.exists():
            analysis = load(analysis_path)
        else:
            analysis = analyzer.analyze(path.parent)
        row = dict(
            phase=path.parents[2].name,
            workload=path.parents[1].name,
            policy=path.parent.name.split("_", 1)[1],
            **analysis,
            **diagnostics(path.parent, analysis),
        )
        rows.append(row)
    out = root / "summary"
    out.mkdir(exist_ok=True)
    (out / "cells.json").write_text(json.dumps(rows, indent=2) + "\n")
    groups = defaultdict(list)
    for row in rows:
        if row["accepted"]:
            groups[row["workload"], row["policy"]].append(row)
    aggregate = []
    columns = [
        "throughput",
        "mean_ttft_ms",
        "p95_ttft_ms",
        "compute_calls",
        "adaptive_poll_cpu_ms",
        "bridge_cpu_ms",
        "accounted_polling_cpu_ms",
        "response_lag_estimate_ms",
        "sleep_after_stop_estimate_ms",
        "coarse_sleep_after_stop_estimate_ms",
        "fine_sleep_after_stop_estimate_ms",
    ]
    for (workload, policy), cells in sorted(groups.items()):
        item = dict(workload=workload, policy=policy, repeats=len(cells))
        item.update({key: statistics.mean(c[key] for c in cells) for key in columns})
        item["exposed_hold_lower_ms"] = statistics.mean(
            c["exposed_lower_ms"].get("total", 0) for c in cells
        )
        item["recovered_opportunities"] = sum(
            c["recovered_compute_opportunities"] for c in cells
        )
        item["total_opportunities"] = sum(c["kv_ready_during_compute"] for c in cells)
        item["admission_percent"] = (
            100 * item["recovered_opportunities"] / item["total_opportunities"]
            if item["total_opportunities"]
            else None
        )
        aggregate.append(item)
    for item in aggregate:
        control = next(
            a
            for a in aggregate
            if a["workload"] == item["workload"] and a["policy"] == "capped50_fine2"
        )
        item["throughput_gain_percent"] = (
            item["throughput"] / control["throughput"] - 1
        ) * 100
        item["mean_ttft_reduction_percent"] = (
            1 - item["mean_ttft_ms"] / control["mean_ttft_ms"]
        ) * 100
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    if aggregate:
        with (out / "aggregate.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
            writer.writeheader()
            writer.writerows(aggregate)
    complete = (
        len(rows) == 48
        and all(c["accepted"] for c in rows)
        and (root / "confirmation/complete.json").exists()
    )
    text = [
        "# Full predicted wait: GLM 5.3 TP8",
        "",
        (
            f"Completed cells: {len(rows)}/48. "
            f"Accepted: {sum(c['accepted'] for c in rows)}."
        ),
        "",
        (
            "Values are arithmetic means across available seed 57/58 cells. "
            "Gains use the matched capped control. "
            + (
                "Both phases are complete. See paired_comparisons.json for "
                "individual seed comparisons; two replays do not establish "
                "statistical significance."
                if complete
                else "Provisional until both phases are complete."
            )
        ),
        "",
        (
            "| Workload | Policy | Repeats | req/s | Gain | Mean TTFT ms | "
            "p95 TTFT ms | Compute rounds | Accounted polling CPU ms | "
            "Exposed hold ms | Sleep after stop ms | Admission ratio |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregate:
        ratio = (
            f"{item['admission_percent']:.1f}%"
            if item["admission_percent"] is not None
            else "n/a"
        )
        text.append(
            f"| {item['workload']} | {item['policy']} | {item['repeats']} | "
            f"{item['throughput']:.3f} | {item['throughput_gain_percent']:+.2f}% | "
            f"{item['mean_ttft_ms']:.1f} | {item['p95_ttft_ms']:.1f} | "
            f"{item['compute_calls']:.1f} | "
            f"{item['accounted_polling_cpu_ms']:.2f} | "
            f"{item['exposed_hold_lower_ms']:.2f} | "
            f"{item['sleep_after_stop_estimate_ms']:.2f} | {ratio} |"
        )
    text.extend(
        [
            "",
            "## Timing and CPU limits",
            "",
            (
                "Sleep-after-stop is the sum of recorded wait intervals after an "
                "eligible stop condition and preceding compute completion. "
                "Candidate ACK times are engine observations; completion timing "
                "uses its upper endpoint. It is a diagnostic estimate, not an "
                "exact causal latency bound."
            ),
            "",
            (
                "Main polling CPU covers iterations leading to adaptive waits "
                "(and observer installation), excluding final/pre-ACK iterations. "
                "Bridge CPU covers the entire observer thread. Their sum is "
                "accounted polling CPU, not whole-engine CPU. Unknown GPU timing "
                "windows are excluded and listed in cells.json. The bridge polls "
                "read-only transport state every 1 ms during coarse waits."
            ),
        ]
    )
    (out / "REPORT.md").write_text("\n".join(text) + "\n")
    (root / "checkpoint.json").write_text(
        json.dumps(
            dict(
                time=time.time(),
                cells=len(rows),
                accepted=sum(c["accepted"] for c in rows),
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"cells={len(rows)} accepted={sum(c['accepted'] for c in rows)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        summarize(ROOT)
        if not args.watch or (ROOT / "confirmation/complete.json").exists():
            break
        if "driver_failed" in (ROOT / "process_log.jsonl").read_text():
            break
        time.sleep(20)
