# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write progressive results with serving load and agent-queue diagnostics."""

import json
import time
from pathlib import Path

ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_request_slots_20261013")


def write_report():
    cells = {}
    for path in (ROOT / "cells").glob("*/result.json"):
        result = json.loads(path.read_text())
        if result["accepted"]:
            cells[result["arm"]] = result
    complete = (ROOT / "completed.json").exists()
    failed = (ROOT / "failure.json").exists()
    lines = [
        "TraceLab: separate model-call slots from agents waiting for tools",
        f"Status: {'complete' if complete else 'failed' if failed else 'running'}",
        f"Accepted cells: {len(cells)}/2",
        "",
        (
            "Arm | measured req/s | HTTP mean TTFT ms | HTTP p95 TTFT ms | "
            "ready mean TTFT ms | mean calls in flight | no HTTP calls in flight s | "
            "ready queue nonempty % | CPU-hit requests %"
        ),
    ]
    for arm in ("baseline", "full"):
        if arm not in cells:
            continue
        r = cells[arm]
        cpu = (
            r["request_outcomes"].get("cpu_hit", 0)
            / r["latency_cohort_submissions"]
            * 100
        )
        lines.append(
            f"{arm} | {r['throughput_rps']:.5f} | {r['ttft_ms']['mean']:.2f} | "
            f"{r['ttft_ms']['p95']:.2f} | {r['ready_ttft_ms']['mean']:.2f} | "
            f"{r['mean_http_in_flight']:.2f} | "
            f"{r['measured_no_http_in_flight_seconds']:.3f} | "
            f"{r['sampled_fraction_ready_queue_nonempty'] * 100:.2f} | {cpu:.2f}"
        )
    if set(cells) == {"baseline", "full"}:
        b, f = cells["baseline"], cells["full"]
        gain = (f["throughput_rps"] / b["throughput_rps"] - 1) * 100
        ttft_gain = (1 - f["ttft_ms"]["mean"] / b["ttft_ms"]["mean"]) * 100
        ready_gain = (1 - f["ready_ttft_ms"]["mean"] / b["ready_ttft_ms"]["mean"]) * 100
        lines += [
            "",
            f"Throughput gain: {gain:+.2f}%",
            f"HTTP mean TTFT reduction: {ttft_gain:+.2f}%",
            f"Ready mean TTFT reduction: {ready_gain:+.2f}%",
        ]
    lines += [
        "",
        "Request limit: 32 model calls; no session-wide concurrency cap.",
        (
            "Full catalogue: 8,058 sessions / 665,453 calls. Native exporter "
            "defaults retained: seed 0, Poisson arrivals at 1 session/s, summed "
            "wall tool waits. No tool waits clipped, scaled, or bypassed."
        ),
        (
            "Timed episode: 600s warmup + 1200s fixed measurement. After 1800s "
            "stop submitting new calls and finish every call already submitted. "
            "Unissued calls and future session arrivals are outside this episode."
        ),
        (
            "Throughput counts responses completed within the fixed window. "
            "Latency cohort contains submissions inside the window, including "
            "their responses during drain, avoiding unfinished-response censoring."
        ),
        (
            "HTTP TTFT starts at POST. Ready TTFT additionally includes prompt "
            "construction and client request-slot queueing after agent readiness."
        ),
        (
            "Cold logical caches at episode start; natural promotion, eviction, "
            "write-back and reload. Realized CPU/GPU/miss proportions are measured."
        ),
        (
            "GLM 5.3 NVFP4 TP8, 1,048,576 total context, 96 GiB/rank GPU KV, "
            "3,028,287,488,000 bytes requested CPU KV server-wide (private memfd)."
        ),
        (
            "Serving configuration and predictor are unchanged from the previous "
            "matched pair. No retraining or model/selector changes."
        ),
        (
            "Synthetic prompt reconstruction and real generated token-ID carry "
            "forward use the upstream runner. This is a structural replay."
        ),
        (
            "Native zero-length append/output rows are clamped to one token by "
            "the upstream exporter, identically for both arms."
        ),
        (
            "All actually submitted calls are audited for token counts, unique "
            "retirement, all-rank transfers, and frozen predictor state. Two "
            "native preflight requests are included separately in server audits."
        ),
        (
            "One pass per arm, baseline first; no confidence intervals. The "
            "same catalogue and arrival policy produce closed-loop load; "
            "actual requests reached during the fixed window can differ by arm."
        ),
        (
            "In-flight HTTP observations measure serving demand, not physical "
            "GPU utilization. Raw worker timelines are retained."
        ),
    ]
    if failed:
        lines += ["", (ROOT / "failure.json").read_text()]
    path = ROOT / "REPORT.txt"
    temp = path.with_suffix(".tmp")
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(path)
    return complete or failed


def main():
    while not write_report():
        time.sleep(30)


if __name__ == "__main__":
    main()
