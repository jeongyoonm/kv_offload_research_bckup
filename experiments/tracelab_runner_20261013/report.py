# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write a readable report as matched upstream-runner cells complete."""

import argparse
import json
import time
from pathlib import Path

ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_runner_20261013")


def write_report():
    cells = []
    for path in sorted((ROOT / "cells").glob("*/result.json")):
        cell = json.loads(path.read_text())
        if cell["accepted"]:
            cells.append(cell)
    complete = (ROOT / "completed.json").exists()
    failed = (ROOT / "failure.json").exists()
    lines = [
        "TraceLab official runner: baseline versus full prototype",
        f"Status: {'complete' if complete else 'failed' if failed else 'running'}",
        f"Accepted cells: {len(cells)}/6",
        "",
        (
            "Active sessions | Baseline req/s | Full req/s | Throughput gain % | "
            "Baseline mean TTFT ms | Full mean TTFT ms | Mean TTFT reduction % | "
            "Baseline p95 TTFT ms | Full p95 TTFT ms"
        ),
    ]
    pairs = []
    for concurrency in (8, 16, 32):
        matched = {c["arm"]: c for c in cells if c["concurrency"] == concurrency}
        if set(matched) != {"baseline", "full"}:
            continue
        b, f = matched["baseline"], matched["full"]
        gain = (f["throughput_rps"] / b["throughput_rps"] - 1) * 100
        ttft_gain = (1 - f["ttft_ms"]["mean"] / b["ttft_ms"]["mean"]) * 100
        pairs.append(
            dict(
                concurrency=concurrency,
                baseline=b,
                full=f,
                throughput_gain_pct=gain,
                mean_ttft_reduction_pct=ttft_gain,
            )
        )
        lines.append(
            f"{concurrency} | {b['throughput_rps']:.5f} | "
            f"{f['throughput_rps']:.5f} | {gain:+.2f} | "
            f"{b['ttft_ms']['mean']:.2f} | {f['ttft_ms']['mean']:.2f} | "
            f"{ttft_gain:+.2f} | {b['ttft_ms']['p95']:.2f} | "
            f"{f['ttft_ms']['p95']:.2f}"
        )
    lines += [
        "",
        "GLM 5.3 NVFP4, TP8; total context limit 1,048,576 tokens.",
        (
            "Requested CPU KV budget 3,028,287,488,000 bytes server-wide; "
            "GPU KV budget 96 GiB per rank. Private memfd CPU layout in both arms."
        ),
        (
            "64 frozen sessions (32 Codex, 32 Claude), 1,880 complete invocations "
            "per cell; up to 64 consecutive invocations per session."
        ),
        (
            "Upstream active-session semaphore retained throughout each session, "
            "including tool waits. No HTTP-request concurrency replacement."
        ),
        (
            "Unmodified upstream exporter: seed 0, shuffled sessions, Poisson "
            "session arrivals at 1/s, summed observed wall tool waits."
        ),
        (
            "12,564.578 seconds aggregate tool waits; longest session's waits "
            "2,557.457 seconds; largest individual wait 1,518.859 seconds. "
            "No waits clipped or scaled."
        ),
        (
            "Throughput covers first invocation submit to final completion, "
            "including session queueing, arrivals and tool waits. "
            "Native TTFT starts at HTTP submission."
        ),
        (
            "Runner Rust source unchanged; tokenizer dependency updated "
            "0.19 -> 0.23.2 for GLM tokenizer JSON compatibility. "
            "enwik9 corpus; default upstream token-pool sizing."
        ),
        (
            "Actual generated output IDs carried forward by the native runner. "
            "Subsequent input lengths follow the fixed CSV, while generated "
            "content can differ between arms."
        ),
        (
            "TraceLab provides lengths/provider cache accounting, not exact "
            "prompt identities; this remains a structural synthetic replay."
        ),
        "No predictor retraining or serving-source/model-selector changes.",
        (
            "Two upstream preflight requests excluded from throughput and "
            "per-round latency, explicitly included in server-accounting audits."
        ),
        "One pass per cell, baseline first then full; no confidence intervals.",
    ]
    if failed:
        lines += ["", "Failure: " + (ROOT / "failure.json").read_text()]
    path = ROOT / "REPORT.txt"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)
    path = ROOT / "matched_pairs.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(pairs, indent=2) + "\n")
    tmp.replace(path)
    return complete or failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        done = write_report()
        if done or not args.watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
