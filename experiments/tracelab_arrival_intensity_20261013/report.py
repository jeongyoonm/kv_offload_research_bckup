# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report arrival calibration and comparisons without mixing warmup cache hits."""

import json
import time

from experiments.tracelab_arrival_intensity_20261013.prepare import ROOT


def write_report():
    done = (ROOT / "completed.json").exists()
    failed = (ROOT / "failure.json").exists()
    results = [
        json.loads(path.read_text())
        for path in sorted((ROOT / "results").glob("*.json"))
    ]
    lines = [
        "TraceLab arrival-intensity calibration and matched baseline/full evaluation",
        f"Status: {'complete' if done else 'failed' if failed else 'running'}",
        "",
        (
            "Episode | sessions/s | req/s | HTTP mean TTFT ms | p95 TTFT ms | "
            "CPU-hit requests % | last-half CPU-hit % | cached prompt tokens % | "
            "queue mean ms | queue p95 ms | GPU mean % | mean calls in flight | "
            "GPU preemptions (whole episode)"
        ),
    ]
    for result in results:
        profile = result["cache_profile"]
        measured = profile["measurement"]
        gpu = profile["gpu_utilization_mean_percent"]
        gpu_text = "N/A" if gpu is None else f"{gpu:.2f}"
        lines.append(
            f"{result['episode']} | "
            f"{result['arrival_rate_sessions_per_second']:.3f} | "
            f"{result['throughput_rps']:.5f} | {result['ttft_ms']['mean']:.2f} | "
            f"{result['ttft_ms']['p95']:.2f} | "
            f"{measured['cpu_hit_request_percent']:.2f} | "
            f"{profile['last_half']['cpu_hit_request_percent']:.2f} | "
            f"{measured['cached_prompt_token_percent']:.2f} | "
            f"{result['slot_queue_ms']['mean']:.2f} | "
            f"{result['slot_queue_ms']['p95']:.2f} | {gpu_text} | "
            f"{result['mean_http_in_flight']:.2f} | "
            f"{result['metric_deltas'].get('vllm:num_preemptions_total', 0):.0f}"
        )
    for family in ("primary", "reverse", "repair"):
        pair = {
            result["arm"]: result
            for result in results
            if result["episode"].startswith(family + "_")
        }
        if set(pair) == {"baseline", "full"}:
            baseline, full = pair["baseline"], pair["full"]
            throughput = (full["throughput_rps"] / baseline["throughput_rps"] - 1) * 100
            mean_ttft = (
                1 - full["ttft_ms"]["mean"] / baseline["ttft_ms"]["mean"]
            ) * 100
            p95_ttft = (1 - full["ttft_ms"]["p95"] / baseline["ttft_ms"]["p95"]) * 100
            lines += [
                "",
                f"{family} throughput gain: {throughput:+.2f}%",
                f"{family} mean HTTP TTFT reduction: {mean_ttft:+.2f}%",
                f"{family} p95 HTTP TTFT reduction: {p95_ttft:+.2f}%",
            ]
    lines += [
        "",
        (
            "Preserved full catalogue: 8058 sessions / 665453 calls, source mix, "
            "seed0 session order, all prefix/input/output lengths and tool waits. "
            "No session sampling, context clipping, or output shortening."
        ),
        (
            "Only synthetic session arrivals are uniformly stretched. Nonarrival "
            "CSV fields have identical SHA256 digests across all variants."
        ),
        (
            "32 in-flight model calls; no session cap. Tool waits consume no call "
            "slots. Natural cache evolution, same maximum validated CPU budget "
            "and 96GiB/rank GPU KV."
        ),
        (
            "Selection uses baseline CPU reuse, queue stability, and GPU activity. "
            "Arrival rate is frozen before the prototype. See selection.json."
        ),
        (
            "Pilots: 300s warmup + 600s measurement. Primary: 600s + 1200s. "
            "Optional reverse-order pair: 300s + 900s, reported separately. "
            "Logical CPU/GPU caches reset each episode; submitted calls drain."
        ),
        (
            "Repair pair: 600s warmup + 900s measurement, full first. Both use "
            "1s HTTP pool idle timeout; no retries. The earlier full run was "
            "rejected after a transport failure. Earlier baseline is a "
            "calibration/control, not paired with the repaired client."
        ),
        (
            "Throughput uses fixed-window completion timestamps. HTTP latency "
            "and cache-hit requests use the measurement POST cohort, including "
            "responses during drain. Cache ratios exclude warmup."
        ),
        (
            "CPU-hit requests means any observed CPU load for that request. "
            "Preemptions can produce repeat loads; physical transfer bytes and "
            "initial-prefill token-source counters are audited separately."
        ),
        (
            "Arrival-limited throughput is not maximum capacity. GPU activity, "
            "queueing, occupancy, and request mix accompany gains. Faster "
            "closed-loop service can advance farther through the same sessions."
        ),
        (
            "Serving model, original selector, predictor, worker and scheduler "
            "are unchanged. No priming, pinning, or eviction-policy changes."
        ),
        (
            "Upstream synthetic prompts and real output token-ID carry forward; "
            "original raw cross-session prompt identities are unavailable."
        ),
        (
            "All submitted calls checked for token counts, success, retirement, "
            "eight-rank transfers, frozen predictor, slots, and timed window. "
            "Single primary pair has no confidence interval."
        ),
    ]
    if failed:
        lines += ["", (ROOT / "failure.json").read_text()]
    temporary = ROOT / "REPORT.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(ROOT / "REPORT.txt")
    return done or failed


def main():
    while not write_report():
        time.sleep(20)


if __name__ == "__main__":
    main()
