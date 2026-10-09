# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse the existing gap audit with explicit GPU-fence and store API markers."""

import json
import statistics
import sys
from pathlib import Path

from experiments.h2d8_deadline_bias_20261009 import report


def main():
    report.ARMS = ("control", "gpu")
    report.PAIRS.update(
        store_reuse_fence_enqueue_begin="store_reuse_fence_enqueue_end",
        store_descriptors_begin="store_descriptors_end",
        store_stream_dependencies_begin="store_stream_dependencies_end",
        store_copy_enqueue_begin="store_copy_enqueue_end",
    )
    report.main()
    root = Path(sys.argv[1])
    comparison = json.loads((root / "comparison.json").read_text())
    comparison["definitions"]["inference"] = (
        "Matched continuous 300-request episodes with a 75-prefix CPU working "
        "set and four distinct suffixes per prefix. Both arms retain global "
        "GPU reuse fences and the frozen scheduler. The candidate additionally "
        "uses store-aware allocation, ascending new GPU assignment, H2D/D2H "
        "coalescing and packet-aligned splitting. Actual prefix restoration "
        "and zero GPU cache hits are required. These are descriptive repeats, "
        "not a significance claim; the aggregate is not a single 300-request "
        "episode when the driver is run with multiple repeats."
    )
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    rows = []
    for result in comparison["episodes"]:
        directory = root / "live/familiar60k" / f"r{result['repeat']}_{result['label']}"
        events = list(
            map(json.loads, (directory / "engine.jsonl").read_text().splitlines())
        )
        computes = sorted(
            [
                e
                for e in events
                if e["event"] == "step_completion" and e.get("selected")
            ],
            key=lambda e: e["step"],
        )
        first = max(r[3] for r in computes[0]["execution_ranks"])
        end = max(r[3] + r[6] for r in computes[-1]["execution_ranks"])
        client_start = min(r["start_ns"] for r in result["responses"])
        client_end = max(r["end_ns"] for r in result["responses"])
        span = (client_end - client_start) / 1e6
        rows.append(
            dict(
                arm=result["label"],
                repeat=result["repeat"],
                client_elapsed_ms=result["seconds"] * 1000,
                response_span_ms=span,
                before_first_compute_ms=(first - client_start) / 1e6,
                gpu_span_ms=(end - first) / 1e6,
                after_last_compute_ms=(client_end - end) / 1e6,
                client_harness_overhead_ms=result["seconds"] * 1000 - span,
            )
        )
    metrics = tuple(k for k in rows[0] if k not in ("arm", "repeat"))
    means = {
        arm: {
            metric: statistics.mean(r[metric] for r in rows if r["arm"] == arm)
            for metric in metrics
        }
        for arm in report.ARMS
    }
    (root / "bookends.json").write_text(
        json.dumps(
            dict(
                episodes=rows,
                means=means,
                definition="Before/after compute use the earliest request start "
                "and latest response end, and conservative all-rank GPU bounds. "
                "The GPU span includes compute and inter-compute gaps. Client "
                "harness overhead includes time before the earliest request "
                "thread starts and after the final response ends. These terms "
                "partition client elapsed time algebraically; they do not "
                "identify the causes of startup or final-response latency.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
