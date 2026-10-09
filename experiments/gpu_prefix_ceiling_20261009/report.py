# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize resident GPU controls separately from capacity-limited references."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"]
    groups = defaultdict(list)
    for episode in summary["results"]:
        assert episode["accepted"]
        groups[episode["workload"], episode["label"], episode["requests"]].append(
            episode
        )
    rows = []
    for (workload, label, count), episodes in groups.items():
        ttft = [r["ttft_ms"] for e in episodes for r in e["responses"]]
        restored = [t for e in episodes for t in e["observed_cached"]]
        expected = [t for e in episodes for t in e.get("expected_cached") or []]
        rows.append(
            dict(
                workload=workload,
                label=label,
                requests_per_episode=count,
                episodes=len(episodes),
                measured_requests=len(ttft),
                throughput_rps=statistics.mean(e["throughput_rps"] for e in episodes),
                aggregate_throughput_rps=len(ttft)
                / sum(e["seconds"] for e in episodes),
                mean_ttft_ms=statistics.mean(ttft),
                p50_ttft_ms=float(np.quantile(ttft, 0.5)),
                p95_ttft_ms=float(np.quantile(ttft, 0.95)),
                restored_prefix_tokens=sum(restored),
                cpu_load_bytes_server_wide=sum(
                    e["metrics_after"].get("vllm:kv_offload_load_bytes_total", 0)
                    - e["metrics_before"].get("vllm:kv_offload_load_bytes_total", 0)
                    for e in episodes
                ),
                gpu_prefix_hit_tokens=sum(
                    e["metrics_after"].get("vllm:prefix_cache_hits_total", 0)
                    - e["metrics_before"].get("vllm:prefix_cache_hits_total", 0)
                    for e in episodes
                ),
                full_residency_required=label != "gpu_capacity_limited",
                restored_expected_fraction=sum(restored) / sum(expected)
                if expected
                else None,
                per_repeat=[
                    dict(
                        repeat=e["repeat"],
                        seed=e["seed"],
                        throughput_rps=e["throughput_rps"],
                        mean_ttft_ms=e["mean_ttft_ms"],
                        gpu_hit_target_fraction=e.get("gpu_hit_target_fraction"),
                    )
                    for e in episodes
                ],
            )
        )
    output = dict(
        rows=rows,
        definitions="Warm priming is excluded; all cells pass cache-hit accounting, "
        "one-output-token and no-preemption checks. GPU-only resident cells require "
        "every expected prefix token in GPU cache, with zero CPU offload loads. CPU "
        "cells reset GPU cache after priming and verify CPU restoration. Full-resident "
        "references fit the unchanged GPU KV pool. Capacity-limited 75-request 60K "
        "episodes explicitly permit prefix misses and are not an all-hits ceiling. "
        "Throughput is the mean of episode rates; aggregate throughput is also given. "
        "Tail TTFT here is pooled across measured requests. CPU and GPU cache modes "
        "use independent allocations. GPU-only uses ordinary worker preparation "
        "without the connector-dependent observer or scheduler holds, so this is "
        "a measured resident reference, not a mathematical upper bound.",
    )
    (args.root / "ceiling_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(root=str(args.root), groups=len(rows))))


if __name__ == "__main__":
    main()
