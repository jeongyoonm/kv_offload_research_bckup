# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure isolated checkpoint GC traversal; freezing is a CPU diagnostic only."""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

from vllm.v1.kv_offload.seal_predictor import ComputeEstimator


def rss_kib():
    return int(
        next(
            s.split()[1]
            for s in Path("/proc/self/status").read_text().splitlines()
            if s.startswith("VmRSS:")
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists()
    gc.collect(2)
    before = len(gc.get_objects())
    before_rss = rss_kib()
    started = time.monotonic_ns()
    state = json.loads(args.state.read_text())
    estimator = ComputeEstimator.from_state_dict(state, state["context"])
    del state
    construction_ms = (time.monotonic_ns() - started) / 1e6
    gc.collect(2)
    loaded = len(gc.get_objects())
    loaded_rss = rss_kib()
    if args.freeze:
        gc.freeze()
    samples = []
    for _ in range(5):
        started = time.monotonic_ns()
        collected = gc.collect(2)
        samples.append(
            dict(duration_ms=(time.monotonic_ns() - started) / 1e6, collected=collected)
        )
    assert estimator.latency_surface is not None
    result = dict(
        freeze_diagnostic=args.freeze,
        construction_ms=construction_ms,
        tracked_before=before,
        tracked_after_loading=loaded,
        added_tracked_objects=loaded - before,
        rss_before_kib=before_rss,
        rss_loaded_kib=loaded_rss,
        samples=samples,
        median_gc_ms=statistics.median(s["duration_ms"] for s in samples),
        caveat="Isolated CPU model construction and explicit GC. No serving "
        "objects, GPU work, or throughput inference. Freeze is confined to "
        "this disposable process, not a proposed serving policy.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
