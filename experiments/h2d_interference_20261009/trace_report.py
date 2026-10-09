# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Relate copy API windows to gaps between compute launches in diagnostic NVTX."""

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def merge(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def overlap(interval, windows):
    a, b = interval
    return sum(max(0, min(b, y) - max(a, x)) for x, y in windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    con = sqlite3.connect(f"file:{root / 'trace.sqlite'}?mode=ro", uri=True)
    rows = []
    ranges = list(
        con.execute(
            "SELECT start,end,globalTid,text FROM NVTX_EVENTS "
            "WHERE text LIKE 'calibration/%' ORDER BY start"
        )
    )
    for a, b, tid, label in ranges:
        # Nsight globalTid reserves the low 24 bits for thread identity.
        pid = tid & ~0xFFFFFF
        launch_records = list(
            con.execute(
                "SELECT r.start,r.end,r.correlationId "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON s.id=r.nameId WHERE r.globalTid=? "
                "AND r.start>=? AND r.end<=? AND s.value LIKE '%LaunchKernel%'",
                (tid, a, b),
            )
        )
        launches = merge([(x, y) for x, y, _ in launch_records])
        assert launches
        gaps = [(launches[i][1], launches[i + 1][0]) for i in range(len(launches) - 1)]
        copy_calls = list(
            con.execute(
                "SELECT r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON s.id=r.nameId WHERE (r.globalTid & ~16777215)=? "
                "AND r.start>=? AND r.end<=? AND s.value='cuMemcpyBatchAsync'",
                (pid, a, b),
            )
        )
        copies = merge(copy_calls)
        slow = [(x, y) for x, y in gaps if y - x >= 500_000]
        top = sorted(gaps, key=lambda xy: xy[1] - xy[0], reverse=True)[:10]
        correlations = {z for _, _, z in launch_records}
        kernels = [
            (x, y)
            for x, y, z in con.execute(
                "SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE globalPid=? AND start>=?",
                (pid, a),
            )
            if z in correlations
        ]
        kernel_windows = merge(kernels)
        assert kernel_windows
        kernel_busy = sum(y - x for x, y in kernel_windows) / 1e6
        kernel_span = (kernel_windows[-1][1] - kernel_windows[0][0]) / 1e6
        record = dict(
            label=label,
            global_tid=tid,
            forward_host_ms=(b - a) / 1e6,
            copy_calls=len(copy_calls),
            copy_api_ms=sum(y - x for x, y in copies) / 1e6,
            max_copy_call_ms=max((y - x for x, y in copy_calls), default=0) / 1e6,
            compute_launch_calls=len(launch_records),
            compute_launch_api_union_ms=sum(y - x for x, y in launches) / 1e6,
            max_compute_launch_gap_ms=max(y - x for x, y in gaps) / 1e6,
            slow_gaps=len(slow),
            slow_gap_ms=sum(y - x for x, y in slow) / 1e6,
            slow_gap_copy_api_overlap_ms=sum(overlap(g, copies) for g in slow) / 1e6,
            top10_gap_copy_overlap_count=sum(overlap(g, copies) > 0 for g in top),
            matched_gpu_kernels=len(kernels),
            first_gpu_kernel_lag_ms=(kernel_windows[0][0] - a) / 1e6,
            last_gpu_kernel_from_host_ms=(kernel_windows[-1][1] - a) / 1e6,
            gpu_kernel_span_ms=kernel_span,
            gpu_kernel_busy_union_ms=kernel_busy,
            gpu_kernel_gap_ms=kernel_span - kernel_busy,
        )
        rows.append(record)
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    summary = []
    for label, group in groups.items():
        assert len(group) == 8
        summary.append(
            dict(
                label=label,
                ranks=8,
                **{
                    k: statistics.mean(r[k] for r in group)
                    for k in group[0]
                    if k not in ("label", "global_tid")
                },
            )
        )
    result = dict(
        diagnostic=True,
        rows=rows,
        aggregate=summary,
        scope="One load per rank; one profiled cell/config; profiler overhead. "
        "Temporal copy-API/launch-gap coincidence is evidence of submission "
        "interference, not proof of a specific mutex. CUDA APIs omitted by the "
        "default trace and uninstrumented host work remain unclassified. "
        "Kernel-resident time includes NCCL waiting kernels and is not useful "
        "compute utilization; its complement is not total GPU bubble time.",
    )
    (root / "trace_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
