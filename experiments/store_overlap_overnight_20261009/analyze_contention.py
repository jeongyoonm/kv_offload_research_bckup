# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate store API time from concurrent native load submission in Nsight."""

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict


def analyze(path):
    connection = sqlite3.connect(path)
    ranges = connection.execute(
        "SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value) "
        "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId "
        "WHERE coalesce(n.text,s.value) LIKE 'd2h_submit/%' ORDER BY n.start"
    ).fetchall()
    calls = connection.execute(
        "SELECT r.start,r.end,r.globalTid,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId WHERE s.value='cuMemcpyBatchAsync' "
        "ORDER BY r.start"
    ).fetchall()
    results = []
    for start, end, thread, label in ranges:
        fields = dict(item.split("=") for item in label.split("/")[1:])
        if int(fields["repeat"]) < 2:
            continue
        own = [c for c in calls if c[2] == thread and start <= c[0] < end]
        other = [c for c in calls if c[2] != thread and c[0] < end and c[1] > start]
        overlap = (
            sum(max(0, min(a[1], b[1]) - max(a[0], b[0])) for a in own for b in other)
            / 1e6
        )
        assert len(own) == 1, (label, len(own))
        results.append(
            dict(
                **fields,
                store_submit_ms=(end - start) / 1e6,
                store_cuda_api_ms=(own[0][1] - own[0][0]) / 1e6,
                simultaneous_h2d_api_ms=overlap,
                overlapping_h2d_calls=len(other),
            )
        )
    groups = defaultdict(list)
    for row in results:
        groups[row["layout"], row["coalesced"], row["background"]].append(row)
    summaries = []
    for (layout, merged, background), rows in groups.items():
        summaries.append(
            dict(
                layout=layout,
                coalesced=merged,
                background=background,
                samples=len(rows),
                **{
                    name: statistics.median(row[name] for row in rows)
                    for name in (
                        "store_submit_ms",
                        "store_cuda_api_ms",
                        "simultaneous_h2d_api_ms",
                        "overlapping_h2d_calls",
                    )
                },
            )
        )
    return dict(
        episodes=results,
        summaries=summaries,
        definition="Two measured repeats per case, following two excluded warmups. "
        "CUDA API durations are profiled observations, not mutex wait measurements. "
        "Simultaneous H2D time intersects other-thread batch-copy calls with the "
        "D2H batch-copy call. Disjoint buffers eliminate source-reuse dependencies "
        "but not shared driver, PCIe, or CPU resources.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite")
    parser.add_argument("output")
    args = parser.parse_args()
    result = analyze(args.sqlite)
    with open(args.output, "w") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
