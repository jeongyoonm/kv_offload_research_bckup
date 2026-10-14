# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit fixed-cohort calibration and summarize useful compute/copy overlap."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.kernels.calibrate_async_h2d import PROFILES, ROW_BYTES


def aggregate_cell(result):
    cell, ranks = result["cell"], result["ranks"]
    assert len(ranks) == 8 and {r["rank"] for r in ranks} == set(range(8))
    assert all(all(row[k] == v for k, v in cell.items()) for row in ranks)
    begin = min(r["host_begin_ns"] for r in ranks)
    ends_lower, ends_upper = [], []
    row = dict(**cell)
    if cell["mode"] != "copy_only":
        execution = [r["execution"] for r in ranks]
        assert all(r[0] == cell["id"] and r[5] == 1 and r[6] > 0 for r in execution)
        assert all(0 < r[1] <= r[2] <= r[3] <= r[4] for r in execution)
        assert all(r[2] <= r[7] <= r[8] <= r[4] for r in execution)
        ends_lower.append(max(r[7] for r in execution))
        ends_upper.append(max(r[8] for r in execution))
        row.update(
            forward_enqueue_ms=max(
                r["compute_enqueued_ns"] - r["host_begin_ns"] for r in ranks
            )
            / 1e6,
            model_submit_ms=max(
                r["compute_enqueued_ns"] - r["compute_submit_begin_ns"] for r in ranks
            )
            / 1e6,
            max_rank_gpu_span_ms=max(r[6] for r in execution) / 1e6,
            compute_finish_ms=(max(r[8] for r in execution) - begin) / 1e6,
            compute_finish_lower_ms=(max(r[7] for r in execution) - begin) / 1e6,
            max_start_uncertainty_ms=max(r[3] - r[2] for r in execution) / 1e6,
            max_end_uncertainty_ms=max(r[8] - r[7] for r in execution) / 1e6,
            start_skew_ms=(max(r[3] for r in execution) - min(r[2] for r in execution))
            / 1e6,
        )
    if cell["mode"] != "compute_only":
        for rank in ranks:
            assert rank["bytes_per_job"] == PROFILES[cell["profile"]] * ROW_BYTES
            assert len(rank["completion_ns"]) == cell["jobs"]
            assert rank["completion_ns"] == sorted(rank["completion_ns"])
            assert rank["io_begin_ns"] <= rank["io_enqueued_ns"]
            assert rank["io_begin_ns"] <= rank["completion_ns"][0]
            assert rank["validated_bytes"] == 99 * 3 * 64
            if cell["mode"] == "after_enqueued":
                assert rank["compute_enqueued_ns"] <= rank["io_begin_ns"]
                if "io_submission_stream" in rank:
                    assert rank["io_submission_stream"] != rank["compute_stream"]
            elif cell["mode"] == "concurrent":
                assert rank["io_enqueued_ns"] <= rank["compute_submit_begin_ns"]
                if "io_submission_stream" in rank:
                    assert rank["io_submission_stream"] == rank["compute_stream"]
        finish = max(r["completion_ns"][-1] for r in ranks)
        io_begin = min(r["io_begin_ns"] for r in ranks)
        ends_lower.append(finish)
        ends_upper.append(finish)
        row.update(
            first_ready_ms=(max(r["completion_ns"][0] for r in ranks) - begin) / 1e6,
            copy_finish_ms=(finish - begin) / 1e6,
            copy_active_ms=(finish - io_begin) / 1e6,
            per_gpu_gbps=PROFILES[cell["profile"]]
            * ROW_BYTES
            * cell["jobs"]
            / (finish - io_begin),
            io_enqueue_ms=max(r["io_enqueued_ns"] - r["io_begin_ns"] for r in ranks)
            / 1e6,
        )
        if cell["mode"] != "copy_only":
            # Guaranteed overlap between compute's observed start and end intervals
            # and the copy cohort's host submission/completion window.
            start_upper = max(r["execution"][3] for r in ranks)
            end_lower = min(r["execution"][7] for r in ranks)
            row["overlap_window_lower_ms"] = (
                max(0, min(end_lower, finish) - max(start_upper, io_begin)) / 1e6
            )
    row["joint_finish_ms"] = (max(ends_upper) - begin) / 1e6
    row["joint_finish_lower_ms"] = (max(ends_lower) - begin) / 1e6
    return row


def paired_gain(pairs):
    logs = [math.log(base / other) for base, other in pairs]
    mean = statistics.mean(logs)
    t = {
        1: 0,
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(len(logs), 1.96)
    margin = t * statistics.stdev(logs) / math.sqrt(len(logs)) if len(logs) > 1 else 0
    return dict(
        gain_percent=100 * math.expm1(mean),
        bounds=[100 * math.expm1(mean - margin), 100 * math.expm1(mean + margin)],
        repeats=len(logs),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "plan.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"] and summary["accepted"] == len(plan["cells"])
    cells = []
    for cell in plan["cells"]:
        result = json.loads((root / f"cell-{cell['id']:04d}.json").read_text())
        assert result["cell"] == cell
        if cell["shape"] is not None:
            geom = plan["geometry"][cell["shape"]]
            assert len(result["output_tokens"]) == geom["batch"]
            for rank in result["ranks"]:
                assert rank["suffix"] == [geom["suffix"]] * geom["batch"]
                assert rank["context"] == [geom["context"]] * geom["batch"]
        cells.append(aggregate_cell(result))
    keys = ("mode", "shape", "profile", "chunk_mib", "depth")
    grouped = defaultdict(list)
    for row in cells:
        grouped[tuple(row[k] for k in keys)].append(row)
    aggregate = []
    for key, group in grouped.items():
        assert len(group) == plan["repeats"]
        metrics = [k for k in group[0] if k not in plan["cells"][0]]
        record = dict(zip(keys, key))
        record["repeats"] = len(group)
        record.update({k: statistics.mean(g[k] for g in group) for k in metrics})
        aggregate.append(record)
    indexed = {
        (
            r["mode"],
            r["shape"],
            r["profile"],
            r["chunk_mib"],
            r["depth"],
            r["repeat"],
        ): r
        for r in cells
    }
    contrasts = []
    for row in aggregate:
        if row["mode"] != "concurrent":
            continue
        contrasts.append(
            dict(
                **{k: row[k] for k in keys},
                joint=paired_gain(
                    [
                        (
                            indexed[
                                (
                                    "concurrent",
                                    row["shape"],
                                    row["profile"],
                                    128,
                                    2,
                                    repeat,
                                )
                            ]["joint_finish_ms"],
                            indexed[(*[row[k] for k in keys], repeat)][
                                "joint_finish_ms"
                            ],
                        )
                        for repeat in range(plan["repeats"])
                    ]
                )
                if [128, 2] in plan["configs"]
                else None,
            )
        )
    out = dict(
        complete=True,
        cells=cells,
        aggregate=aggregate,
        contrasts=contrasts,
        scope=plan["scope"],
        all_rank_cells=len(cells),
        interval_note="Nominal paired log-ratio intervals; no multiplicity adjustment; "
        "unique suffix-leading token is not identical across cells.",
    )
    (root / "comparison.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(dict(cells=len(cells), groups=len(aggregate)), indent=2))
    for r in aggregate:
        if r["mode"] == "concurrent" and r["depth"] == 2:
            print(
                r["shape"],
                r["profile"],
                r["chunk_mib"],
                f"joint={r['joint_finish_ms']:.2f} ms",
                f"enqueue={r['forward_enqueue_ms']:.2f} ms",
                f"copy={r['per_gpu_gbps']:.2f} GB/s",
            )


if __name__ == "__main__":
    main()
