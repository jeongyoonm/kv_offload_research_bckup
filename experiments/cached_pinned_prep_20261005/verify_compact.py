# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit matched compact-packet TP8 cells and execution timing."""

import argparse
import datetime
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    provenance = read(root / "canary/gpu/provenance.json")
    source = Path(provenance["source"])
    for name, expected in provenance["source_sha256"].items():
        # Results documentation is updated after the serving run.
        if name.endswith(".md"):
            continue
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, (
            name
        )
    cells = read(root / "cells.json")
    assert len(cells) == 9
    rows = []
    groups = defaultdict(list)
    for cell in cells:
        path = Path(cell["path"])
        result = read(path / "result.json")
        assert result["accepted"] and not result["measurement_jit_lines"]
        assert not result["runtime_error_lines"] and not result["unexpected_gpu_pids"]
        assert result["cached_distribution"] == {"58368": 80}
        assert not result["gpu_hits"] and not result["preemptions"]
        assert result["h2d_bytes"] == 1781858304000
        usage = read(path / "usage.json")
        assert len(usage) == len({r["request_id"] for r in usage}) == 80
        assert all(r["prompt"] == 61440 and r["cached"] == 58368 for r in usage)
        setting = read(path / "policy_setting.json")
        assert setting["late_seal_freeze_estimator"] is True
        assert setting["late_compact_tokens"] is (cell["mode"] == "compact_pinned")
        assert not read(path / "server_cpu.json")["errors"]
        events = [
            json.loads(line)
            for line in (path / "engine.jsonl").read_text().splitlines()
        ]
        computed = [
            e for e in events if e["event"] == "step_completion" and e.get("selected")
        ]
        assert sum(sum(e["selected"].values()) for e in computed) == 245760
        assert len({r for e in computed for r in e["selected"]}) == 80
        assert all(e["valid"] and len(e["execution_ranks"]) == 8 for e in computed)
        assert all(
            r[5] == 1 and r[6] > 0 for e in computed for r in e["execution_ranks"]
        )
        computed.sort(key=lambda e: e["dispatch_ns"])
        lower, upper = 0, 0
        for before, after in zip(computed, computed[1:]):
            end_lower = max(r[7] for r in before["execution_ranks"])
            end_upper = max(r[8] for r in before["execution_ranks"])
            start_lower = max(r[2] for r in after["execution_ranks"])
            start_upper = max(r[3] for r in after["execution_ranks"])
            lower += max(0, start_lower - end_upper) / 1e6
            upper += max(0, start_upper - end_lower) / 1e6
        row = dict(
            cell=path.name,
            mode=cell["mode"],
            compute_calls=len(computed),
            gpu_execution_span_total_ms=sum(
                max(r[6] for r in e["execution_ranks"]) for e in computed
            )
            / 1e6,
            intercompute_gap_lower_ms=lower,
            intercompute_gap_upper_ms=upper,
            worker_coverage=cell["worker_coverage"],
            reference_requests=cell["reference_requests"],
        )
        rows.append(row)
        groups[row["mode"]].append(row)
    aggregate = [
        dict(
            mode=mode,
            repeats=len(values),
            **{
                key: statistics.mean(r[key] for r in values)
                for key in [
                    "compute_calls",
                    "gpu_execution_span_total_ms",
                    "intercompute_gap_lower_ms",
                    "intercompute_gap_upper_ms",
                ]
            },
        )
        for mode, values in sorted(groups.items())
    ]
    correctness = {
        name: read(root / f"canary/gpu/{name}/summary.json")
        for name in ["correctness", "correctness_distinct"]
    }
    for summary in correctness.values():
        assert set(summary) == {"current", "cached_pinned", "compact_pinned"}
        assert all(r["passed"] and r["requests"] == 16 for r in summary.values())
    for r in correctness["correctness"].values():
        assert r["text_matches"] == 16
    for r in correctness["correctness_distinct"].values():
        assert r["factual_matches"] == 16
    report = dict(
        verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=provenance["commit"],
        source_sha256=provenance["source_sha256"],
        accepted_cells=9,
        measured_requests=720,
        checks="Code hashes (excluding updated documentation), distinct request IDs, "
        "tokens, frozen policies, byte counts, errors, and all-rank coverage verified",
        correctness=correctness,
        timing_aggregate=aggregate,
        timing_cells=rows,
        timing_scope="CUDA-event spans include host submission gaps, connector work "
        "and output synchronization; not isolated kernel math. "
        "Intercompute bounds use the slowest-rank boundaries.",
    )
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(accepted_cells=9, measured_requests=720, timing_aggregate=aggregate),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
