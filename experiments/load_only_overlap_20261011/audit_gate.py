# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-correctness gate for synchronized audit cells, separate from performance."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = args.root / "live/short16k/rwarmup_fullsafe/result.json"
    cell = json.loads(path.read_text())
    assert cell["frozen_unchanged"]
    assert cell["observed_cached"] == cell["expected_cached"]
    assert cell["requests"] == 32 and len(cell["responses"]) == 32
    assert all(r["usage"]["completion_tokens"] == 1 for r in cell["responses"])
    for key in ("vllm:num_preemptions_total", "vllm:prefix_cache_hits_total"):
        assert cell["metrics_before"].get(key, 0) == cell["metrics_after"].get(key, 0)
    ranks = []
    for rank in range(8):
        path = args.root / "dma_audit" / f"rank{rank}.jsonl"
        totals = dict.fromkeys(("H2D", "D2H", "before_forward"), 0)
        counts = dict.fromkeys(totals, 0)
        for line in path.open():
            row = json.loads(line)
            assert row.get("mismatched_bytes", 0) == 0
            assert row.get("changed_cpu_blocks", 0) == 0
            totals[row["direction"]] += row["verified_bytes"]
            counts[row["direction"]] += 1
        assert all(value > 0 for value in totals.values())
        assert counts["H2D"] == 32
        assert totals["H2D"] == totals["before_forward"]
        ranks.append(dict(rank=rank, verified_bytes=totals, records=counts))
    tracking = []
    for path in (args.root / "server/timing/queue_stages").glob("*.json"):
        data = json.loads(path.read_text())
        tracking.extend(
            e for e in data["events"] if e["event"] == "load_block_dependencies"
        )
    assert len(tracking) == 8 * 32
    assert {e["rank"] for e in tracking} == set(range(8))
    args.output.write_text(
        json.dumps(
            dict(
                byte_correctness_accepted=True,
                original_performance_acceptance=cell["accepted"],
                preparation_coverage_required_for_performance=True,
                diagnostic_preparation_coverage=cell["optimized_preparation_verified"],
                scope=(
                    "Synchronized byte validation distorts timing and "
                    "preparation coverage. No throughput or preparation overlap "
                    "claim from this diagnostic cell."
                ),
                requests=32,
                all_rank_dependency_records=len(tracking),
                ranks=ranks,
            ),
            indent=2,
        )
    )
    print("All-rank transfer and pre-forward byte correctness passed", flush=True)


if __name__ == "__main__":
    main()
