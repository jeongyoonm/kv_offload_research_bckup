# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit rank-specific execution tails without changing the measured endpoint."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    episodes = []
    for path in sorted((args.root / "live").glob("*/*/result.json")):
        result = json.loads(path.read_text())
        if result["repeat"] == "warmup":
            continue
        rows = []
        for line in (path.parent / "engine.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["event"] != "step_completion" or not e.get("selected"):
                continue
            ranks = e["execution_ranks"]
            assert e["valid"] and len(ranks) == 8
            ends = [r[3] + r[6] for r in ranks]
            start = max(r[3] for r in ranks)
            primary = e["ranks"]
            rows.append(
                dict(
                    step=e["step"],
                    requests=len(e["selected"]),
                    max_suffix=max(s[0] for s in e["request_shapes"]),
                    envelope_ms=(max(ends) - start) / 1e6,
                    maximum_rank=ends.index(max(ends)),
                    maximum_minus_median_end_ms=(max(ends) - statistics.median(ends))
                    / 1e6,
                    primary_submission_after_start_ms=[
                        (r[3] - start) / 1e6 for r in primary
                    ],
                    elapsed_ms=[r[6] / 1e6 for r in ranks],
                )
            )
        episodes.append(
            dict(
                workload=result["workload"],
                arm=result["label"],
                repeat=result["repeat"],
                compute_rounds=len(rows),
                rank_spread_over_50ms=sum(
                    r["maximum_minus_median_end_ms"] > 50 for r in rows
                ),
                rank0_spread_over_50ms=sum(
                    r["maximum_rank"] == 0 and r["maximum_minus_median_end_ms"] > 50
                    for r in rows
                ),
                rows=rows,
            )
        )
    payload = dict(
        partial=not (args.root / "summary.json").exists(),
        episodes=episodes,
        caveat="The current end event joins output-copy completion. Rank spread "
        "alone cannot distinguish model compute, DMA, sampling, or observer "
        "delay; this audit establishes no root cause.",
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in e.items() if k != "rows"} for e in episodes]))


if __name__ == "__main__":
    main()
