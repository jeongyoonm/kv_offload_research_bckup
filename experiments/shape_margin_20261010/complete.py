# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Include scheduler finalization in the frozen successor setup allowance."""

import argparse
import copy
import json
from pathlib import Path

from experiments.shape_margin_20261010.audit import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    audit = copy.deepcopy(json.loads(args.audit.read_text()))
    rows = audit["setup_rows"]
    seals = {}
    for root in {Path(r["source"]).parent for r in rows}:
        for path in (root / "live").glob("*/*/late.json"):
            for event in json.loads(path.read_text())["records"]:
                if event["event"] == "membership_sealed":
                    key = (str(root), event["step"])
                    assert key not in seals
                    seals[key] = event
    bins = {}
    for r in rows:
        e = seals[str(Path(r["source"]).parent), r["step"]]
        seal_ns = e["transport"].get("seal_ns")
        assert seal_ns is not None and e["time_ns"] >= seal_ns
        r["post_seal_scheduler_ms"] = (e["time_ns"] - seal_ns) / 1e6
        r["seal_to_forward_cpu_ms"] = (
            r["post_seal_scheduler_ms"] + r["entry_to_forward_ms"]
        )
        bins.setdefault(r["shape_bin"], []).append(r["seal_to_forward_cpu_ms"])
    audit["host_setup_complete_by_B_shape"] = {k: summarize(v) for k, v in bins.items()}
    audit["definitions"] += (
        " Complete CPU setup adds scheduler snapshot-to-final-output work to "
        "worker-entry-to-forward work. It excludes FIFO wait and deliberate "
        "seal hold; the existing 2ms transport allowance remains. This is an "
        "empirical margin, not a guaranteed bound on host or device stalls."
    )
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit["host_setup_complete_by_B_shape"], indent=2))


if __name__ == "__main__":
    main()
