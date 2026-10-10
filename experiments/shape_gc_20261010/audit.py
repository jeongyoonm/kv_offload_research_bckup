# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correlate observed GC intervals with sealing and physical compute gaps."""

import argparse
import json
from pathlib import Path

import regex as re

from experiments.native_scheduler_ab_20261010.report import union

PATTERN = re.compile(
    r"\((?P<owner>[^)]+) pid=(?P<pid>\d+)\).*GC timing "
    r"start_ns=(?P<start>\d+) end_ns=(?P<end>\d+) "
    r"duration_ms=(?P<duration>[\d.]+) generation=(?P<generation>\d+)"
)


def overlap(begin, end, intervals):
    return sum(max(0, min(end, b) - max(begin, a)) for a, b in intervals) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pauses = []
    for line in (args.root / "server/server.log").read_text().splitlines():
        match = PATTERN.search(line)
        if match:
            values = match.groupdict()
            pauses.append(
                dict(
                    owner=values["owner"],
                    pid=int(values["pid"]),
                    begin=int(values["start"]),
                    end=int(values["end"]),
                    generation=int(values["generation"]),
                )
            )
    engine = union(
        [(p["begin"], p["end"]) for p in pauses if p["owner"] == "EngineCore"]
    )
    episodes = []
    for path in sorted((args.root / "live").glob("*/*/result.json")):
        result = json.loads(path.read_text())
        if result["repeat"] == "warmup":
            continue
        events = [
            json.loads(s)
            for s in (path.parent / "engine.jsonl").read_text().splitlines()
        ]
        compute = union(
            [
                (
                    max(r[3] for r in e["execution_ranks"]),
                    max(r[3] + r[6] for r in e["execution_ranks"]),
                )
                for e in events
                if e["event"] == "step_completion" and e.get("selected")
            ]
        )
        assert compute
        gaps = [(a[1], b[0]) for a, b in zip(compute, compute[1:])]
        records = json.loads((path.parent / "late.json").read_text())["records"]
        finalizations = []
        for event in records:
            if event["event"] != "membership_sealed":
                continue
            begin, end = event["transport"]["seal_ns"], event["time_ns"]
            assert begin <= end
            finalizations.append(
                dict(
                    step=event["step"],
                    duration_ms=(end - begin) / 1e6,
                    core_gc_overlap_ms=overlap(begin, end, engine),
                )
            )
        episodes.append(
            dict(
                workload=result["workload"],
                arm=result["label"],
                repeat=result["repeat"],
                inter_compute_gap_ms=sum(b - a for a, b in gaps) / 1e6,
                core_gc_in_gap_ms=sum(overlap(a, b, engine) for a, b in gaps),
                core_gc_during_compute_ms=sum(
                    overlap(a, b, engine) for a, b in compute
                ),
                scheduler_finalization_ms=sum(e["duration_ms"] for e in finalizations),
                core_gc_in_finalization_ms=sum(
                    e["core_gc_overlap_ms"] for e in finalizations
                ),
                largest_finalizations=sorted(
                    finalizations, key=lambda e: e["duration_ms"], reverse=True
                )[:10],
            )
        )
    payload = dict(
        pauses=pauses,
        episodes=episodes,
        scope="Observational overlap, not causal attribution. Core and worker "
        "pauses remain distinct. Only pauses >=10ms were logged; native "
        "execution endpoints include output copy. No collector-policy change.",
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(dict(pauses=len(pauses), episodes=len(episodes))))


if __name__ == "__main__":
    main()
