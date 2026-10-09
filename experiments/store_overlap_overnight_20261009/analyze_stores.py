# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize store evidence by episode and rank without summing overlap twice."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def stats(values):
    if not values:
        return None
    return dict(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        min=min(values),
        max=max(values),
    )


def analyze(root):
    exports = defaultdict(list)
    for path in (root / "server/timing/queue_stages").glob("*.json"):
        data = json.loads(path.read_text())
        assert not data["overflow"], path
        if data.get("rank", -1) >= 0:
            exports[data["generation"], data["rank"]].extend(data["events"])
    rows = []
    for path in sorted((root / "live").rglob("result.json")):
        result = json.loads(path.read_text())
        if not isinstance(result["repeat"], int):
            continue
        generations = json.loads((path.parent / "late.json").read_text())["generations"]
        engine = [
            json.loads(line)
            for line in (path.parent / "engine.jsonl").read_text().splitlines()
        ]
        compute = sorted(
            [
                event
                for event in engine
                if event["event"] == "step_completion" and event.get("selected")
            ],
            key=lambda event: event["step"],
        )
        gaps = [
            (
                max(rank[3] + rank[6] for rank in previous["execution_ranks"]),
                max(rank[3] for rank in current["execution_ranks"]),
            )
            for previous, current in zip(compute, compute[1:])
        ]
        per_rank = {}
        for rank in range(8):
            events = sorted(
                [
                    event
                    for generation in generations
                    for event in exports[generation, rank]
                ],
                key=lambda event: event["time_ns"],
            )
            pending = {}
            spans = defaultdict(list)
            descriptor_ratios, descriptors, original_descriptors = [], [], []
            producer_states = Counter()
            allocation = defaultdict(list)
            load_descriptors = defaultdict(list)
            load_ready_states = Counter()
            for event in events:
                name = event["event"]
                if name == "load_ready_dependency":
                    load_ready_states[str(event["current_stream_ready"])] += 1
                if name == "store_specific_producer_dependency":
                    producer_states[
                        f"producer_ready={event['producer_ready']},"
                        f"stream_ready={event['current_stream_ready']}"
                    ] += 1
                if name == "store_descriptors_end":
                    count = event["num_descriptors"]
                    original = event.get("original_descriptors", count)
                    descriptors.append(count)
                    original_descriptors.append(original)
                    descriptor_ratios.append(count / original if original else 1)
                if name == "store_allocation_audit":
                    for record in event.get("allocations", []):
                        for field, value in record.items():
                            if isinstance(value, (int, float)):
                                allocation[field].append(value)
                if name in ("load_descriptors_coalesced", "load_descriptors_prepared"):
                    for field in (
                        "original_descriptors",
                        "num_descriptors",
                        "max_descriptor_bytes",
                        "num_bytes",
                    ):
                        load_descriptors[field].append(event[field])
                if name.startswith("store_") and name.endswith("_begin"):
                    pending[name[:-6], event.get("job_id")] = event["time_ns"]
                elif name.startswith("store_") and name.endswith("_end"):
                    key = name[:-4], event.get("job_id")
                    if key in pending:
                        spans[key[0]].append((pending.pop(key), event["time_ns"]))
            assert not pending, (path, rank, pending)
            per_rank[str(rank)] = dict(
                durations_ms={
                    key: stats([(end - start) / 1e6 for start, end in values])
                    for key, values in spans.items()
                },
                exposed_inter_compute_ms={
                    key: sum(
                        max(0, min(end, gap_end) - max(start, gap_start))
                        for start, end in values
                        for gap_start, gap_end in gaps
                    )
                    / 1e6
                    for key, values in spans.items()
                },
                descriptors=stats(descriptors),
                original_descriptors=stats(original_descriptors),
                descriptor_ratio=stats(descriptor_ratios),
                producer_states=dict(producer_states),
                allocation={key: stats(values) for key, values in allocation.items()},
                allocation_totals={
                    key: sum(values)
                    for key, values in allocation.items()
                    if key
                    in (
                        "requested",
                        "scanned",
                        "skipped_pending",
                        "allocated_pending",
                        "duration_ns",
                    )
                },
                load_descriptors={
                    key: stats(values) for key, values in load_descriptors.items()
                },
                load_ready_states=dict(load_ready_states),
            )
        rows.append(
            dict(
                arm=result["label"],
                workload=result["workload"],
                repeat=result["repeat"],
                generations=generations,
                ranks=per_rank,
            )
        )
    output = dict(
        episodes=rows,
        definitions="Durations include all store markers in each measured policy "
        "generation, including final drain. Exposed times intersect conservative "
        "all-rank inter-compute gaps. Nested markers overlap: do not sum them as "
        "an additive latency breakdown. Producer/stream readiness is a CPU-side "
        "snapshot at dependency submission. Allocation counters describe the "
        "selection from already-known pending stores only.",
    )
    (root / "store_diagnostics.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    output = analyze(args.root)
    print(json.dumps(dict(episodes=len(output["episodes"]), root=str(args.root))))


if __name__ == "__main__":
    main()
