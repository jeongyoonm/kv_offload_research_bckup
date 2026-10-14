# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay observed load-only readiness; do not infer counterfactual throughput."""

import argparse
import json
from pathlib import Path

import numpy as np


def work_rate(count, suffix, context, policy):
    if not count:
        return None
    q = np.array(
        [
            1,
            count / 16,
            count * suffix / 61440,
            context / 60000,
            suffix / 4096,
            count * suffix * context / 1e9,
        ]
    )
    duration = float(q @ policy["coefficients"])
    work = float(q[[2, 5]] @ policy["work_coefficients"])
    return work / max(work, duration) if min(work, duration) > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    calibration = json.loads(args.fit.read_text())["calibration"]
    policy = calibration["productive_load_policy"]
    support = calibration["support"]
    lo, hi = calibration["geometry_support"]["suffix_per_context"]
    episodes = []
    for path in sorted((args.root / "live").glob("*/*/result.json")):
        result = json.loads(path.read_text())
        if result["repeat"] == "warmup":
            continue
        engine = [
            json.loads(s)
            for s in (path.parent / "engine.jsonl").read_text().splitlines()
        ]
        shapes = {
            tuple(s)
            for e in engine
            if e["event"] == "step_completion" and e.get("selected")
            for s in e["request_shapes"]
        }
        if len(shapes) != 1:
            continue
        suffix, context = shapes.pop()
        if not (
            support["suffix"][0] <= suffix <= support["suffix"][1]
            and support["context"][0] <= context <= support["context"][1]
            and lo <= suffix / context <= hi
        ):
            continue
        records = json.loads((path.parent / "late.json").read_text())["records"]
        by_step = {}
        for e in records:
            if "step" in e:
                by_step.setdefault(e["step"], []).append(e)
        rows = []
        for step, events in by_step.items():
            sealed = next(
                (e for e in events if e["event"] == "membership_sealed"), None
            )
            eligibility = next(
                (e for e in events if e["event"] == "seal_eligibility"), None
            )
            if not sealed or not eligibility:
                continue
            t = sealed["transport"]
            if (
                t.get("seal_reason") != "candidate_target_ready"
                or t.get("prediction")
                or not t.get("base_ready_ns")
                or t.get("cancelled")
            ):
                continue
            ids = eligibility["candidates"]
            base = len(sealed["selected"]) - len(sealed["activated"])
            if base + len(ids) > support["requests"][1]:
                continue
            full = work_rate(base + len(ids), suffix, context, policy)
            if full is None:
                continue
            ready, ack, earliest = set(), None, None
            ready_count = None
            for e in events:
                if e["event"] == "candidate_kv_transition":
                    ready = set(e["ready"])
                elif e["event"] == "preparation_ack_transition":
                    ack = e
                if ack is None or ack["base"] != [1] * 8:
                    continue
                count = sum(
                    rid in ready and ack["candidates"][i] == [1] * 8
                    for i, rid in enumerate(ids)
                )
                if count < policy["minimum_ready_candidates"]:
                    continue
                current = work_rate(base + count, suffix, context, policy)
                if (
                    current is not None
                    and current / full >= policy["minimum_efficiency"]
                ):
                    earliest = max(e["time_ns"], t["base_ready_ns"])
                    ready_count = count
                    break
            if earliest is not None:
                rows.append(
                    dict(
                        step=step,
                        base=base,
                        published=len(ids),
                        ready=ready_count,
                        observed_wait_saving_ms=max(0, t["seal_ns"] - earliest) / 1e6,
                    )
                )
        episodes.append(
            dict(
                workload=result["workload"],
                arm=result["label"],
                repeat=result["repeat"],
                rows=rows,
                summed_observed_wait_saving_ms=sum(
                    r["observed_wait_saving_ms"] for r in rows
                ),
            )
        )
    payload = dict(
        episodes=episodes,
        efficiency=policy["minimum_efficiency"],
        caveat="Replay uses future trace only for analysis. These are "
        "observed eligibility windows, not additive GPU gap savings "
        "or a throughput prediction. Earlier dispatch changes later "
        "loads, batching, and closed-loop arrivals. Homogeneous "
        "supported episodes only; cancellation excluded.",
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in e.items() if k != "rows"} for e in episodes]))


if __name__ == "__main__":
    main()
