# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exclude first-use/JIT labels and audit unrelated-transfer overlap."""

import datetime
import json
import time
from collections import Counter
from pathlib import Path

import regex as re

ROOT = Path("/mnt/localnvme/glm53_dataset_coverage_20261007")


def inference_jit_events():
    jit = []
    for log in (ROOT / "server").glob("server*.log"):
        for line in log.read_text(errors="replace").splitlines():
            if "during inference:" not in line:
                continue
            stamp = re.search(r"(\d\d-\d\d \d\d:\d\d:\d\d)", line)
            if stamp:
                epoch = datetime.datetime.strptime(
                    "2026-" + stamp[1], "%Y-%m-%d %H:%M:%S"
                ).timestamp()
                jit.append((epoch, line))
    return jit


def audit():
    wall_offset = time.time() - time.monotonic()
    jit = inference_jit_events()
    summaries = Counter()
    records = []
    for path in sorted((ROOT / "cells").rglob("result.json")):
        result = json.loads(path.read_text())
        meta = json.loads((path.parent / "family.json").read_text())
        responses = result.get("responses", [])
        if not responses:
            continue
        begin = min(r["started_ns"] for r in responses) / 1e9 + wall_offset
        end = max(r["ended_ns"] for r in responses) / 1e9 + wall_offset
        compile_events = [line for t, line in jit if begin - 1 <= t <= end + 1]
        # Warn-once JIT logging cannot certify every subsequent shape. Discard
        # the first training occurrence, and the first novel development seed.
        warmup = (meta["split"] == "train" and meta["repeat"] == 0) or (
            meta["split"] == "development" and meta["seed"] == 801
        )
        clean = result["accepted"] and not compile_events and not warmup
        reason = (
            "rejected_cell"
            if not result["accepted"]
            else "first_use_warmup"
            if warmup
            else "jit_in_measurement"
            if compile_events
            else "clean"
        )
        events = [
            json.loads(line)
            for line in (path.parent / "engine.jsonl").read_text().splitlines()
        ]
        jobs = {e["job_id"]: e for e in events if e["event"] == "kv_load_dispatched"}
        done = {
            e["job_id"]: e for e in events if e["event"] == "kv_all_rank_completion"
        }
        overlaps, own_waits = [], []
        for e in events:
            if e["event"] != "step_completion" or not e.get("valid"):
                continue
            ranks = e.get("execution_ranks", [])
            if len(ranks) != 8 or not e.get("selected"):
                continue
            start = max(r[3] for r in ranks)
            end_ns = max(r[3] + r[6] for r in ranks)
            own = set(e["selected"])
            overlap_tokens = 0
            for key, job in jobs.items():
                complete = done.get(key, {}).get("completion_ns", 0)
                if job["request"] in own and (not complete or complete > start):
                    own_waits.append(dict(step=e["step"], job=key))
                if (
                    "coverage-bg-" in job["request"]
                    and job["dispatch_ns"] < end_ns
                    and (not complete or complete > start)
                ):
                    overlap_tokens += job["external_tokens"]
            overlaps.append(dict(step=e["step"], overlapping_tokens=overlap_tokens))
        if own_waits:
            clean = False
            reason = "own_KV_not_confirmed_before_compute"
        record = dict(
            path=str(path.parent),
            usable=clean,
            reason=reason,
            jit_events=compile_events,
            own_waits=own_waits,
            transfer_overlap=overlaps,
            block=meta["block"],
            split=meta["split"],
        )
        (path.parent / "measurement_audit.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        summaries[f"{meta['split']}:{reason}"] += 1
        records.append(record)
    (ROOT / "measurement_audit.json").write_text(
        json.dumps(dict(counts=summaries, cells=records), indent=2) + "\n"
    )
    print(json.dumps(dict(event="measurement_audit", counts=summaries)), flush=True)


if __name__ == "__main__":
    audit()
