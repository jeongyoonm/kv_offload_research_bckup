# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recover native session provenance and report realized fixed-window workloads."""

import csv
import json
from collections import Counter

import duckdb
import numpy as np

from experiments.tracelab_arrival_intensity_20261013.prepare import ORIGINAL, ROOT
from experiments.tracelab_request_slots_20261013.prepare import DB


def metadata():
    connection = duckdb.connect(str(DB), read_only=True)
    records = connection.execute(
        "SELECT session_id,provider,newly_append_tokens,prefix_tokens,output_tokens "
        "FROM (SELECT *,row_number() OVER (PARTITION BY session_id ORDER BY "
        "round_index,round_pk) AS position,min(ingest_seq) OVER (PARTITION BY "
        "session_id) AS first_ingest FROM rounds) WHERE position=1 "
        "ORDER BY first_ingest"
    ).fetchall()
    connection.close()
    assert len(records) == 8058
    order = np.random.default_rng(0).permutation(len(records))
    mapping = {}
    for index, source in enumerate(order):
        name, provider, suffix, prefix, output = records[int(source)]
        mapping[str(index)] = dict(
            original_session_id=name,
            provider=provider,
            first_input_len=max(1, int(suffix)),
            first_prefix_len=max(0, int(prefix)),
            first_output_len=max(1, int(output)),
        )
    count = 0
    with ORIGINAL.open() as stream:
        for row in csv.DictReader(stream):
            if int(row["round_idx"]) != 0:
                continue
            item = mapping[row["id"]]
            for field in ("input_len", "prefix_len", "output_len"):
                assert int(row[field]) == item["first_" + field], row
            count += 1
    assert count == len(mapping)
    (ROOT / "session_metadata.json").write_text(json.dumps(mapping, indent=2) + "\n")
    return mapping


def summarize():
    mapping = json.loads((ROOT / "session_metadata.json").read_text())
    output = {}
    for path in sorted((ROOT / "results").glob("*.json")):
        result = json.loads(path.read_text())
        rows = [
            json.loads(line)
            for line in (
                ROOT
                / "episodes"
                / result["episode"]
                / "cells"
                / result["arm"]
                / "rounds.jsonl"
            ).open()
        ]
        begin = result["replay_start_timestamp"] + result["warmup_seconds"]
        end = begin + result["measurement_seconds"]
        posted = [row for row in rows if begin <= row["post_timestamp"] < end]
        distributions = {}
        for name, field in (
            ("total_input_tokens", "prompt_len"),
            ("recorded_prefix_tokens", "prefix_len"),
            ("recorded_append_tokens", "input_len"),
            ("computed_prompt_tokens", "server_uncached_prompt_tokens"),
            ("output_tokens", "output_len_target"),
        ):
            values = [row[field] for row in posted]
            distributions[name] = dict(
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                p95=float(np.percentile(values, 95)),
                min=min(values),
                max=max(values),
                sum=sum(values),
            )
        output[result["episode"]] = dict(
            post_cohort_requests=len(posted),
            sessions=len({row["session_id"] for row in posted}),
            provider_requests=dict(
                Counter(mapping[row["session_id"]]["provider"] for row in posted)
            ),
            lengths=distributions,
        )
    (ROOT / "realized_workload_mix.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    return output


if __name__ == "__main__":
    if not (ROOT / "session_metadata.json").exists():
        metadata()
    print(json.dumps(summarize(), indent=2))
