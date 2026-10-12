# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze the full released session catalogue with native exporter timing."""

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import duckdb

ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_request_slots_20261013")
UPSTREAM = Path("/mnt/localnvme/jeongyoon/tracelab_request_slots_src_20261013")
DB = Path(
    "/mnt/localnvme/jeongyoon/trace_characterization_20261012/tracelab/"
    "tracelab_v0.0.2.duckdb"
)


def main():
    assert not (ROOT / "workload.csv").exists()
    command = [
        sys.executable,
        str(UPSTREAM / "artifacts/trace_facts/csv_export/convert.py"),
        "--db",
        str(DB),
        "-o",
        str(ROOT / "workload.csv"),
    ]
    with (ROOT / "export.log").open("w") as output:
        subprocess.run(command, cwd=UPSTREAM, stdout=output, stderr=output, check=True)
    sessions, rows, max_total, total_wait = Counter(), 0, 0, 0.0
    with (ROOT / "workload.csv").open() as stream:
        for row in csv.DictReader(stream):
            sessions[row["id"]] += 1
            rows += 1
            total = sum(
                int(row[key]) for key in ("prefix_len", "input_len", "output_len")
            )
            max_total = max(total, max_total)
            assert total <= 1048576, row
            total_wait += float(row["tool_wait_after_ms"])
    con = duckdb.connect(str(DB), read_only=True)
    source_counts = con.execute(
        "select provider,count(distinct session_id),count(*) from rounds "
        "group by provider order by provider"
    ).fetchall()
    zeros = con.execute(
        "select count(*) from rounds where input_tokens_total=0 or output_tokens=0"
    ).fetchone()[0]
    con.close()
    assert len(sessions) == 8058 and rows == 665453
    manifest = dict(
        sessions=len(sessions),
        calls=rows,
        source_counts=source_counts,
        source_sha256=hashlib.sha256(DB.read_bytes()).hexdigest(),
        csv_sha256=hashlib.sha256((ROOT / "workload.csv").read_bytes()).hexdigest(),
        exporter_command=command,
        max_input_plus_output=max_total,
        total_recorded_tool_wait_seconds=total_wait / 1000,
        zero_length_source_rows=zeros,
        zero_handling="Native exporter maps append/output lengths to minimum 1",
        sampling="All released sessions and calls; no windows or session sampling",
        timing="Native defaults: shuffled seed 0, Poisson arrivals 1 session/s, "
        "summed wall tool waits; no clipping or scaling",
        evaluation="Bounded 600s warmup + 1200s measurement. Session arrivals "
        "beyond 1800s and not-yet-submitted calls remain unissued. "
        "Calls already submitted drain to completion.",
    )
    (ROOT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
