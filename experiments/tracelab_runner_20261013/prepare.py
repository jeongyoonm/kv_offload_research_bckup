# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Export the previous frozen cohort using TraceLab's unmodified converter."""

import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_runner_20261013")
UPSTREAM = Path("/mnt/localnvme/jeongyoon/tracelab_replay_review_20261013")
PREVIOUS = Path("/mnt/localnvme/jeongyoon/agentic_1m_20261012/datasets")
SOURCE = Path(
    "/mnt/localnvme/jeongyoon/trace_characterization_20261012/tracelab/tracelab_v0.0.2.duckdb"
)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    pks = []
    windows = []
    for provider in ("codex", "claude"):
        data = json.loads((PREVIOUS / f"tracelab_{provider}_windows.json").read_text())
        for session in data["sessions"]:
            windows.append(
                dict(
                    provider=provider,
                    session=session["session"],
                    source_start=session["source_start"],
                    source_requests=session["source_requests"],
                    requests=len(session["requests"]),
                )
            )
            pks.extend(row["round_pk"] for row in session["requests"])
    assert len(pks) == len(set(pks)) == 1880
    path = ROOT / "selected.duckdb"
    assert not path.exists(), path
    con = duckdb.connect(str(path))
    con.execute(
        "ATTACH '" + str(SOURCE).replace("'", "''") + "' AS original (READ_ONLY)"
    )
    con.execute("CREATE TABLE selected_pks (round_pk BIGINT)")
    con.executemany("INSERT INTO selected_pks VALUES (?)", [(pk,) for pk in pks])
    for table in ("rounds", "timing_events", "tool_calls"):
        con.execute(
            f"CREATE TABLE {table} AS SELECT t.* FROM original.{table} t "
            "JOIN selected_pks s USING (round_pk)"
        )
    con.execute("CREATE TABLE trace_source AS SELECT * FROM original.trace_source")
    assert (
        con.execute("SELECT count(DISTINCT session_id) FROM rounds").fetchone()[0] == 64
    )
    con.close()
    csv_path = ROOT / "workload.csv"
    command = [
        sys.executable,
        str(UPSTREAM / "artifacts/trace_facts/csv_export/convert.py"),
        "--db",
        str(path),
        "-o",
        str(csv_path),
    ]
    with (ROOT / "export.log").open("w") as output:
        subprocess.run(command, cwd=UPSTREAM, stdout=output, stderr=output, check=True)
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1880
    waits = defaultdict(float)
    for row in rows:
        waits[row["id"]] += float(row["tool_wait_after_ms"]) / 1000
        assert (
            int(row["prefix_len"]) + int(row["input_len"]) + int(row["output_len"])
            <= 1048576
        )
    manifest = dict(
        upstream_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
        ).strip(),
        exporter_command=command,
        csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        sessions=len(waits),
        requests=len(rows),
        windows=windows,
        tool_wait_seconds_total=sum(waits.values()),
        tool_wait_seconds_longest_session=max(waits.values()),
        tool_wait_seconds_by_session=dict(waits),
        largest_single_wait_seconds=max(
            float(r["tool_wait_after_ms"]) / 1000 for r in rows
        ),
        concurrency_definition=(
            "Unmodified runner max_active_sessions: permit held "
            "for complete session including waits"
        ),
        timing=(
            "Unmodified converter defaults: seeded Poisson session arrivals "
            "at 1/s, seed 0, summed wall tool waits; unmodified runner sleeps"
        ),
        scope=(
            "Same frozen 64-session/1880-round windows as previous comparison; "
            "complete request lengths; no delay scaling or wait clipping"
        ),
    )
    (ROOT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in manifest.items()
                if k not in ("windows", "tool_wait_seconds_by_session")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
