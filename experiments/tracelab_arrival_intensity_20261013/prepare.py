# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scale synthetic session arrivals while verifying unchanged trace contents."""

import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_arrival_intensity_20261013")
ORIGINAL = Path("/mnt/localnvme/jeongyoon/tracelab_request_slots_20261013/workload.csv")
RATES = ("0.05", "0.075", "0.1", "0.15", "0.2", "0.25")


def content_hash(path):
    digest = hashlib.sha256()
    sessions = set()
    count = 0
    waits = Decimal(0)
    with path.open() as stream:
        reader = csv.DictReader(stream)
        fields = [key for key in reader.fieldnames if key != "arrival_time"]
        for row in reader:
            digest.update(json.dumps([row[key] for key in fields]).encode() + b"\n")
            count += 1
            sessions.add(row["id"])
            waits += Decimal(row["tool_wait_after_ms"])
    return dict(
        nonarrival_sha256=digest.hexdigest(),
        calls=count,
        sessions=len(sessions),
        tool_wait_ms=str(waits),
    )


def main():
    destination = ROOT / "workloads"
    destination.mkdir(exist_ok=False)
    original = content_hash(ORIGINAL)
    assert original["calls"] == 665453 and original["sessions"] == 8058
    artifacts = {}
    for rate in RATES:
        path = destination / f"rate_{rate}.csv"
        with ORIGINAL.open() as source, path.open("w") as output:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                row["arrival_time"] = format(
                    Decimal(row["arrival_time"]) / Decimal(rate), ".6f"
                )
                writer.writerow(row)
        evidence = content_hash(path)
        assert evidence == original
        artifacts[rate] = dict(
            path=str(path),
            csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            **evidence,
        )
    manifest = dict(
        original=str(ORIGINAL),
        original_csv_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
        original_content=original,
        variants=artifacts,
        changed="Only synthetic session arrival_time is divided by arrival rate.",
        preserved=(
            "Session IDs and order, every round, prefix/input/output lengths, "
            "tool waits, source mix, random seed, and native prompt construction."
        ),
        sampling="All 8058 sessions and 665453 calls retained; no windows or sampling.",
    )
    (ROOT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
