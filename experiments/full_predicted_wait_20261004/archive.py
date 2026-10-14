# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retain final comparisons, provenance, probes, and all study process logs."""

import gzip
import shutil
import time

from compare import compare
from summarize import ROOT, SOURCE, load, summarize


def archive():
    assert load(ROOT / "study.json")["status"] == "complete"
    assert load(ROOT / "summary/verification.json")["status"] == "verified"
    summarize(ROOT)
    compare()
    evidence = SOURCE / "experiments/full_predicted_wait_20261004/results"
    shutil.copytree(ROOT / "summary", evidence / "summary", dirs_exist_ok=True)
    for path in ROOT.glob("*.json"):
        shutil.copy2(path, evidence / path.name)
    shutil.copy2(ROOT / "process_log.jsonl", evidence / "process_log.jsonl")
    for path in ROOT.glob("*.log"):
        with (
            path.open("rb") as src,
            gzip.open(evidence / (path.name + ".gz"), "wb") as dst,
        ):
            shutil.copyfileobj(src, dst)
    for phase in ("screen", "confirmation"):
        target = evidence / phase
        target.mkdir(exist_ok=True)
        for name in ("provenance.json", "protocol.json", "complete.json"):
            path = ROOT / phase / name
            if path.exists():
                shutil.copy2(path, target / name)
        for kind in ("correctness", "correctness_distinct"):
            shutil.copy2(
                ROOT / phase / kind / "summary.json", target / (kind + ".json")
            )
    server = ROOT / "screen/server"
    for name in ("command.json", "gpu_processes_after_stop.csv"):
        path = server / name
        if path.exists():
            shutil.copy2(path, evidence / name)
    with (
        (server / "server.log").open("rb") as src,
        gzip.open(evidence / "server.log.gz", "wb") as dst,
    ):
        shutil.copyfileobj(src, dst)
    print(f"Archived final evidence at {time.time()} to {evidence}")


if __name__ == "__main__":
    archive()
