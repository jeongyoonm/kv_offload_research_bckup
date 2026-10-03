# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the completed paired study and retain compact evidence on its branch."""

import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from summarize import ROOT, SOURCE, analyzer, load, summarize


def finalize():
    for phase in ("screen", "confirmation"):
        complete = load(ROOT / phase / "complete.json")
        assert complete == dict(cells=24, accepted=24), complete
    sys.path.insert(0, str(SOURCE / "experiments/poll_prediction_20261003"))
    spec = importlib.util.spec_from_file_location(
        "poll_validate", SOURCE / "experiments/poll_prediction_20261003/validate.py"
    )
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    rows = validation.validate_root(ROOT)
    accepted = {name: row for name, row in rows.items() if row["accepted"]}
    assert len(accepted) == 48
    counts = {"p8192_f95_h100": 300, "p61440_f95_h100": 80, "p61440_f99_h100": 80}
    nominal = {
        "p8192_f95_h100": 134400,
        "p61440_f95_h100": 245760,
        "p61440_f99_h100": 51200,
    }
    timing_missing = {}
    for name, row in accepted.items():
        directory = Path(name)
        assert sum(row["cached_distribution"].values()) == counts[directory.parent.name]
        data = analyzer.analyze(directory)
        if data["missing_execution_timing_steps"]:
            timing_missing[name] = data["missing_execution_timing_steps"]
        usage = load(directory / "usage.json")
        ids = {u["request_id"] for u in usage}
        work = 0
        for line in (directory / "engine.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "step_completion":
                assert event["valid"], (directory, event["step"])
            if event["event"] == "dispatch":
                work += sum(
                    n
                    for req, n in event.get("selected", {}).items()
                    if any(request in req for request in ids)
                )
        assert work == nominal[directory.parent.name], (directory, work)
    probes = {}
    for phase in ("screen", "confirmation"):
        for name in ("correctness", "correctness_distinct"):
            values = load(ROOT / phase / name / "summary.json")
            assert len(values) == 8 and all(v["passed"] for v in values.values())
            probes[phase + "/" + name] = values
    summarize(ROOT)
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    assert not gpu, gpu
    report = dict(
        time=time.time(),
        status="verified",
        accepted_cells=48,
        measured_requests=7360,
        nominal_work_equal=True,
        all_primary_completions_valid=True,
        missing_execution_timing_steps=timing_missing,
        correctness=probes,
        gpu_processes_remaining=0,
        runtime_commit=load(ROOT / "driver_process.json")["runtime_commit"],
    )
    (ROOT / "summary/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    study = load(ROOT / "study.json")
    study.update(
        status="complete",
        accepted_cells=48,
        missing_execution_timing_steps=timing_missing,
        gpu_processes_remaining=0,
    )
    (ROOT / "study.json").write_text(json.dumps(study, indent=2) + "\n")
    evidence = SOURCE / "experiments/full_predicted_wait_20261004/results"
    evidence.mkdir(exist_ok=True)
    for name in (
        "study.json",
        "policies.json",
        "continuation.json",
        "driver_process.json",
        "native_hashes.json",
        "process_log.jsonl",
    ):
        shutil.copy2(ROOT / name, evidence / name)
    shutil.copytree(ROOT / "summary", evidence / "summary", dirs_exist_ok=True)
    for name in (
        "cpu_checks.log",
        "protocol_checks.log",
        "option_checks.log",
        "runtime_commit.log",
        "driver.log",
        "analysis_watch.log",
    ):
        with (
            (ROOT / name).open("rb") as src,
            gzip.open(evidence / (name + ".gz"), "wb") as dst,
        ):
            shutil.copyfileobj(src, dst)
    index = {}
    for name in accepted:
        directory = Path(name)
        for filename in (
            "result.json",
            "engine.jsonl",
            "late.json",
            "usage.json",
            "replay/replay_raw.jsonl",
        ):
            path = directory / filename
            index[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    (evidence / "raw_artifact_index.json").write_text(
        json.dumps(dict(root=str(ROOT), sha256=index), indent=2) + "\n"
    )
    with (ROOT / "process_log.jsonl").open("a") as log:
        log.write(
            json.dumps(dict(event="study_verified", time=time.time(), accepted=48))
            + "\n"
        )
    print("Verified 48 accepted cells; compact evidence prepared.", flush=True)


if __name__ == "__main__":
    while not (ROOT / "confirmation/complete.json").exists():
        if "driver_failed" in (ROOT / "process_log.jsonl").read_text():
            raise RuntimeError("Driver failed; retained logs, no completion claimed")
        time.sleep(20)
    finalize()
