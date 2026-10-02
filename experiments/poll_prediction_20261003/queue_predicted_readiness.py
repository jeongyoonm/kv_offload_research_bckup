"""Serialize the additional mechanism evaluation after the original server exits."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/localnvme/glm53_poll_prediction_20261003")
SOURCE = Path(__file__).resolve().parents[2]
started = time.monotonic()
while not (ROOT / "confirmation/complete.json").exists():
    if "Traceback (most recent call last)" in (ROOT / "driver.log").read_text():
        raise RuntimeError("Original driver failed; new GPU run not started")
    if time.monotonic() - started > 6 * 3600:
        raise TimeoutError("Waiting for original server shutdown")
    time.sleep(20)
command = [
    str(SOURCE / ".venv/bin/python"),
    "experiments/poll_prediction_20261003/predicted_readiness_run.py",
    "--name",
    "predicted_readiness",
    "--policies",
    "fixed,prediction_1ms,readiness_2ms,predicted_readiness_50ms",
    "--workloads",
    "8192:95:100,61440:95:100,61440:99:100",
    "--workload-request-counts",
    "300,80,80",
    "--max-model-len",
    "65536",
    "--distinct-fact-probe",
]
with (ROOT / "process_log.jsonl").open("a") as log:
    log.write(
        json.dumps(
            dict(
                event="predicted_readiness_queued_run_start",
                time=time.time(),
                command=command,
            )
        )
        + "\n"
    )
with (ROOT / "predicted_readiness_driver.log").open("w") as log:
    result = subprocess.run(
        command,
        cwd=SOURCE,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(os.environ, TMPDIR=str(ROOT / "tmp")),
    )
with (ROOT / "process_log.jsonl").open("a") as log:
    log.write(
        json.dumps(
            dict(
                event="predicted_readiness_queued_run_exit",
                time=time.time(),
                exit_code=result.returncode,
            )
        )
        + "\n"
    )
sys.exit(result.returncode)
