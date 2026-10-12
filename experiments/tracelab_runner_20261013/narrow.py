# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the live baseline cell, then hand off to a 32-session-only campaign."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from experiments.tracelab_runner_20261013.campaign import ROOT, SOURCE, log, save


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split(") ", 1)[1][0] != "Z"


def main():
    old_pid = json.loads((ROOT / "controller_owner.json").read_text())["pid"]
    save(ROOT / "narrow_owner.json", dict(pid=os.getpid(), old_controller=old_pid))
    result_path = ROOT / "cells/baseline_c32/result.json"
    log("scope_narrowing_queued", concurrencies=[32], preserve_live_cell=True)
    while True:
        finished = False
        if result_path.exists():
            for line in (ROOT / "process_log.jsonl").read_text().splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    entry.get("event") == "cell_finished"
                    and entry.get("arm") == "baseline"
                    and entry.get("concurrency") == 32
                ):
                    finished = True
        if finished:
            break
        if not alive(old_pid):
            raise RuntimeError("Original controller exited before baseline audit")
        time.sleep(1)
    assert json.loads(result_path.read_text())["accepted"], result_path
    if alive(old_pid):
        command = Path(f"/proc/{old_pid}/cmdline").read_bytes()
        assert b"experiments.tracelab_runner_20261013.campaign" in command
        os.kill(old_pid, signal.SIGINT)
    deadline = time.monotonic() + 300
    while alive(old_pid) and time.monotonic() < deadline:
        time.sleep(1)
    assert not alive(old_pid), "Original controller did not shut down cleanly"
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    assert not occupied, occupied
    archive = ROOT / "attempts/intentional_scope_handoff"
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("failure.json", "controller.log", "controller_owner.json"):
        path = ROOT / name
        if path.exists():
            path.rename(archive / name)
    log(
        "scope_handoff_complete",
        baseline_retained=str(result_path),
        concurrencies=[32],
        reason="User requested only 32 if native waits slow",
    )
    with (ROOT / "controller.log").open("w") as output:
        process = subprocess.Popen(
            [
                str(SOURCE / ".venv/bin/python"),
                "-m",
                "experiments.tracelab_runner_20261013.campaign",
                "--concurrencies",
                "32",
                "--arms",
                "full",
                "--resume",
            ],
            cwd=SOURCE,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
    save(ROOT / "controller_owner.json", dict(pid=process.pid, time=time.time()))
    reporter_owner = ROOT / "reporter_owner.json"
    if reporter_owner.exists():
        old_reporter = json.loads(reporter_owner.read_text())["pid"]
        if alive(old_reporter):
            command = Path(f"/proc/{old_reporter}/cmdline").read_bytes()
            assert b"experiments.tracelab_runner_20261013.report" in command
            os.kill(old_reporter, signal.SIGTERM)
    with (ROOT / "reporter_resume.log").open("w") as output:
        reporter = subprocess.Popen(
            [
                str(SOURCE / ".venv/bin/python"),
                "-m",
                "experiments.tracelab_runner_20261013.report",
                "--watch",
            ],
            cwd=SOURCE,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
    save(ROOT / "reporter_owner.json", dict(pid=reporter.pid, time=time.time()))
    save(ROOT / "narrow_complete.json", dict(time=time.time(), controller=process.pid))


if __name__ == "__main__":
    main()
