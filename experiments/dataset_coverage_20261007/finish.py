# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Continue a completed collection through CPU fits and budgeted live confirmation."""

import fcntl
import json
import os
import signal
import subprocess
import time

import run


def command(name, python, arguments):
    directory = run.SOURCE / "experiments/dataset_coverage_20261007"
    run.log("phase_start", phase=name, arguments=arguments)
    with (run.ROOT / f"{name}.log").open("a") as stream:
        result = subprocess.run(
            [str(python), str(directory / arguments[0]), *arguments[1:]],
            cwd=run.SOURCE,
            env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1"),
            stdout=stream,
            stderr=stream,
        )
    run.log("phase_end", phase=name, returncode=result.returncode)
    result.check_returncode()


def main():
    lock = (run.ROOT / "finish.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        events = [
            json.loads(line)
            for line in (run.ROOT / "process_log.jsonl").read_text().splitlines()
        ]
        latest_start = max(
            i for i, e in enumerate(events) if e["event"] == "server_started"
        )
        recent = events[latest_start:]
        if any(e["event"] == "stopped" for e in recent):
            if not any(e["event"] == "shadow_test_complete" for e in recent):
                raise RuntimeError("collection stopped before the shadow-test phase")
            break
        server = json.loads((run.ROOT / "server/process.json").read_text())["pid"]
        pids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        )
        foreign = []
        for pid in {int(line.strip()) for line in pids.splitlines() if line.strip()}:
            try:
                if os.getpgid(pid) != server:
                    foreign.append(pid)
            except ProcessLookupError:
                pass
        with (run.ROOT / "gpu_ownership.jsonl").open("a") as stream:
            stream.write(
                json.dumps(dict(time=time.time(), server=server, foreign=foreign))
                + "\n"
            )
        if foreign:
            run.log("foreign_gpu_process", pids=foreign)
            os.killpg(server, signal.SIGTERM)
            raise RuntimeError("foreign GPU process contaminated collection")
        time.sleep(30)
    cpu_python = run.HIST / "eval-venv/bin/python"
    serving_python = run.SOURCE / ".venv/bin/python"
    command("audit", serving_python, ["audit.py"])
    command("normalize", serving_python, ["run.py", "--normalize"])
    command("development", cpu_python, ["study.py"])
    command("final", cpu_python, ["study.py", "--final"])
    command("live", serving_python, ["live.py"])
    command("report", cpu_python, ["report.py"])
    run.log("study_complete")


if __name__ == "__main__":
    main()
