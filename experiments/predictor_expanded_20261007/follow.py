# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run CPU exploration after the current plan, within a persisted deadline."""

import fcntl
import json
import os
import signal
import subprocess
import time

import study


def log(event, **fields):
    record = dict(event=event, time=time.time(), **fields)
    with (study.ROOT / "process_log.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def unfinished(name, deadline):
    completed = len(list((study.ROOT / "folds").glob("*/completed.json")))
    improvements = len(list((study.ROOT / "improvements").glob("*/completed.json")))
    study.arch.dump(
        study.ROOT / "status.json",
        dict(
            phase="budget_stopped",
            interrupted_phase=name,
            completed_architecture_tasks=completed,
            completed_improvement_tasks=improvements,
            deadline=deadline,
        ),
    )
    (study.ROOT / "UNFINISHED.md").write_text(
        f"# Exploration stopped at its time budget\n\nInterrupted phase: {name}.\n\n"
        f"Completed architecture tasks: {completed}; improvement tasks: "
        f"{improvements}. Saved checkpoints and predictions remain available. "
        "Incomplete comparisons do not establish a selected improvement.\n"
    )


def phase(name, arguments, deadline):
    directory = study.SOURCE / "experiments/predictor_expanded_20261007"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    with (study.ROOT / f"{name}.log").open("a") as stream:
        proc = subprocess.Popen(
            [
                str(study.HIST / "eval-venv/bin/python"),
                str(directory / arguments[0]),
                *arguments[1:],
            ],
            stdout=stream,
            stderr=stream,
            cwd=study.SOURCE,
            env=env,
            start_new_session=True,
        )
        log("phase_start", phase=name, pid=proc.pid, arguments=arguments)
        while proc.poll() is None:
            if time.time() >= deadline:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=15)
                log("budget_stop", phase=name)
                unfinished(name, deadline)
                return False
            completed = len(list((study.ROOT / "folds").glob("*/completed.json")))
            study.arch.dump(
                study.ROOT / "status.json",
                dict(
                    phase=name,
                    pid=proc.pid,
                    completed_architecture_tasks=completed,
                    deadline=deadline,
                    remaining_seconds=max(0, deadline - time.time()),
                ),
            )
            time.sleep(30)
    log("phase_end", phase=name, returncode=proc.returncode)
    if proc.returncode:
        raise RuntimeError(f"CPU phase failed: {name}; inspect {name}.log")
    return True


def main():
    study.ROOT.mkdir(parents=True, exist_ok=True)
    lock = (study.ROOT / "follow.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    budget_path = study.ROOT / "budget.json"
    if not budget_path.exists():
        study.arch.dump(
            budget_path,
            dict(
                started=time.time(),
                deadline=time.time() + 9.5 * 3600,
                gpu_allocation_extension=False,
                cpu_workers=4,
            ),
        )
    budget = json.loads(budget_path.read_text())
    deadline = budget["deadline"]
    log("waiting_for_coverage_completion", deadline=deadline)
    while time.time() < deadline:
        events = [
            json.loads(line)
            for line in (study.COVERAGE / "process_log.jsonl").read_text().splitlines()
        ]
        if any(e["event"] == "study_complete" for e in events):
            break
        study.arch.dump(
            study.ROOT / "status.json",
            dict(
                phase="waiting_for_current_plan",
                deadline=deadline,
                remaining_seconds=deadline - time.time(),
            ),
        )
        time.sleep(30)
    else:
        log("budget_stop", phase="waiting_for_current_plan")
        unfinished("waiting_for_current_plan", deadline)
        return
    commands = [
        ("architecture", ["study.py", "--workers", "4"]),
        ("architecture_report", ["report.py"]),
        ("improvements", ["improve.py", "--workers", "4"]),
        ("continuous_trend", ["trend.py", "--workers", "4"]),
        ("benchmark", ["benchmark.py", "--workers", "4"]),
        ("branches", ["branches.py"]),
        ("final_report", ["final_report.py"]),
    ]
    for name, arguments in commands:
        if not phase(name, arguments, deadline):
            return
    log("exploration_complete")
    study.arch.dump(
        study.ROOT / "status.json",
        dict(
            phase="exploration_complete",
            deadline=deadline,
            remaining_seconds=deadline - time.time(),
        ),
    )


if __name__ == "__main__":
    main()
