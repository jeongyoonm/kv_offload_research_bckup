# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resume valid cells after a server failure without changing serving policies."""

import argparse
import json
import os
import signal
import subprocess
import time

from experiments.tracelab_capacity_sweep_20261013 import campaign as study


def archive_incomplete():
    root = study.ROOT
    destination = root / "rejected" / str(int(time.time()))
    destination.mkdir(parents=True)
    accepted = {result["episode"] for result in study.analyze.results()}
    rejected = []
    for episode in (root / "episodes").iterdir():
        if episode.name in accepted:
            continue
        rejected.append(episode.name)
        episode.rename(destination / episode.name)
        result = root / "results" / f"{episode.name}.json"
        if result.exists():
            result.rename(destination / result.name)
    failure = root / "failure.json"
    if failure.exists():
        failure.rename(destination / "failure.json")
    if not rejected and not list(destination.iterdir()):
        destination.rmdir()
        return rejected
    study.base.save(
        destination / "exclusion.json",
        dict(episodes=rejected, reason="Incomplete replay; excluded from performance"),
    )
    return rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-low-rate-32", action="store_true")
    args = parser.parse_args()
    root, base = study.ROOT, study.base
    if args.retry_low_rate_32:
        completed = root / "completed.json"
        assert completed.exists(), "Previous controller must finish first"
        assert study.DEADLINE - time.time() >= 3300
        completed.rename(root / f"stage_completed_{int(time.time())}.json")
        base.save(
            root / "progress.json", dict(complete=False, phase="additional_retry")
        )
    base.ROOT, base.DEADLINE = root, study.DEADLINE
    replay = base.replay
    replay.ROOT, replay.SOURCE = root, base.SOURCE
    replay.BINARY, replay.RUNNER = study.BINARY, study.RUNNER
    replay.setup()
    rejected = archive_incomplete()
    previous_owner = json.loads((root / "owner.json").read_text())
    owner_path = root / f"owner_history_{int(time.time())}.json"
    base.save(owner_path, previous_owner)
    base.save(
        root / "owner.json",
        dict(
            pid=os.getpid(),
            time=time.time(),
            deadline=study.DEADLINE,
            source=str(base.SOURCE),
            resumed_from=previous_owner,
        ),
    )
    signal.signal(signal.SIGALRM, study.timeout_handler)
    signal.alarm(max(1, int(study.DEADLINE - time.time())))
    monitor_log = (root / "monitor.log").open("a")
    monitor = subprocess.Popen(
        [
            str(base.SOURCE / ".venv/bin/python"),
            "-m",
            "experiments.tracelab_capacity_sweep_20261013.monitor",
        ],
        cwd=base.SOURCE,
        stdout=monitor_log,
        stderr=subprocess.STDOUT,
    )
    failures = []
    remaining = {
        "baseline": [
            ("0.15", 40),
            ("0.15", 48),
            ("0.15", 32),
            ("0.25", 40),
            ("0.25", 48),
        ],
        "full": [("0.15", 32), ("0.25", 40), ("0.25", 48), ("0.15", 48)],
    }
    if args.retry_low_rate_32:
        remaining = {"full": [("0.15", 32)]}
    study.decision(
        "resume_after_connector_assertion",
        rejected=rejected,
        rationale="Preserve failed evidence and fixed policies; finish controls, "
        "then remaining full cells including one unchanged retry. "
        "Original ten-hour deadline remains in effect.",
        cases=remaining,
        monitor_pid=monitor.pid,
    )
    try:
        for arm, cases in remaining.items():
            attempt = 0
            while cases:
                accepted = {
                    (
                        r["arrival_rate_sessions_per_second"],
                        r["measurement_request_slots"],
                        r["arm"],
                    )
                    for r in study.analyze.results()
                }
                cases = [
                    (rate, limit)
                    for rate, limit in cases
                    if (float(rate), limit, arm) not in accepted
                ]
                if not cases:
                    break
                if study.DEADLINE - time.time() < 2200 + study.RESERVE:
                    study.decision(
                        "remaining_cases_skipped",
                        arm=arm,
                        cases=cases,
                        rationale="Retain full windows and shutdown reserve",
                    )
                    break
                label = ("b" if arm == "baseline" else "f") + f"r{attempt}"
                if args.retry_low_rate_32:
                    label = "fr3"
                try:
                    study.run_group(label, arm, cases)
                    break
                except Exception as error:
                    rejected = archive_incomplete()
                    failures.append(dict(arm=arm, episodes=rejected, error=repr(error)))
                    study.decision("resumed_group_failed", **failures[-1])
                    if args.retry_low_rate_32:
                        break
                    # Continue other cases; do not retry a second failed measurement.
                    cases = [
                        (rate, limit)
                        for rate, limit in cases
                        if f"r{rate}_c{limit}_rep0_{arm}" not in rejected
                    ]
                    attempt += 1
        base.save(
            root / "completed.json",
            dict(
                time=time.time(),
                cells=len(study.analyze.results()),
                groups_completed=True,
                resumed=True,
                failures=failures,
            ),
        )
        base.save(root / "progress.json", dict(complete=True, time=time.time()))
        study.decision(
            "campaign_complete",
            cells=len(study.analyze.results()),
            resumed=True,
            failures=failures,
        )
    except BaseException as error:
        base.save(root / "failure.json", dict(time=time.time(), error=repr(error)))
        raise
    finally:
        signal.alarm(0)
        monitor.wait(timeout=20)
        study.analyze.write_report()


if __name__ == "__main__":
    main()
