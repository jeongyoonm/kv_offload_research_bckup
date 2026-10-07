# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rerun rejected evaluation cells after fixing normal-observer audit overflow."""

import json
import threading
import time
import urllib.error

import run


def main():
    allocation = json.loads((run.ROOT / "allocation.json").read_text())
    used = allocation["total_seconds"]
    matrix = json.loads((run.ROOT / "matrix.json").read_text())
    started = time.monotonic()
    proc = watchdog = None
    completed = {}
    try:
        proc = run.launch()
        watchdog = threading.Timer(max(1, 14400 - used - 60), run.stop_server, (proc,))
        watchdog.daemon = True
        watchdog.start()
        run.log("server_started", pid=proc.pid, recovery=True)
        while time.monotonic() - started < 1500:
            if proc.poll() is not None:
                raise RuntimeError("recovery server exited before health")
            try:
                run.call("/health", timeout=2)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(2)
        else:
            raise RuntimeError("recovery startup exceeded allowance")
        state = json.loads((run.HIST / "boosted_residual/model.json").read_text())[
            "state"
        ]
        run.configure(state=state)
        run.log("server_ready", recovery=True, seconds=time.monotonic() - started)
        # Keep every original seed and family. Attempt a1 distinguishes the new
        # engine session from invalid a0 records; no invalid labels are rescued.
        for split, seeds in [
            ("development", [801, 802, 803]),
            ("final", [901, 902, 903]),
        ]:
            completed[split] = 0
            consecutive_rejections = 0
            for seed in seeds:
                for i, family in enumerate(matrix["challenge"] + matrix["train"][::8]):
                    if used + time.monotonic() - started >= 12700:
                        run.log("recovery_budget_stop", completed=completed)
                        break
                    result = run.cohort(family, seed, i, split, retries=1)
                    consecutive_rejections = (
                        0 if result["accepted"] else consecutive_rejections + 1
                    )
                    if consecutive_rejections >= 3:
                        raise RuntimeError("three consecutive recovery rejections")
                    completed[split] += int(result["accepted"])
                else:
                    continue
                break
        run.log("shadow_test_complete", recovery=True, completed=completed)
    except BaseException as error:
        run.log("failed", recovery=True, error=repr(error))
        raise
    finally:
        if watchdog:
            watchdog.cancel()
        if proc:
            run.stop_server(proc)
        run.save(
            run.ROOT / "allocation.json",
            dict(
                seconds=time.monotonic() - started,
                total_seconds=used + time.monotonic() - started,
                limit_seconds=14400,
                server_stopped=proc is None or proc.poll() is not None,
            ),
        )
        run.log("stopped", recovery=True, seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
