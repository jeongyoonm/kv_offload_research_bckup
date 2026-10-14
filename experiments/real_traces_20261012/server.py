# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned trace-comparison server with a hard campaign deadline."""

import argparse
import datetime
import os
import shutil
import time
from pathlib import Path

from experiments.load_only_overlap_20261011 import evaluate as old

DEADLINE = datetime.datetime(
    2026, 10, 12, 3, 40, 0, tzinfo=datetime.timezone.utc
).timestamp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=["baseline", "full"], required=True)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    assert len(str(root / "tmp")) <= 45
    run = old.base.collector
    old.base.module.ROOT = old.base.module.driver.ROOT = run.ROOT = root
    run.BASE = "http://127.0.0.1:18472"
    os.environ["VLLM_LATE_ADMISSION_RECORD_CAPACITY"] = "262144"
    shutil.copy2(
        "/mnt/localnvme/jeongyoon/glm53_dataset_coverage_20261007/libseal_execution_observer.so",
        root / "libseal_execution_observer.so",
    )
    proc = old.launch(
        root, "baseline" if args.arm == "baseline" else "optimized", 18472
    )
    run.save(
        root / "owner.json",
        dict(
            supervisor_pid=os.getpid(),
            server_pid=proc.pid,
            deadline=DEADLINE,
            arm=args.arm,
        ),
    )
    try:
        ready = False
        while time.time() < DEADLINE and not (root / "stop.request").exists():
            if proc.poll() is not None:
                raise RuntimeError(f"server exited: {proc.returncode}")
            if not ready:
                try:
                    run.call("/health", timeout=1)
                    ready = True
                    run.save(root / "ready.json", dict(time=time.time()))
                    print("SERVER READY", flush=True)
                except (OSError, TimeoutError):
                    pass
            time.sleep(2)
    finally:
        run.stop_server(proc)
        run.save(root / "stopped.json", dict(time=time.time(), returncode=proc.poll()))


if __name__ == "__main__":
    main()
