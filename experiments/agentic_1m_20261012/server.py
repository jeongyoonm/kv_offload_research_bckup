# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned TP8 server; context differs from the previous trace campaign only."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from experiments.agentic_1m_20261012.common import (
    BASE,
    DEADLINE,
    LIMIT,
    PORT,
    SOURCE,
    log,
    save,
)
from experiments.load_only_overlap_20261011 import evaluate as old


def launch(root, arm):
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    assert not occupied, occupied
    command, config = old.base.server_command()
    command[0] = str(SOURCE / ".venv/bin/python")
    command[command.index("--max-model-len") + 1] = str(LIMIT)
    command[command.index("--port") + 1] = str(PORT)
    extra = config["kv_connector_extra_config"]
    if arm == "baseline":
        extra.update(
            async_h2d_submission=False,
            async_d2h_submission=False,
            cohort_status_diagnostics=True,
        )
        for name in list(extra):
            if name.startswith(("early_load_", "_early_load_")) or name in (
                "schedule_delay_ms",
                "coalesced_seal_notifications",
            ):
                del extra[name]
    command[command.index("--kv-transfer-config") + 1] = json.dumps(config)
    inherited = json.loads(
        Path("/mnt/localnvme/jeongyoon/lo11c/server/environment.json").read_text()
    )
    env = dict(os.environ, **inherited)
    env.update(
        PYTHONPATH=str(SOURCE),
        TMPDIR=str(root / "tmp"),
        VLLM_DEFERRAL_TIMING_DIR=str(root / "server/timing"),
        VLLM_DATASET_TARGET_GATE=str(root / "target_hold"),
        VLLM_DEFERRAL_NATIVE_OBSERVER=str(root / "libseal_execution_observer.so"),
        VLLM_KV_READINESS_TIMELINE_DIR=str(root / "server/load_timeline"),
        VLLM_DEFERRAL_LOAD_AUDIT="1",
        VLLM_LATE_ADMISSION_RECORD_CAPACITY="262144",
    )
    if arm == "baseline":
        for key in list(env):
            if key.startswith(("VLLM_DEFERRAL_", "VLLM_PREDICTOR_", "VLLM_DATASET_")):
                del env[key]
        env.update(VLLM_COOPERATIVE_RESPONSE_PROGRESS="0", VLLM_DEFERRAL_LOAD_AUDIT="1")
    env["PATH"] = (
        str(SOURCE / ".venv/bin") + ":/usr/local/cuda-13.0/bin:" + env.get("PATH", "")
    )
    for directory in (root / "server/load_timeline", root / "tmp"):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        "/mnt/localnvme/jeongyoon/glm53_dataset_coverage_20261007/libseal_execution_observer.so",
        root / "libseal_execution_observer.so",
    )
    save(root / "server/command.json", command)
    save(root / "server/config.json", config)
    save(
        root / "server/environment.json",
        {
            k: v
            for k, v in env.items()
            if k.startswith(("VLLM_", "HF_", "CUDA_", "PYTHON", "FLASHINFER_"))
        },
    )
    with (root / "server/server.log").open("w") as output:
        return subprocess.Popen(
            command,
            cwd=SOURCE,
            env=env,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=["baseline", "full"], required=True)
    args = parser.parse_args()
    root = args.root
    assert len(str(root / "tmp")) <= 45
    root.mkdir(parents=True, exist_ok=True)
    proc = launch(root, args.arm)
    run = old.base.collector
    run.BASE = BASE
    save(
        root / "owner.json",
        dict(
            supervisor_pid=os.getpid(),
            server_pid=proc.pid,
            deadline=DEADLINE,
            arm=args.arm,
        ),
    )
    ready = False
    try:
        while time.time() < DEADLINE and not (root / "stop.request").exists():
            if proc.poll() is not None:
                raise RuntimeError(f"server exited: {proc.returncode}")
            if not ready:
                try:
                    run.call("/health", timeout=1)
                    save(root / "ready.json", dict(time=time.time()))
                    log("server_ready", root=str(root), arm=args.arm)
                    ready = True
                except (OSError, TimeoutError):
                    pass
            time.sleep(2)
    finally:
        run.stop_server(proc)
        save(root / "stopped.json", dict(time=time.time(), returncode=proc.poll()))
        log("server_stopped", root=str(root), arm=args.arm)


if __name__ == "__main__":
    main()
