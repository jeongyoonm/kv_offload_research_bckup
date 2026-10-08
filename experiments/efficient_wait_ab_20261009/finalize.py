# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit a completed allocation and verify owned GPU resources are released."""

import json
import os
import subprocess
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_efficient_wait_ab_20261009")
PYTHON = Path("/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python")


def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2) + "\n")


def main():
    while True:
        try:
            allocation = json.loads((ROOT / "allocation.json").read_text())
            summary = json.loads((ROOT / "summary.json").read_text())
            break
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(10)
    if not summary["complete"] or not allocation["server_stopped"]:
        raise RuntimeError("Serving allocation incomplete; preserve logs for diagnosis")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for name in ["analyze", "plot"]:
        with (ROOT / f"{name}.log").open("w") as output:
            subprocess.run(
                [
                    str(PYTHON),
                    str(
                        SOURCE / "experiments/efficient_wait_ab_20261009" / f"{name}.py"
                    ),
                ],
                cwd=SOURCE,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
    save("analysis_complete.json", dict(passed=True, time=time.time()))
    ownership = json.loads((ROOT / "ownership.json").read_text())
    owned_pids = {p["pid"] for p in ownership["processes"]}
    samples = []
    for _ in range(30):
        process_rows = subprocess.check_output(
            ["ps", "-eo", "pid,sid,stat,comm"], text=True
        ).splitlines()[1:]
        remaining = [
            row
            for row in process_rows
            if int(row.split()[1]) == ownership["session_id"]
        ]
        sample = dict(time=time.time(), owned_session_processes=remaining)
        live = [row for row in remaining if not row.split()[2].startswith("Z")]
        try:
            sample["gpu_memory"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=30,
            )
            sample["compute_apps"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=30,
            )
            apps = {int(p) for p in sample["compute_apps"].splitlines() if p.strip()}
            all_idle = all(
                int(row.split(",")[1]) == 0 for row in sample["gpu_memory"].splitlines()
            )
            released = not live and not (apps & owned_pids)
            if released and (all_idle or apps):
                samples.append(sample)
                save(
                    "cleanup.json",
                    dict(
                        owned_released=True,
                        all_gpus_idle=all_idle,
                        external_compute_apps=sorted(apps),
                        samples=samples,
                    ),
                )
                save(
                    "SESSION_STATE.json",
                    dict(
                        status="complete",
                        cells=54,
                        source=str(SOURCE),
                        artifact_root=str(ROOT),
                        branch="experiment/glm53-efficient-wait-ablation-20261009",
                    ),
                )
                print("Analysis complete; owned server resources released", flush=True)
                return
        except subprocess.TimeoutExpired:
            sample["gpu_query_timeout"] = True
        samples.append(sample)
        save("cleanup.json", dict(owned_released=False, samples=samples))
        time.sleep(10)
    raise RuntimeError("Owned resource cleanup not verified; inspect cleanup.json")


if __name__ == "__main__":
    main()
