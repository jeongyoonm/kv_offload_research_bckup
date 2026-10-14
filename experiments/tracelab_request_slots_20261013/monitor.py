# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sample GPU utilization independently of request and model execution paths."""

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_request_slots_20261013")


def main():
    with (ROOT / "gpu_samples.jsonl").open("a") as output:
        while not (
            (ROOT / "completed.json").exists() or (ROOT / "failure.json").exists()
        ):
            started = time.time()
            progress = ROOT / "progress.json"
            state = json.loads(progress.read_text()) if progress.exists() else {}
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,utilization.memory,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            rows = [
                [float(v.strip()) for v in line.split(",")] for line in raw.splitlines()
            ]
            output.write(
                json.dumps(
                    dict(
                        time=started,
                        arm=state.get("arm"),
                        phase=state.get("phase"),
                        gpus=rows,
                    )
                )
                + "\n"
            )
            output.flush()
            time.sleep(5)


if __name__ == "__main__":
    main()
