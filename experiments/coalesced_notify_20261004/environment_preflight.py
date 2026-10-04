# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject incomplete vendor installs and wait for GPU memory cleanup."""

import json
import os
import subprocess
import time


def check(source, directory, uuids):
    required = (
        "vllm/third_party/flashmla/flash_mla_interface.py",
        "vllm/third_party/deep_gemm/__init__.py",
    )
    for path in required:
        if not (source / path).is_file():
            raise RuntimeError(f"Missing vendored dependency: {source / path}")
    samples = []
    deadline = time.monotonic() + 600
    while True:
        inventory = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        used = {
            uuid.strip(): int(memory.strip())
            for uuid, memory in (line.split(",") for line in inventory.splitlines())
        }
        samples.append(dict(time=time.time(), used_mib=used))
        (directory / "environment-preflight.json").write_text(
            json.dumps(dict(samples=samples), indent=2) + "\n"
        )
        if all(used[uuid] < 1024 for uuid in uuids):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("GPU memory cleanup did not finish within 600 seconds")
        print("Waiting for GPU memory cleanup", used, flush=True)
        time.sleep(2)
    # A separate process avoids retaining CUDA state in the benchmark driver.
    env = dict(os.environ, PYTHONPATH=str(source))
    env.setdefault("VLLM_CACHE_ROOT", "/mnt/localnvme/glm53-compat-cache")
    env.setdefault("CUDA_HOME", "/usr/local/cuda-13.0")
    result = subprocess.run(
        [
            str(source / ".venv/bin/python"),
            "-c",
            "import vllm.third_party.flashmla.flash_mla_interface; "
            "from vllm.utils.deep_gemm import _import_deep_gemm; "
            "m = _import_deep_gemm(); "
            "assert callable(m.fp8_gemm_nt); "
            "assert callable(m.fp8_fp4_mqa_logits); "
            "print(m.__file__)",
        ],
        cwd=source,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (directory / "dependency-preflight.json").write_text(
        json.dumps(
            dict(
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
            ),
            indent=2,
        )
        + "\n"
    )
    result.check_returncode()
