# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serialize tensor-major GPU byte verification between serving allocations."""

import argparse
import json
import time
from pathlib import Path

from experiments.store_overlap_overnight_20261009 import idle_probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor", type=Path, required=True)
    args = parser.parse_args()
    summary = args.predecessor / "summary.json"
    while not summary.exists():
        time.sleep(20)
    previous = json.loads(summary.read_text())
    assert previous["complete"] and previous["stopped"]
    source = idle_probes.BASE / "vllm_cpu_tensor_layout_20261009"
    idle_probes.run(
        "tensor_layout_real_gpu_tests",
        source,
        [
            str(source / ".venv/bin/python"),
            "-m",
            "pytest",
            "tests/v1/kv_offload/cpu/test_gpu_worker.py",
            "-q",
            "-k",
            "private_tensor_glm_geometry",
        ],
        expected_passes=2,
    )
    idle_probes.idle()
    gate = idle_probes.ARTIFACTS / "tensor_gate"
    gate.mkdir(exist_ok=True)
    temporary = gate / "summary.tmp"
    temporary.write_text(
        json.dumps(dict(complete=True, stopped=True, tests=2, time=time.time())) + "\n"
    )
    temporary.replace(gate / "summary.json")
    idle_probes.log("tensor_gate_complete", tests=2)


if __name__ == "__main__":
    main()
