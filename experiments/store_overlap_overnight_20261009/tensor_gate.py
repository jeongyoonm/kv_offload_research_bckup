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
    independent = idle_probes.BASE / "vllm_load_independent_ready_20261009"
    idle_probes.run(
        "independent_ready_real_gpu_tests",
        independent,
        [
            str(independent / ".venv/bin/python"),
            "-m",
            "pytest",
            "tests/v1/kv_offload/cpu/test_gpu_worker.py",
            "-q",
            "-k",
            "independent_load_ready_bypasses",
        ],
        expected_passes=2,
    )
    idle_probes.idle()
    descriptor = idle_probes.BASE / "vllm_descriptor_order_20261009"
    idle_probes.run(
        "descriptor_order_real_gpu_tests",
        descriptor,
        [
            str(descriptor / ".venv/bin/python"),
            "-m",
            "pytest",
            "tests/v1/kv_offload/cpu/test_gpu_worker.py",
            "-q",
            "-k",
            "descriptor_preserve_order_glm_geometry",
        ],
        expected_passes=2,
    )
    idle_probes.run(
        "actual_cpu_layout_load_matrix",
        descriptor,
        [
            "numactl",
            "--physcpubind=0-15,128-143",
            "--membind=0",
            str(descriptor / ".venv/bin/python"),
            "benchmarks/kernels/measure_cpu_layout_loads.py",
            "--output",
            str(idle_probes.ARTIFACTS / "cpu_layout_loads.json"),
            "--repeats",
            "6",
        ],
        env={"VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS": "0"},
    )
    matrix = json.loads((idle_probes.ARTIFACTS / "cpu_layout_loads.json").read_text())
    assert matrix["complete"] and len(matrix["records"]) == 12
    idle_probes.idle()
    gate = idle_probes.ARTIFACTS / "tensor_gate"
    gate.mkdir(exist_ok=True)
    temporary = gate / "summary.tmp"
    temporary.write_text(
        json.dumps(
            dict(
                complete=True,
                stopped=True,
                tests=6,
                load_matrix_cases=12,
                time=time.time(),
            )
        )
        + "\n"
    )
    temporary.replace(gate / "summary.json")
    idle_probes.log("tensor_gate_complete", tests=6, load_matrix_cases=12)


if __name__ == "__main__":
    main()
