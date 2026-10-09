# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run byte checks and disjoint submission probes after the serving queue drains."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

BASE = Path("/mnt/localnvme/jeongyoon")
ARTIFACTS = BASE / "store_overlap_overnight_20261009"


def log(event, **fields):
    record = dict(time=time.time(), event=event)
    record.update(fields)
    with (ARTIFACTS / "idle_probe_process_log.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def idle():
    for _ in range(36):
        occupied = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        if not occupied:
            return
        time.sleep(5)
    raise RuntimeError(f"GPUs remain occupied: {occupied}")


def run(label, source, command, *, env=None, expected_passes=None, optional=False):
    idle()
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True
    ).strip(), source
    environment = dict(os.environ, PYTHONPATH=str(source), **(env or {}))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    log(
        "probe_started", label=label, source=str(source), commit=commit, command=command
    )
    path = ARTIFACTS / f"{label}.log"
    with path.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=source,
            env=environment,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = -1
    if expected_passes is not None and code == 0:
        assert f"{expected_passes} passed" in path.read_text(), path
    log("probe_complete", label=label, code=code, optional=optional)
    if not optional:
        assert code == 0, path
    return code == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor", type=Path, required=True)
    args = parser.parse_args()
    summary = args.predecessor / "summary.json"
    while not summary.exists():
        time.sleep(20)
    data = json.loads(summary.read_text())
    assert data["complete"] and data["stopped"]
    log("predecessor_complete", root=str(args.predecessor))
    native = BASE / "vllm_store_native_async_20261009"
    coalescing = BASE / "vllm_kv_order_coalescing_20261009"
    probe = BASE / "vllm_store_contention_probe_20261009"
    packet = BASE / "vllm_kv_packet_coalescing_20261009"
    run(
        "native_async_real_gpu_tests",
        native,
        [
            str(native / ".venv/bin/python"),
            "-m",
            "pytest",
            "tests/v1/kv_offload/cpu/test_gpu_worker.py",
            "-q",
            "-k",
            "native_store_orders or native_store_fifo",
        ],
        expected_passes=3,
    )
    run(
        "bounded_load_real_gpu_tests",
        coalescing,
        [
            str(coalescing / ".venv/bin/python"),
            "-m",
            "pytest",
            "tests/v1/kv_offload/cpu/test_gpu_worker.py",
            "-q",
            "-k",
            "coalesced_async_load or load_coalescing_preserves",
        ],
        expected_passes=10,
    )
    run(
        "packet_alignment_real_gpu_tests",
        packet,
        [
            str(packet / ".venv/bin/python"),
            "-m",
            "pytest",
            "tests/v1/kv_offload/cpu/test_gpu_worker.py",
            "-q",
            "-k",
            "coalesced_async_load",
        ],
        expected_passes=6,
    )
    for cap in (0, 256, 512):
        command = [
            "numactl",
            "--physcpubind=0-15,128-143",
            "--membind=0",
            str(probe / ".venv/bin/python"),
            "benchmarks/kernels/measure_store_submission_contention.py",
            "--output",
            str(ARTIFACTS / f"contention_cap{cap}.json"),
            "--repeats",
            "12",
        ]
        run(
            f"contention_cap{cap}",
            probe,
            command,
            env={"VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS": str(cap)},
        )
    prefix = ARTIFACTS / "contention_nsys"
    profile = [
        "/usr/local/cuda/bin/nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "-o",
        str(prefix),
        "numactl",
        "--physcpubind=0-15,128-143",
        "--membind=0",
        str(probe / ".venv/bin/python"),
        "benchmarks/kernels/measure_store_submission_contention.py",
        "--output",
        str(ARTIFACTS / "contention_profiled.json"),
        "--repeats",
        "2",
    ]
    profiled = run(
        "contention_profile",
        probe,
        profile,
        env={"VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS": "0"},
        optional=True,
    )
    if profiled:
        run(
            "contention_sqlite_export",
            probe,
            [
                "/usr/local/cuda/bin/nsys",
                "export",
                "--type=sqlite",
                "-o",
                str(prefix) + ".sqlite",
                str(prefix) + ".nsys-rep",
            ],
            optional=True,
        )
    idle()
    result = dict(
        complete=True,
        stopped=True,
        profiled=profiled,
        predecessor=str(args.predecessor),
        time=time.time(),
        native_gpu_tests=3,
        bounded_coalescing_tests=10,
        packet_alignment_gpu_tests=6,
    )
    (ARTIFACTS / "idle_probes_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    gate = ARTIFACTS / "idle_gate"
    gate.mkdir(exist_ok=True)
    temporary = gate / "summary.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(gate / "summary.json")
    log("all_probes_complete", **result)


if __name__ == "__main__":
    main()
