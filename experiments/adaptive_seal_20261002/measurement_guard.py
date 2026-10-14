"""Capture compilation and GPU process ownership during measured replays."""

import contextlib
import json
import subprocess

import psutil


def inventory():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )


@contextlib.contextmanager
def measure_guard(directory, server_dir):
    log_path = server_dir / "server.log"
    offset = log_path.stat().st_size
    pid = json.loads((server_dir / "process.json").read_text())["pid"]
    owned = {pid} | {p.pid for p in psutil.Process(pid).children(recursive=True)}
    (directory / "gpu_processes_before.csv").write_text(inventory())
    state = {}
    with (directory / "gpu_process_telemetry.csv").open("w") as telemetry:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
                "-l",
                "1",
            ],
            stdout=telemetry,
            stderr=telemetry,
        )
        try:
            yield state
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    (directory / "gpu_processes_after.csv").write_text(inventory())
    with log_path.open() as stream:
        stream.seek(offset)
        runtime = stream.read()
    (directory / "measured_server.log").write_text(runtime)
    state["measurement_jit_lines"] = [
        line
        for line in runtime.splitlines()
        if any(
            term in line
            for term in (
                "JIT compilation during inference",
                "Compiling ",
                "Downloading ",
            )
        )
    ]
    state["runtime_error_lines"] = [
        line
        for line in runtime.splitlines()
        if " ERROR " in line or "Traceback (most recent call last)" in line
    ]
    observed = set()
    for name in (
        "gpu_processes_before.csv",
        "gpu_processes_after.csv",
        "gpu_process_telemetry.csv",
    ):
        for line in (directory / name).read_text().splitlines():
            value = line.split(",")[0].strip()
            if value.isdigit():
                observed.add(int(value))
    state["unexpected_gpu_pids"] = sorted(observed - owned)
    (directory / "measurement_guard.json").write_text(
        json.dumps(state, indent=2) + "\n"
    )
