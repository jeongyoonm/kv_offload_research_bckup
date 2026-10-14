# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the normal-observer audit-retention fix without rebuilding model kernels."""

import hashlib
import subprocess

import run


def main():
    destination = run.ROOT / "libseal_execution_observer.so"
    temporary = destination.with_suffix(".so.new")
    command = [
        "g++",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O2",
        "-pthread",
        "-I",
        "csrc/libtorch_stable",
        "-I",
        "/usr/local/cuda-13.0/include",
        "experiments/coalesced_notify_20261004/observer.cpp",
        "-L",
        "/usr/local/cuda-13.0/lib64",
        "-lcudart",
        "-Wl,-rpath,/usr/local/cuda-13.0/lib64",
        "-o",
        str(temporary),
    ]
    subprocess.run(command, cwd=run.SOURCE, check=True)
    # Existing workers retain their mapped inode; future opens get the new build.
    temporary.replace(destination)
    run.save(
        run.ROOT / "observer_build.json",
        dict(
            command=command,
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        ),
    )


if __name__ == "__main__":
    main()
