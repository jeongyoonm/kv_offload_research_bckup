# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-only regression for an ordinary observer exceeding its ring capacity."""

import ctypes
import json
import mmap
import os
import struct
import tempfile
import time

import run


def main():
    lib = ctypes.CDLL(str(run.ROOT / "libseal_execution_observer.so"))
    lib.timing_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
    lib.timing_create.restype = ctypes.c_void_p
    for name in ["timing_start", "timing_finish"]:
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        fn.restype = ctypes.c_int
    lib.timing_destroy.argtypes = [ctypes.c_void_p]
    record = struct.Struct("9Q")
    fd, path = tempfile.mkstemp(dir=run.ROOT / "tmp", prefix="observer-longrun-")
    os.ftruncate(fd, 4096 * record.size)
    mem = mmap.mmap(fd, 4096 * record.size)
    handle = lib.timing_create(path.encode(), 0, b"")
    assert handle
    started = time.monotonic()
    try:
        for step in range(1, 4201):
            assert lib.timing_start(handle, step, 0), ("start", step)
            assert lib.timing_finish(handle, step, 0), ("finish", step)
            deadline = time.monotonic() + 2
            while True:
                result = record.unpack_from(mem, (step % 4096) * record.size)
                if result[0] == step and result[8]:
                    assert result[5] == 1, ("failed_record", step, result)
                    break
                assert time.monotonic() < deadline, ("timeout", step, result)
                time.sleep(0.0005)
        print(
            json.dumps(
                dict(
                    completed=4200,
                    normal_mode=True,
                    all_completed_records_valid=True,
                    seconds=time.monotonic() - started,
                )
            ),
            flush=True,
        )
    finally:
        lib.timing_destroy(handle)
        mem.close()
        os.close(fd)
        os.unlink(path)


if __name__ == "__main__":
    main()
