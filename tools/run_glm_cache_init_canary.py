# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch an isolated TP8 GLM cache-startup and CPU-restore canary."""

import argparse
import ctypes
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import regex as re

MODEL = (
    "/mnt/localnvme/huggingface/hub/models--Inferact--GLM-5.3-NVFP4/"
    "snapshots/ce67b36f3669192b5bb233819f0fda6c8a9837f8"
)


def _call(base, path, payload=None, timeout=600):
    request = urllib.request.Request(
        base + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def _counters(raw):
    counters = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.split()[:2]
        name = name.split("{")[0]
        if name in (
            "vllm:prefix_cache_hits_total",
            "vllm:external_prefix_cache_hits_total",
        ):
            counters[name] = counters.get(name, 0) + float(value)
    return counters


def _reset(base, external):
    for _ in range(30):
        result = json.loads(
            _call(
                base, f"/reset_prefix_cache?reset_external={str(external).lower()}", {}
            )
        )
        if result["success"]:
            return
        time.sleep(2)
    raise RuntimeError("cache reset failed to drain transfers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18387)
    parser.add_argument("--flashinfer-workspace", type=Path)
    parser.add_argument(
        "--layout", choices=("block", "worker", "private"), default="private"
    )
    parser.add_argument("--cpu-bytes", type=int, default=2130300354560)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as port_check:
        port_check.bind(("127.0.0.1", args.port))
    existing = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).strip()
    if existing:
        raise RuntimeError(f"GPUs are occupied: {existing}")
    cache = args.output / "vllm_cache"
    shutil.copytree("/mnt/localnvme/glm53-compat-cache", cache, dirs_exist_ok=True)
    flashinfer_base = args.flashinfer_workspace or args.output / "flashinfer_workspace"
    flashinfer_cache = Path.home() / ".cache" / "flashinfer"
    if flashinfer_cache.exists() and not flashinfer_base.exists():
        shutil.copytree(
            flashinfer_cache,
            flashinfer_base / ".cache" / "flashinfer",
            dirs_exist_ok=True,
        )
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(source),
        HF_HOME="/mnt/localnvme/huggingface",
        VLLM_CACHE_ROOT=str(cache),
        PYTHONHASHSEED="0",
        VLLM_SERVER_DEV_MODE="1",
        CUDA_HOME="/usr/local/cuda-13.0",
        PATH=f"{Path(sys.executable).parent}:/usr/local/cuda-13.0/bin:"
        + os.environ.get("PATH", ""),
        FLASHINFER_WORKSPACE_BASE=str(flashinfer_base),
        VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS="0",
        TORCHINDUCTOR_CACHE_DIR=str(args.output / "inductor_cache"),
        TRITON_CACHE_DIR=str(args.output / "triton_cache"),
    )
    config = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use": args.cpu_bytes,
            "cpu_cache_layout": args.layout,
            "async_h2d_submission": True,
            "async_h2d_chunk_bytes": 134217728,
        },
    }
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--exec-server",
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        MODEL,
        "--served-model-name",
        "glm53-cache-init",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        "8",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-model-len",
        "65536",
        "--max-num-batched-tokens",
        "61440",
        "--max-num-seqs",
        "64",
        "--kv-cache-memory-bytes",
        "103079215104",
        "--gpu-memory-utilization",
        "0.80",
        "--enable-prompt-tokens-details",
        "--block-size",
        "64",
        "--no-enable-flashinfer-autotune",
        "--async-scheduling",
        "--kv-transfer-config",
        json.dumps(config),
        "--numa-bind",
        "--numa-bind-nodes",
        "0",
        "0",
        "0",
        "0",
        "1",
        "1",
        "1",
        "1",
        "--numa-bind-cpus",
    ]
    command.extend(
        f"{r * 16}-{r * 16 + 15},{128 + r * 16}-{143 + r * 16}" for r in range(8)
    )
    (args.output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (args.output / "connector_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    base = f"http://127.0.0.1:{args.port}"
    started = time.monotonic()
    proc = None
    summary = {}
    try:
        with (args.output / "server.log").open("w") as log:
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
                cwd=source,
            )
            (args.output / "process.json").write_text(
                json.dumps({"pid": proc.pid}) + "\n"
            )
            while time.monotonic() - started < 2400:
                if proc.poll() is not None:
                    raise RuntimeError("server exited before becoming healthy")
                try:
                    _call(base, "/health", timeout=2)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    time.sleep(2)
            else:
                raise TimeoutError("server startup exceeded 40 minutes")
            summary["server_ready_seconds"] = time.monotonic() - started
            print(json.dumps(summary), flush=True)
            probes = []
            for index in range(4):
                _reset(base, True)
                tokens = [1000 + (index * 131 + i * 17) % 10000 for i in range(64)]
                tokens += [1000 + (i * 23) % 10000 for i in range(4096)]
                payload = {
                    "model": "glm53-cache-init",
                    "prompt": tokens,
                    "temperature": 0,
                    "max_tokens": 2,
                    "ignore_eos": True,
                }
                cold = json.loads(_call(base, "/v1/completions", payload))
                _reset(base, False)
                before = _counters(_call(base, "/metrics"))
                restored = json.loads(_call(base, "/v1/completions", payload))
                raw = _call(base, "/metrics")
                (args.output / f"probe_{index}.metrics").write_text(raw)
                after = _counters(raw)
                delta = {key: after.get(key, 0) - before.get(key, 0) for key in after}
                probe = {
                    "index": index,
                    "cold": cold,
                    "restored": restored,
                    "counter_delta": delta,
                    "text_equal": cold["choices"][0]["text"]
                    == restored["choices"][0]["text"],
                }
                probes.append(probe)
                (args.output / "probes.json").write_text(
                    json.dumps(probes, indent=2) + "\n"
                )
                assert delta.get("vllm:external_prefix_cache_hits_total", 0) > 0
                assert delta.get("vllm:prefix_cache_hits_total", 0) == 0
                assert probe["text_equal"], "cold/restore text mismatch"
                print(
                    json.dumps({"probe": index, "passed": True, "delta": delta}),
                    flush=True,
                )
            summary["probes_passed"] = len(probes)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        log_path = args.output / "server.log"
        raw_log = log_path.read_text() if log_path.exists() else ""
        summary.update(
            population_seconds=[
                float(v)
                for v in re.findall(
                    r"CPU cache population .*seconds=([0-9.]+)", raw_log
                )
            ],
            registration_seconds=[
                float(v)
                for v in re.findall(
                    r"CPU cache registration rank=.*seconds=([0-9.]+)", raw_log
                )
            ],
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=180)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--exec-server"]:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(41, 1, 0, 0, 0):
            raise OSError(ctypes.get_errno(), "PR_SET_THP_DISABLE failed")
        os.execvp(sys.argv[2], sys.argv[2:])
    main()
