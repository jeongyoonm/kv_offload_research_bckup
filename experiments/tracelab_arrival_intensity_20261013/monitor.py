# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record serving demand, GPU activity, and cache counters throughout each cell."""

import json
import subprocess
import time
import urllib.error
import urllib.request

from experiments.tracelab_arrival_intensity_20261013.prepare import ROOT


def main():
    with (
        (ROOT / "gpu_samples.jsonl").open("a") as gpu_output,
        (ROOT / "cache_samples.jsonl").open("a") as cache_output,
    ):
        iteration = 0
        while not any(
            (ROOT / name).exists() for name in ("completed.json", "failure.json")
        ):
            active_path = ROOT / "active.json"
            active = json.loads(active_path.read_text()) if active_path.exists() else {}
            progress_path = active.get("progress_path")
            if progress_path:
                from pathlib import Path

                path = Path(progress_path)
                if path.exists():
                    active.update(json.loads(path.read_text()))
            active["time"] = time.time()
            temporary = ROOT / "progress.tmp"
            temporary.write_text(json.dumps(active, indent=2) + "\n")
            temporary.replace(ROOT / "progress.json")
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,utilization.memory,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            gpu_output.write(
                json.dumps(
                    dict(
                        time=time.time(),
                        episode=active.get("episode"),
                        arm=active.get("arm"),
                        phase=active.get("phase"),
                        gpus=[
                            [float(value.strip()) for value in line.split(",")]
                            for line in raw.splitlines()
                        ],
                    )
                )
                + "\n"
            )
            gpu_output.flush()
            if iteration % 2 == 0 and active.get("phase") != "server_startup":
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:18484/metrics", timeout=3
                    ) as response:
                        text = response.read().decode()
                    values = {}
                    for line in text.splitlines():
                        if not line.startswith(
                            (
                                "vllm:prefix_cache_",
                                "vllm:external_prefix_cache_",
                                "vllm:prompt_tokens_",
                                "vllm:kv_offload_load_bytes_total",
                                "vllm:kv_offload_store_bytes_total",
                                "vllm:num_requests_",
                                "vllm:num_preemptions_total",
                            )
                        ):
                            continue
                        key, value = line.split()[:2]
                        values[key] = float(value)
                    cache_output.write(
                        json.dumps(
                            dict(
                                time=time.time(),
                                episode=active.get("episode"),
                                values=values,
                            )
                        )
                        + "\n"
                    )
                    cache_output.flush()
                except (OSError, TimeoutError, urllib.error.URLError):
                    pass
            iteration += 1
            time.sleep(5)


if __name__ == "__main__":
    main()
