# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Size private RAM backing against available memory and balanced NUMA ranks."""

import argparse
from pathlib import Path

import regex as re

from experiments.agentic_1m_20261012.common import ROOT, log, save


def read_meminfo(path):
    return {
        match[1]: int(match[2]) * 1024
        for line in path.read_text().splitlines()
        if (match := re.search(r"(?:Node \d+ )?([A-Za-z_()]+):\s+(\d+) kB", line))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-numa-shmem-kib", type=int)
    args = parser.parse_args()
    host = read_meminfo(Path("/proc/meminfo"))
    nodes = [
        read_meminfo(Path(f"/sys/devices/system/node/node{n}/meminfo")) for n in (0, 1)
    ]
    # Reclaimable file pages exclude shmem; private CPU ranks divide equally.
    available = [
        n["MemFree"] + n["Active(file)"] + n["Inactive(file)"] + n["SReclaimable"]
        for n in nodes
    ]
    startup = 80 * 1024**3
    reserve = 128 * 1024**3
    upper = min(host["MemAvailable"], 2 * min(available)) - startup - reserve
    observed_bound = None
    if args.observed_numa_shmem_kib:
        observed_bound = 2 * args.observed_numa_shmem_kib * 1024 - reserve
        upper = min(upper, observed_bound)
    budget = upper // (64 * 1024**2) * (64 * 1024**2)
    assert budget > 2130300354560
    result = dict(
        cpu_bytes_to_use=budget,
        cpu_cache_backing="memfd",
        host=host,
        nodes=nodes,
        node_reclaimable_estimates=available,
        startup_allowance_bytes=startup,
        runtime_headroom_bytes=reserve,
        observed_numa_limit_bytes=observed_bound,
        method=(
            "Largest aligned budget within current available RAM and equal NUMA "
            "rank placement after80GiB observed-server allowance and64GiB/node "
            "runtime headroom. Full-capacity CUDA registration and restoration "
            "must pass before measurements."
        ),
    )
    save(ROOT / "cache_budget.json", result)
    log(
        "maximum_cache_budget_selected",
        bytes=budget,
        TiB=budget / 1024**4,
        reserve_bytes=reserve,
        startup_bytes=startup,
    )


if __name__ == "__main__":
    main()
