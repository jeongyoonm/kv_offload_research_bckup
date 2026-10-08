# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read process CPU counters only at measurement boundaries."""

import json
import os
import subprocess
import time
from pathlib import Path


def snapshot(root):
    session = json.loads((root / "server/process.json").read_text())["pid"]
    processes = {}
    inventory = subprocess.check_output(["ps", "-eo", "pid,sid,comm"], text=True)
    for line in inventory.splitlines()[1:]:
        pid, sid, name = line.split(maxsplit=2)
        if int(sid) != session:
            continue
        path = Path("/proc") / pid
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            threads = {}
            for task in (path / "task").iterdir():
                try:
                    status = (task / "status").read_text().splitlines()
                    switches = sum(
                        int(row.split(":")[1])
                        for row in status
                        if row.startswith(
                            ("voluntary_ctxt_switches:", "nonvoluntary_ctxt_switches:")
                        )
                    )
                    threads[task.name] = switches
                except FileNotFoundError:
                    pass
            processes[pid] = dict(
                name=name,
                start_ticks=int(fields[19]),
                cpu_ticks=int(fields[11]) + int(fields[12]),
                threads=threads,
            )
        except FileNotFoundError:
            pass
    return dict(time_ns=time.monotonic_ns(), processes=processes)


def difference(before, after):
    tick_ms = 1000 / os.sysconf("SC_CLK_TCK")
    rows = []
    for pid, end in after["processes"].items():
        begin = before["processes"].get(pid)
        if begin is None or begin["start_ticks"] != end["start_ticks"]:
            continue
        common = begin["threads"].keys() & end["threads"].keys()
        rows.append(
            dict(
                pid=int(pid),
                name=end["name"],
                cpu_ms=(end["cpu_ticks"] - begin["cpu_ticks"]) * tick_ms,
                context_switches=sum(
                    end["threads"][tid] - begin["threads"][tid] for tid in common
                ),
                exited_threads=len(begin["threads"].keys() - end["threads"].keys()),
                new_threads=len(end["threads"].keys() - begin["threads"].keys()),
            )
        )
    engine = [row for row in rows if "EngineCor" in row["name"]]
    if len(engine) != 1:
        raise RuntimeError(f"Expected one EngineCore, found {engine}")
    return dict(
        interval_ms=(after["time_ns"] - before["time_ns"]) / 1e6,
        engine_cpu_ms=engine[0]["cpu_ms"],
        engine_context_switches=engine[0]["context_switches"],
        processes=rows,
        before=before,
        after=after,
        context_switch_scope="Threads present at both boundaries; transient threads "
        "are excluded. Context switches are not identical to scheduler wakeups.",
    )
