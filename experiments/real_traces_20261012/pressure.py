# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched, bounded causal-window cache-pressure comparison."""

import datetime
import json
import subprocess
import time
from pathlib import Path

from experiments.real_traces_20261012 import replay

CUTOFF = datetime.datetime(
    2026, 10, 12, 3, 34, tzinfo=datetime.timezone.utc
).timestamp()


def main():
    campaign = replay.CAMPAIGN
    baseline = Path("/mnt/localnvme/jeongyoon/tr12b2")
    full = Path("/mnt/localnvme/jeongyoon/tr12f2")
    while not (campaign / "extras_complete.json").exists():
        if (baseline / "stopped.json").exists():
            raise RuntimeError("baseline stopped")
        time.sleep(5)
    ref = json.loads(
        Path(
            "/mnt/localnvme/jeongyoon/tr12b1/cells/inferact_s64_n32_r0/result.json"
        ).read_text()
    )["seconds"]
    spec = dict(
        id="inferact_s128_n32_t16_r0",
        dataset="inferact",
        sessions=128,
        concurrency=32,
        turns=16,
    )
    reserve = ref * 1.9 * (1 + 1.15) + 660
    decision = dict(
        time=time.time(),
        spec=spec,
        required_seconds=reserve,
        remaining_seconds=CUTOFF - time.time(),
        selected=time.time() + reserve < CUTOFF,
        reason=(
            "1803 calls, 1.103M output tokens and 7.615M unique prompt tokens. "
            "Fixed 16-turn causal prefix window exceeds 5.582M CPU capacity; "
            "unchanged input/output lengths and 32 outstanding requests. "
            "Reserve both arms and handoff."
        ),
    )
    (campaign / "pressure_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    print(json.dumps(decision), flush=True)
    if not decision["selected"]:
        return
    replay.cell(baseline, "baseline", spec)
    (baseline / "stop.request").touch()
    while not (baseline / "stopped.json").exists():
        time.sleep(2)
    with (campaign / "full2_supervisor.log").open("w") as output:
        proc = subprocess.Popen(
            [
                ".venv/bin/python",
                "-m",
                "experiments.real_traces_20261012.server",
                "--root",
                str(full),
                "--arm",
                "full",
            ],
            stdout=output,
            stderr=output,
        )
    while not (full / "ready.json").exists():
        if proc.poll() is not None:
            raise RuntimeError("pressure full startup failed")
        time.sleep(2)
    replay.cell(
        full,
        "full",
        dict(
            id="warmup_moon8",
            dataset="mooncake_conversation",
            requests=8,
            concurrency=8,
            time_scale=4,
        ),
    )
    replay.cell(full, "full", spec)
    (campaign / "pressure_complete.json").write_text(
        json.dumps(dict(time=time.time(), spec=spec))
    )
    print("PRESSURE COMPARISON COMPLETE", flush=True)


if __name__ == "__main__":
    main()
