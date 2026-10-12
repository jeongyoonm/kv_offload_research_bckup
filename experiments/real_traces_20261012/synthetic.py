# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional matched structural control from Mooncake's synthetic release."""

import datetime
import json
import subprocess
import time
from pathlib import Path

from experiments.real_traces_20261012 import replay

CUTOFF = datetime.datetime(
    2026, 10, 12, 3, 39, tzinfo=datetime.timezone.utc
).timestamp()


def main():
    campaign = replay.CAMPAIGN
    full = Path("/mnt/localnvme/jeongyoon/tr12f2")
    baseline = Path("/mnt/localnvme/jeongyoon/tr12b3")
    while not (campaign / "pressure_complete.json").exists():
        if (full / "stopped.json").exists():
            raise RuntimeError("full stopped")
        time.sleep(5)
    ref = json.loads(
        Path(
            "/mnt/localnvme/jeongyoon/tr12b1/cells/moon_conversation_stress_r0/result.json"
        ).read_text()
    )["seconds"]
    spec = dict(
        id="moon_synthetic_original_r0",
        dataset="mooncake_synthetic",
        requests=768,
        concurrency=128,
        time_scale=1.0,
    )
    reserve = ref * (768 / 1566) * 1.2 * 2 + 690
    decision = dict(
        time=time.time(),
        spec=spec,
        required_seconds=reserve,
        remaining_seconds=CUTOFF - time.time(),
        selected=time.time() + reserve < CUTOFF,
        reason=(
            "Secondary synthetic structural control, first 768 compatible "
            "requests, original arrival spacing, exact lengths. Reserve both "
            "arms, handoff, shutdown and archive; no policy change."
        ),
    )
    (campaign / "synthetic_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    print(json.dumps(decision), flush=True)
    if not decision["selected"]:
        return
    replay.cell(full, "full", spec)
    (full / "stop.request").touch()
    while not (full / "stopped.json").exists():
        time.sleep(2)
    with (campaign / "baseline3_supervisor.log").open("w") as output:
        proc = subprocess.Popen(
            [
                ".venv/bin/python",
                "-m",
                "experiments.real_traces_20261012.server",
                "--root",
                str(baseline),
                "--arm",
                "baseline",
            ],
            stdout=output,
            stderr=output,
        )
    while not (baseline / "ready.json").exists():
        if proc.poll() is not None:
            raise RuntimeError("synthetic baseline startup failed")
        time.sleep(2)
    replay.cell(
        baseline,
        "baseline",
        dict(
            id="warmup_moon8",
            dataset="mooncake_conversation",
            requests=8,
            concurrency=8,
            time_scale=4,
        ),
    )
    replay.cell(baseline, "baseline", spec)
    (campaign / "synthetic_complete.json").write_text(
        json.dumps(dict(time=time.time(), spec=spec))
    )
    print("SYNTHETIC COMPARISON COMPLETE", flush=True)


if __name__ == "__main__":
    main()
