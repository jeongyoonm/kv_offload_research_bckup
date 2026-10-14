# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reserve every mirror before selecting reverse-order follow-up cells."""

import datetime
import json
import subprocess
import time
from pathlib import Path

from experiments.real_traces_20261012 import replay

CUTOFF = datetime.datetime(
    2026, 10, 12, 3, 34, tzinfo=datetime.timezone.utc
).timestamp()


def log(event, **kwargs):
    entry = dict(time=time.time(), event=event, **kwargs)
    with (replay.CAMPAIGN / "extras_process_log.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)


def main():
    campaign = replay.CAMPAIGN
    full = Path("/mnt/localnvme/jeongyoon/tr12f1")
    baseline = Path("/mnt/localnvme/jeongyoon/tr12b2")
    while not (campaign / "primary_complete.json").exists():
        if (full / "stopped.json").exists():
            raise RuntimeError("full stopped")
        time.sleep(5)
    ref = json.loads(
        Path(
            "/mnt/localnvme/jeongyoon/tr12b1/cells/inferact_s64_n32_r0/result.json"
        ).read_text()
    )["seconds"]
    specs = json.loads((campaign / "extra_inferact.json").read_text())
    selected = []
    reserved = 0
    for spec in specs:
        factor = 1 if spec["sessions"] == 64 else 2.2
        full_budget = ref * factor
        baseline_budget = ref * factor * 1.2
        required = full_budget + reserved + baseline_budget + 600
        if time.time() + required > CUTOFF:
            log(
                "extra_deferred",
                spec=spec,
                required_seconds=required,
                remaining_seconds=CUTOFF - time.time(),
                reason="Reserve mirror and server handoff before deadline",
            )
            continue
        log(
            "extra_selected",
            spec=spec,
            required_seconds=required,
            remaining_seconds=CUTOFF - time.time(),
        )
        result = replay.cell(full, "full", spec)
        assert result["accepted"]
        selected.append(spec)
        reserved += max(baseline_budget, result["seconds"] * 1.2)
        (campaign / "extra_selected.json").write_text(
            json.dumps(selected, indent=2) + "\n"
        )
    if not selected:
        log("extras_none_selected")
        return
    (full / "stop.request").touch()
    while not (full / "stopped.json").exists():
        time.sleep(2)
    with (campaign / "baseline2_supervisor.log").open("w") as output:
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
            raise RuntimeError("mirror baseline startup failed")
        time.sleep(2)
    # Warm kernels only; each measured cell independently resets both caches.
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
    for spec in selected:
        log("mirror_started", spec=spec, remaining_seconds=CUTOFF - time.time())
        assert time.time() < CUTOFF
        replay.cell(baseline, "baseline", spec)
    (campaign / "extras_complete.json").write_text(
        json.dumps(dict(time=time.time(), selected=selected, baseline=str(baseline)))
    )
    log("extras_complete", selected=selected)


if __name__ == "__main__":
    main()
