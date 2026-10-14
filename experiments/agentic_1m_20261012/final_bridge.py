# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Finish the last requested pair with a bounded failed-attempt recovery window."""

import json
import time
from pathlib import Path

from experiments.agentic_1m_20261012 import campaign
from experiments.agentic_1m_20261012.common import DEADLINE, ROOT, log, save


def main():
    full = Path("/mnt/localnvme/jeongyoon/a1mf3")
    baseline = Path("/mnt/localnvme/jeongyoon/a1mb9")
    deadline = DEADLINE + 1980
    campaign.MEASURE_CUTOFF = deadline - 300
    specs = json.loads((ROOT / "primary_plan_mixed.json").read_text())
    target = next(s for s in specs if s["id"] == "tracelab_mixed_windows_n8_r0")
    for spec in specs:
        if spec == target or spec.get("optional"):
            continue
        path = full / "cells" / spec["id"] / "result.json"
        while not path.exists():
            if (full / "stopped.json").exists() or time.time() > DEADLINE:
                raise RuntimeError(f"Preceding requested cell did not finish: {path}")
            time.sleep(2)
        result = json.loads(path.read_text())
        assert result["accepted"], spec
    save(
        ROOT / "final_bridge_plan.json",
        dict(
            target=target,
            original_deadline=DEADLINE,
            recovery_deadline=deadline,
            reason=(
                "Complete the already requested C8 comparison. One full C32 "
                "attempt was rejected after HTTP disconnects; allow at most its "
                "33-minute recovery time. No additional repeats or new cells."
            ),
        ),
    )
    owned = [full]
    results = []
    try:
        result = campaign.run_cell(full, "full", target)
        results.append(dict(root=str(full), **result))
        campaign.stop_server(full)
        owned.append(baseline)
        campaign.start_server(baseline, "baseline", deadline=deadline)
        campaign.preflight(baseline, "baseline")
        result = campaign.run_cell(baseline, "baseline", target)
        results.append(dict(root=str(baseline), **result))
        campaign.stop_server(baseline)
        save(
            ROOT / "final_bridge_complete.json", dict(time=time.time(), results=results)
        )
        log("final_requested_pair_complete", results=len(results))
    finally:
        for root in owned:
            if (root / "owner.json").exists() and not (root / "stopped.json").exists():
                campaign.stop_server(root)


if __name__ == "__main__":
    main()
