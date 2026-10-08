# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit proportional load-only comparison with the original paired analyzer."""

import importlib.util
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "deadline_analysis", SOURCE / "experiments/offset_deadline_ab_20261008/analyze.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
analysis.ROOT = Path("/mnt/localnvme/glm53_load_only_fraction_20261008")

if __name__ == "__main__":
    for path in (analysis.ROOT / "live").glob("*/r[0-7]_*/late.json"):
        for event in json.loads(path.read_text())["records"]:
            if event["event"] != "membership_sealed":
                continue
            transport = event["transport"]
            if transport.get("has_compute") is not False:
                continue
            count = transport["load_only_pending_candidates"]
            target = (count * 50 + 99) // 100
            assert transport["load_only_ready_percent"] == 50
            assert transport["load_only_fraction_target"] == target
            assert transport["ready_target"] == min(
                target, count - len(transport["cancelled"])
            )
    analysis.main()
