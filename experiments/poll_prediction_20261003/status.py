# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record completed cells and report the current experiment phase."""

import json
import time
from pathlib import Path

ROOT = Path("/mnt/localnvme/glm53_poll_prediction_20261003")
seen_path = ROOT / "observed_cells.json"
seen = json.loads(seen_path.read_text()) if seen_path.exists() else {}
new = []
for path in sorted(ROOT.glob("*/p*/*/result.json")):
    if str(path) in seen:
        continue
    row = json.loads(path.read_text())
    item = dict(
        path=str(path),
        accepted=row["accepted"],
        throughput=row["throughput"],
        mean_ttft_ms=row["mean_ttft_ms"],
    )
    seen[str(path)] = item
    new.append(item)
with (ROOT / "process_log.jsonl").open("a") as stream:
    for row in new:
        stream.write(
            json.dumps(dict(event="cell_observed", time=time.time(), **row)) + "\n"
        )
seen_path.write_text(json.dumps(seen, indent=2) + "\n")
log = ROOT / "driver.log"
lines = log.read_text().splitlines() if log.exists() else []
last = next(
    (
        line
        for line in reversed(lines)
        if line.startswith(("START", "CORRECTNESS", "READY"))
    ),
    None,
)
print(
    json.dumps(
        dict(
            cells=len(seen),
            accepted=sum(r["accepted"] for r in seen.values()),
            new=new,
            phase=last,
        ),
        indent=2,
    )
)
