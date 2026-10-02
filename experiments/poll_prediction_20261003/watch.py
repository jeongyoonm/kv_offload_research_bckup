"""Update preliminary summaries as completed replay cells become available."""

import json
import time
from pathlib import Path

from analyze import analyze
from report import summarize

ROOT = Path("/mnt/localnvme/glm53_poll_prediction_20261003")
started = time.monotonic()
while time.monotonic() - started < 6 * 3600:
    changed = False
    for path in sorted(ROOT.glob("*/p*/*/result.json")):
        if (path.parent / "analysis.json").exists():
            continue
        row = analyze(path.parent)
        changed = True
        with (ROOT / "process_log.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    dict(
                        event="cell_analyzed",
                        time=time.time(),
                        path=str(path.parent),
                        accepted=row["accepted"],
                        throughput=row["throughput"],
                    )
                )
                + "\n"
            )
        print(path.parent, row["throughput"], row["accepted"], flush=True)
    if changed:
        summarize(ROOT)
    log = ROOT / "driver.log"
    if log.exists() and "Traceback (most recent call last)" in log.read_text():
        break
    if (ROOT / "screen/complete.json").exists():
        break
    time.sleep(20)
