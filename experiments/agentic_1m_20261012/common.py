# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locations and immutable campaign limits for the long-context trace study."""

import hashlib
import json
import time
from pathlib import Path

ROOT = Path("/mnt/localnvme/jeongyoon/agentic_1m_20261012")
SOURCE = Path(__file__).resolve().parents[2]
CHAR = Path("/mnt/localnvme/jeongyoon/trace_characterization_20261012")
DATA = ROOT / "datasets"
LIMIT = 1048576
PORT = 18482
BASE = f"http://127.0.0.1:{PORT}"
START = 1791784841
DEADLINE = START + 36000
MEASURE_CUTOFF = DEADLINE - 900
STATE = ROOT / "frozen_predictor.json"
STATE_SHA = "f159f32d226894457fb56b3195c47d4ff6e2a5d982d8d600b1970ea000294f96"


def cache_settings():
    return json.loads((ROOT / "cache_budget.json").read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            result.update(chunk)
    return result.hexdigest()


def log(event, **fields):
    ROOT.mkdir(parents=True, exist_ok=True)
    item = dict(time=time.time(), event=event, **fields)
    with (ROOT / "process_log.jsonl").open("a") as stream:
        stream.write(json.dumps(item) + "\n")
    print(json.dumps(item), flush=True)
