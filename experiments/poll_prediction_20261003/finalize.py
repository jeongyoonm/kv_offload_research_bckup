# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and summarize every registered replay after server teardown."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/localnvme/glm53_poll_prediction_20261003")
SOURCE = Path(__file__).resolve().parents[2]
HELPERS = SOURCE / "experiments/poll_prediction_20261003"
PHASES = {"screen": 42, "confirmation": 24, "predicted_readiness": 12}
WORK = {"p8192_f95_h100": 134400, "p61440_f95_h100": 245760, "p61440_f99_h100": 51200}


def main():
    started = time.monotonic()
    while not all((ROOT / phase / "complete.json").exists() for phase in PHASES):
        if time.monotonic() - started > 3 * 3600:
            raise TimeoutError("Registered GPU phases have not completed")
        time.sleep(20)
    gates = {}
    requests = 0
    for phase, count in PHASES.items():
        directory = ROOT / phase
        complete = json.loads((directory / "complete.json").read_text())
        assert complete["cells"] == complete["accepted"] == count, complete
        for helper in ("analyze", "batch_shapes"):
            with (ROOT / f"{phase}_{helper}.log").open("w") as log:
                subprocess.run(
                    [sys.executable, str(HELPERS / f"{helper}.py"), str(directory)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    cwd=SOURCE,
                )
        validated = json.loads((directory / "validation_complete.json").read_text())
        assert validated["cells"] == validated["accepted"] == count, validated
        shapes = json.loads((directory / "batch_shapes.json").read_text())["cells"]
        assert len(shapes) == count
        for row in shapes:
            workload = Path(row["path"]).parent.name
            assert not row["invalid_steps"], row["path"]
            assert row["total_uncached_tokens"] == WORK[workload], row["path"]
            requests += 300 if workload == "p8192_f95_h100" else 80
        probes = {}
        for kind in ("correctness", "correctness_distinct"):
            path = directory / kind / "summary.json"
            if not path.exists():
                assert phase == "screen" and kind == "correctness_distinct"
                continue
            summary = json.loads(path.read_text())
            assert summary and all(row["passed"] for row in summary.values()), path
            probes[kind] = {
                policy: {
                    key: value
                    for key, value in row.items()
                    if key
                    in {
                        "requests",
                        "text_matches",
                        "factual_matches",
                        "exact_cold_matches",
                        "passed",
                    }
                }
                for policy, row in summary.items()
            }
        gates[phase] = dict(
            cells=count,
            accepted=count,
            probes=probes,
            equal_uncached_work=True,
            valid_gpu_timings=True,
        )
    assert requests == 11960, requests
    native = {}
    for branch in json.loads((ROOT / "branches.json").read_text()):
        for filename, expected in branch["native"].items():
            with (Path(branch["path"]) / filename).open("rb") as library:
                actual = hashlib.file_digest(library, "sha256").hexdigest()
            assert actual == expected, (branch["path"], filename)
        native[branch["branch"]] = len(branch["native"])
    for helper in ("report", "prediction_diagnostics", "plot", "findings"):
        with (ROOT / f"final_{helper}.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(HELPERS / f"{helper}.py"), str(ROOT)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=SOURCE,
            )
    result = dict(
        time=time.time(),
        phases=gates,
        measured_requests=requests,
        native_hashes_verified=native,
        status="verified",
    )
    (ROOT / "summary/verification.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "process_log.jsonl").open("a") as log:
        log.write(json.dumps(dict(event="all_phases_verified", **result)) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
