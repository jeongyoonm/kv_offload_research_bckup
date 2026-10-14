# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resume matched serving campaigns with explicit wall-clock coverage decisions."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
CAMPAIGN = Path("/mnt/localnvme/jeongyoon/load_only_overlap_20261011")
BASE = CAMPAIGN.parent
DEADLINE = 1791738420


def accepted(root, expected):
    data = json.loads((root / "summary.json").read_text())
    assert data["complete"] and data["stopped"], root
    assert len(data["results"]) == expected, root
    assert all(r["accepted"] for r in data["results"]), root
    return data


def coverage(concurrencies, reason):
    plan = dict(
        expected_cells=126 + 39 * len(concurrencies),
        profiled_class_concurrencies=[1, 2, 4, 8, 16, 32, 64],
        other_shape_concurrencies=sorted([32, *concurrencies]),
        heterogeneous_and_evolving_requests=300,
        heterogeneous_and_evolving_repeats=2,
        homogeneous_requests=64,
        homogeneous_repeats=1,
        decision_utc=time.time(),
        reason=reason,
        initial_core_cells=243,
        full_illustrative_cells=360,
    )
    (CAMPAIGN / "coverage_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(event="coverage_decision", **plan)), flush=True)


def execute(rootname, kind, suite, expected, options):
    root = BASE / rootname
    if (root / "summary.json").exists():
        accepted(root, expected)
        print(json.dumps(dict(event="reuse_completed", root=str(root))), flush=True)
        return
    assert not root.exists(), root
    assert time.time() < DEADLINE - 300, "Campaign deadline"
    command = [
        str(SOURCE / ".venv/bin/python"),
        "-m",
        "experiments.load_only_overlap_20261011.evaluate",
        "--output",
        str(root),
        "--server-kind",
        kind,
        "--suite",
        suite,
        "--idle-target",
        "4",
        "--seconds",
        str(int(DEADLINE - time.time() - 60)),
        *options,
    ]
    print(json.dumps(dict(event="launch", root=str(root), command=command)), flush=True)
    with (CAMPAIGN / f"{rootname}.log").open("x") as log:
        subprocess.run(
            command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    data = accepted(root, expected)
    print(
        json.dumps(
            dict(
                event="completed",
                root=str(root),
                cells=expected,
                seconds=data["seconds"],
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--superseded-pid", type=int, required=True)
    args = parser.parse_args()
    command = Path(f"/proc/{args.superseded_pid}/cmdline").read_bytes()
    assert str(CAMPAIGN / "run_remaining_campaign.py").encode() in command
    coverage([1, 64], "Measured cold priming cost; prioritize complete matched arms")
    root = BASE / "dc11bx"
    while not (root / "summary.json").exists():
        assert time.time() < DEADLINE - 300
        time.sleep(20)
    accepted(root, 4)
    os.kill(args.superseded_pid, signal.SIGKILL)
    print(json.dumps(dict(event="superseded_controller_stopped")), flush=True)
    classes = ["--classes", str(CAMPAIGN / "profiled_classes.json")]
    paired = ["--requests", "300", "--repeats", "2", "--concurrencies", "32"]
    execute(
        "dc11ox",
        "optimized",
        "heterogeneous",
        8,
        [*paired, *classes, "--seed-start", "12511"],
    )
    execute("dc11be", "baseline", "evolving", 4, [*paired, "--seed-start", "12601"])
    execute("dc11oe", "optimized", "evolving", 8, [*paired, "--seed-start", "12601"])
    classes_data = json.loads((CAMPAIGN / "profiled_classes.json").read_text())
    selected = [f"c{c}_q{q}" for c, q in classes_data.values()]
    assert len(set(selected)) == 3
    sweep = [
        "--requests",
        "64",
        "--repeats",
        "1",
        "--concurrencies",
        "1",
        "2",
        "4",
        "8",
        "16",
        "64",
        "--seed-start",
        "12501",
        "--ids",
        *selected,
    ]
    execute("dc11bc", "baseline", "grid", 18, sweep)
    execute("dc11oc", "optimized", "grid", 36, sweep)
    remaining = [
        f"c{c}_q{q}"
        for c in (4096, 16384, 32768, 65536)
        for q in (128, 512, 2048, 8192)
        if f"c{c}_q{q}" not in selected
    ]
    # Completed grid wall times imply about 150 minutes for all three points.
    # Leave 20 minutes for teardown, reporting, and uncertainty.
    points = [1, 16, 64] if DEADLINE - time.time() > 170 * 60 else [1, 64]
    coverage(
        points, "Measured remaining time after primary mixed and dense-class sweeps"
    )
    sampled = [
        "--requests",
        "64",
        "--repeats",
        "1",
        "--concurrencies",
        *map(str, points),
        "--seed-start",
        "12501",
        "--ids",
        *remaining,
    ]
    execute("dc11bs", "baseline", "grid", 13 * len(points), sampled)
    execute("dc11os", "optimized", "grid", 26 * len(points), sampled)
    print(json.dumps(dict(event="campaign_queue_complete")), flush=True)


if __name__ == "__main__":
    main()
