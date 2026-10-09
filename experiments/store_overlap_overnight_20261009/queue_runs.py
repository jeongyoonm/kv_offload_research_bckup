# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run reviewed TP8 proposals sequentially, stopping on failure or budget."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/mnt/localnvme/jeongyoon/store_overlap_overnight_20261009")


def log(event, **fields):
    record = dict(time=time.time(), event=event, **fields)
    with (ARTIFACTS / "queue_process_log.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    deadline = (
        datetime.fromisoformat(plan["deadline_utc"])
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    if args.predecessor:
        summary = args.predecessor / "summary.json"
        while not summary.exists():
            assert time.time() < deadline, "Budget expired while waiting"
            time.sleep(20)
        result = json.loads(summary.read_text())
        assert result["complete"] and result["stopped"], args.predecessor
        log("predecessor_complete", root=str(args.predecessor))
    for cell in plan["runs"]:
        source, root = Path(cell["source"]), Path(cell["output"])
        assert not root.exists(), f"Use a fresh root: {root}"
        if time.time() + cell.get("reserve_seconds", 2700) >= deadline:
            log("budget_stop", pending=cell["label"])
            return
        occupied = ""
        for _ in range(12):
            occupied = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            if not occupied:
                break
            time.sleep(5)
        assert not occupied, f"GPUs remain occupied: {occupied}"
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        assert not subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=source,
            text=True,
        ).strip(), source
        command = [
            str(source / ".venv/bin/python"),
            cell["driver"],
            "--output",
            str(root),
            *cell.get("args", []),
        ]
        log(
            "run_started",
            label=cell["label"],
            source=str(source),
            root=str(root),
            commit=commit,
            command=command,
        )
        environment = {"PYTHONPATH": str(source)}
        import os

        environment = dict(os.environ, **environment)
        with (ARTIFACTS / f"{cell['label']}_driver.log").open("w") as output:
            subprocess.run(
                command,
                cwd=source,
                env=environment,
                stdout=output,
                stderr=output,
                check=True,
            )
        summary = json.loads((root / "summary.json").read_text())
        assert summary["complete"] and summary["stopped"], root
        if cell.get("report"):
            with (ARTIFACTS / f"{cell['label']}_report.log").open("w") as output:
                subprocess.run(
                    [str(source / ".venv/bin/python"), cell["report"], str(root)],
                    cwd=source,
                    env=environment,
                    stdout=output,
                    stderr=output,
                    check=True,
                )
        if cell.get("store_analysis", True):
            subprocess.run(
                [
                    str(SOURCE / ".venv/bin/python"),
                    str(Path(__file__).with_name("analyze_stores.py")),
                    str(root),
                ],
                cwd=SOURCE,
                check=True,
            )
        log(
            "run_complete",
            label=cell["label"],
            root=str(root),
            seconds=summary["seconds"],
        )


if __name__ == "__main__":
    main()
