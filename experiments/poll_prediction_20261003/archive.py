# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy compact, reproducible evidence into each local strategy branch."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def main(root):
    verification = json.loads((root / "summary/verification.json").read_text())
    assert verification["status"] == "verified"
    assert verification["measured_requests"] == 11960
    branches = json.loads((root / "branches.json").read_text())
    plain = [
        *root.joinpath("summary").glob("*"),
        root / "branches.json",
        root / "study.json",
        root / "policies.json",
        root / "predicted_readiness_policies.json",
        root / "continuation.json",
        root / "packaged_dependency_sha256.json",
    ]
    logs = [*root.glob("*.log"), root / "process_log.jsonl"]
    for branch in branches:
        destination = (
            Path(branch["path"]) / "experiments/poll_prediction_20261003/results"
        )
        destination.mkdir(parents=True, exist_ok=True)
        for path in plain:
            if path.is_file():
                subdir = "summary" if path.parent.name == "summary" else "manifests"
                target = destination / subdir / path.name
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(path, target)
        for path in logs:
            with gzip.open(destination / (path.name + ".gz"), "wb") as output:
                output.write(path.read_bytes())
        for phase in verification["phases"]:
            target = destination / phase
            target.mkdir(exist_ok=True)
            for filename in (
                "complete.json",
                "validation_complete.json",
                "provenance.json",
            ):
                shutil.copy2(root / phase / filename, target / filename)
            with gzip.open(target / "batch_shapes.json.gz", "wb") as output:
                output.write((root / phase / "batch_shapes.json").read_bytes())
            for kind in ("correctness", "correctness_distinct"):
                path = root / phase / kind / "summary.json"
                if path.exists():
                    shutil.copy2(path, target / (kind + ".json"))
        digest = {
            str(path.relative_to(destination)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(destination.rglob("*"))
            if path.is_file() and path.name not in {"index.json", "README.md"}
        }
        (destination / "index.json").write_text(json.dumps(digest, indent=2) + "\n")
        (destination / "README.md").write_text(
            "# Local experiment evidence\n\n"
            f"Strategy branch: `{branch['branch']}`.\n\n"
            f"Implemented runtime commit: `{branch['runtime_commit']}`. "
            "The combined screen runs all strategies from the hybrid branch; "
            "the prediction-to-readiness comparison uses its separate branch. "
            "Per-phase provenance records the actual evaluated checkout.\n\n"
            "All 78 registered replays and 11,960 measured requests passed the "
            "cache, transfer, output, GPU ownership and nominal-work gates. "
            "The summary preserves weaker results and all matched controls.\n\n"
            f"Complete raw telemetry, requests and server logs: `{root}`. "
            "This archive contains reports, plots, manifests, correctness results, "
            "CPU checks and compressed process/driver logs. Large raw telemetry "
            "and model/native files remain at the original paths. "
            "`index.json` records archive SHA256 hashes.\n\n"
            "See [findings](summary/FINDINGS.md), [decisions](summary/DECISIONS.md) "
            "and [full tables](summary/report.html).\n"
        )
        print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
