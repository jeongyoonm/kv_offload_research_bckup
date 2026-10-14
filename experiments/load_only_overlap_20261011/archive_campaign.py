# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report and archive the completed campaign or its explicit deadline snapshot."""

import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path

from experiments.load_only_overlap_20261011.campaign import BASE, CAMPAIGN, SOURCE
from experiments.load_only_overlap_20261011.report_diverse import ROOTS

SNAPSHOT_DEADLINE = 1791738510  # 17:08:30 UTC, after owned-server watchdog cleanup.


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(8 * 1024**2):
            digest.update(data)
    return digest.hexdigest()


def exclude_scratch(info):
    if "tmp" in Path(info.name).parts or "__pycache__" in Path(info.name).parts:
        return None
    return info


def main():
    while time.time() < SNAPSHOT_DEADLINE:
        states = [BASE / name / "summary.json" for name in ROOTS]
        if all(p.exists() for p in states) and all(
            json.loads(p.read_text()).get("stopped") for p in states
        ):
            break
        time.sleep(30)
    archive = CAMPAIGN / "archives"
    archive.mkdir(exist_ok=True)
    with (CAMPAIGN / "report_final.log").open("w") as log:
        reported = subprocess.run(
            [
                str(SOURCE / ".venv/bin/python"),
                "-m",
                "experiments.load_only_overlap_20261011.report_diverse",
                "--output",
                str(CAMPAIGN / "final_report"),
            ],
            cwd=SOURCE,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    progress = json.loads((CAMPAIGN / "progress.json").read_text())
    branches = progress["branches"]
    for branch in branches.values():
        branch["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=branch["path"], text=True
        ).strip()
        branch["dirty"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=branch["path"], text=True
        ).strip()
    parent = json.loads((CAMPAIGN / "campaign.json").read_text())["parent"]
    bundle = archive / "proposals.bundle"
    assert not bundle.exists()
    subprocess.run(
        [
            "git",
            "bundle",
            "create",
            str(bundle),
            *(b["branch"] for b in branches.values()),
            "^" + parent,
        ],
        cwd=SOURCE,
        check=True,
    )
    verified = subprocess.run(
        ["git", "bundle", "verify", str(bundle)],
        cwd=SOURCE,
        capture_output=True,
        text=True,
        check=True,
    )
    (archive / "bundle_verify.log").write_text(verified.stdout + verified.stderr)
    manifest = dict(
        snapshot_utc=time.time(),
        branches=branches,
        parent=parent,
        report_returncode=reported.returncode,
        bundle_sha256=sha256(bundle),
        exclusions=["all tmp directories", "__pycache__", "model weights"],
        scope="Raw successful and failed campaign evidence; external runtime "
        "dependencies are recorded by fingerprint and remain in their original paths.",
    )
    roots = set(ROOTS) | {
        "lo11a",
        "lo11b",
        "lo11c",
        "hb11qa",
        "hb11pb",
        "dc11bg",
        "dc11oh",
        "dc11pr",
        "dc11ps",
    }
    manifest["raw_roots"] = sorted(name for name in roots if (BASE / name).exists())
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    raw = archive / "raw_evidence.tar.gz"
    assert not raw.exists()
    with tarfile.open(raw, "w:gz", compresslevel=1, dereference=False) as tar:
        for name in manifest["raw_roots"]:
            tar.add(BASE / name, arcname=name, filter=exclude_scratch)
        for path in CAMPAIGN.iterdir():
            if path != archive:
                tar.add(path, arcname=f"campaign/{path.name}", filter=exclude_scratch)
    manifest.update(
        raw_archive_bytes=raw.stat().st_size, raw_archive_sha256=sha256(raw)
    )
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
