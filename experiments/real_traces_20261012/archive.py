# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Archive closed evidence and verify the compressed archive."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from experiments.real_traces_20261012.replay import CAMPAIGN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    for root in args.roots:
        assert (root / "stopped.json").exists(), root
    source = Path(__file__).resolve().parents[2]
    snapshot = CAMPAIGN / "code"
    shutil.copytree(
        source / "experiments/real_traces_20261012",
        snapshot,
        ignore=shutil.ignore_patterns("__pycache__"),
        dirs_exist_ok=True,
    )
    patch = subprocess.check_output(
        ["git", "diff", "c3c1a84", "HEAD", "--", "experiments/real_traces_20261012"],
        cwd=source,
    )
    (CAMPAIGN / "implementation.patch").write_bytes(patch)
    history = subprocess.check_output(
        ["git", "log", "--format=fuller", "c3c1a84..HEAD"], cwd=source, text=True
    )
    (CAMPAIGN / "commits.txt").write_text(history)
    provenance = dict(
        source=str(source),
        branch=subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=source, text=True
        ).strip(),
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        serving_code_unchanged=not subprocess.check_output(
            ["git", "diff", "c3c1a84", "HEAD", "--", "vllm"], cwd=source, text=True
        ),
        roots=[str(r) for r in args.roots],
    )
    (CAMPAIGN / "final_source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    directory = CAMPAIGN / "archives"
    directory.mkdir(exist_ok=True)
    archive = directory / "evidence.tar.zst"
    assert not archive.exists()
    parent = CAMPAIGN.parent
    inputs = [r.name for r in args.roots] + [CAMPAIGN.name]
    originals = parent / "trace_characterization_20261012"
    inputs += [
        str(p.relative_to(parent))
        for p in (originals / "mooncake").glob("*_trace.jsonl")
    ]
    inputs += [
        str((originals / "inferact/codex_swebenchpro.json").relative_to(parent)),
        str((originals / "download_manifest.json").relative_to(parent)),
    ]
    command = [
        "tar",
        "--use-compress-program=zstd -T8 -3",
        "--exclude=" + CAMPAIGN.name + "/archives",
        "--exclude=*/tmp",
        "-cf",
        str(archive),
        "-C",
        str(parent),
        *inputs,
    ]
    subprocess.run(command, check=True)
    listing = directory / "members.txt"
    with listing.open("w") as stream:
        subprocess.run(
            ["tar", "--use-compress-program=zstd -d", "-tf", str(archive)],
            check=True,
            stdout=stream,
        )
    subprocess.run(["zstd", "-t", str(archive)], check=True)
    h = hashlib.sha256()
    with archive.open("rb") as stream:
        while chunk := stream.read(16 * 1024**2):
            h.update(chunk)
    members = listing.read_text().splitlines()
    assert all(any(m.startswith(r.name + "/") for m in members) for r in args.roots)
    result = dict(
        time=time.time(),
        archive=str(archive),
        bytes=archive.stat().st_size,
        sha256=h.hexdigest(),
        members=len(members),
        verified=True,
        roots=provenance["roots"],
    )
    (CAMPAIGN / "archive_verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
