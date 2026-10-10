# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Archive drained experiment evidence, branch patches and reproducible source."""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
BASE = "875d1cf474c6348b796604bd19c17026dca86096"


def digest(stream):
    h = hashlib.sha256()
    while block := stream.read(1024**2):
        h.update(block)
    return h.hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return digest(stream)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=SOURCE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--worktree", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve earlier archives"
    args.output.mkdir(parents=True)
    sources = []
    for worktree in args.worktree:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=worktree, text=True
        )
        assert not status, (worktree, status)
        sources.append(
            dict(
                path=str(worktree),
                branch=subprocess.check_output(
                    ["git", "branch", "--show-current"], cwd=worktree, text=True
                ).strip(),
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=worktree, text=True
                ).strip(),
            )
        )
    branches = [s["branch"] for s in sources]
    bundle = args.output / "proposals.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle), *branches, "^" + BASE],
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
    (args.output / "bundle_verification.txt").write_text(
        verified.stdout + verified.stderr
    )
    patches = args.output / "patches"
    patches.mkdir()
    for source in sources:
        name = source["branch"].replace("/", "_") + ".patch"
        (patches / name).write_bytes(git("diff", "--binary", BASE, source["commit"]))
    parent = args.output / "baseline_source.tar.gz"
    proc = subprocess.Popen(
        ["git", "archive", "--format=tar", BASE], cwd=SOURCE, stdout=subprocess.PIPE
    )
    with gzip.open(parent, "wb", compresslevel=1) as stream:
        shutil.copyfileobj(proc.stdout, stream, length=1024**2)
    proc.stdout.close()
    assert proc.wait() == 0
    manifest = []
    evidence = args.output / "evidence.tar"
    with tarfile.open(evidence, "w") as archive:
        for root in args.root:
            assert root.is_dir()
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if (
                    path.is_symlink()
                    or "tmp" in relative.parts
                    or path.suffix == ".mmap"
                ):
                    continue
                if not path.is_file():
                    continue
                before = path.stat()
                member = str(Path(root.name) / relative)
                checksum = file_digest(path)
                archive.add(path, arcname=member, recursive=False)
                after = path.stat()
                assert (before.st_size, before.st_mtime_ns) == (
                    after.st_size,
                    after.st_mtime_ns,
                ), path
                manifest.append(dict(path=member, size=before.st_size, sha256=checksum))
    expected = {m["path"]: m for m in manifest}
    assert len(expected) == len(manifest), "Duplicate archive member"
    with tarfile.open(evidence) as archive:
        for member in archive:
            assert member.isfile()
            assert member.size == expected[member.name]["size"]
            with archive.extractfile(member) as stream:
                assert digest(stream) == expected[member.name]["sha256"]
    metadata = dict(
        verified_utc=datetime.now(timezone.utc).isoformat(),
        all_member_hashes_match=True,
        files=len(manifest),
        members=manifest,
        sources=sources,
        baseline_commit=BASE,
        restoration="Extract baseline_source, then apply a branch patch. "
        "The Git bundle preserves proposal commits and requires the recorded "
        "baseline commit; it was verified against the experiment repository.",
        exclusions="Virtual environments, model weights and ephemeral tmp/mmap "
        "storage are excluded. Raw logs, frozen models, datasets, traces, "
        "reports, preflights and failed startup evidence are retained.",
        archives={
            p.name: dict(bytes=p.stat().st_size, sha256=file_digest(p))
            for p in [parent, bundle, evidence]
        },
    )
    (args.output / "verification.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in [
                    "verified_utc",
                    "all_member_hashes_match",
                    "files",
                    "archives",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
