# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Archive a completed serving allocation with member hashes and replay inputs."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def archive(source, destination):
    summary = json.loads((source / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"], source
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "raw_traces.tar.gz"
    assert not target.exists(), "Keep existing archives"
    members = []
    with tarfile.open(target, "w:gz", compresslevel=6) as bundle:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = str(path.relative_to(source))
            data = path.read_bytes()
            if relative == "server/environment.json":
                environment = json.loads(data)
                for key in list(environment):
                    if key.endswith(("_TOKEN", "_KEY", "_SECRET", "_PASSWORD")):
                        environment.pop(key)
                data = (json.dumps(environment, indent=2) + "\n").encode()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mtime = 0
            bundle.addfile(info, io.BytesIO(data))
            members.append(
                dict(
                    path=relative,
                    bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
    with tarfile.open(target, "r:gz") as bundle:
        expected = {m["path"]: m for m in members}
        for member in bundle.getmembers():
            data = bundle.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest() == expected[member.name]["sha256"]
        assert len(bundle.getmembers()) == len(members)
    manifest = dict(
        source=str(source),
        archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        archive_bytes=target.stat().st_size,
        members=members,
        verified=True,
    )
    (destination / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for name in (
        "comparison.json",
        "comparison.csv",
        "bookends.json",
        "plan.json",
        "source_provenance.json",
        "summary.json",
        "loaded_predictor.json",
    ):
        path = source / name
        if path.exists():
            (destination / name).write_bytes(path.read_bytes())
    print(json.dumps({k: v for k, v in manifest.items() if k != "members"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    archive(args.source, args.destination)


if __name__ == "__main__":
    main()
