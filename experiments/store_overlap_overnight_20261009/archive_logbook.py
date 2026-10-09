# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Archive the experiment logbook, deduplicating identical replay binaries."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    assert not args.destination.is_relative_to(args.source)
    args.destination.mkdir(parents=True, exist_ok=True)
    target = args.destination / "logbook.tar.gz"
    assert not target.exists(), "Use a fresh archive path"
    members, hashes = [], {}
    with tarfile.open(target, "w:gz", compresslevel=6) as bundle:
        for path in sorted(args.source.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = str(path.relative_to(args.source))
            data = path.read_bytes()
            if path.name == "environment.json":
                environment = json.loads(data)
                for key in list(environment):
                    if key.endswith(("_TOKEN", "_KEY", "_SECRET", "_PASSWORD")):
                        environment.pop(key)
                data = (json.dumps(environment, indent=2) + "\n").encode()
            digest = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(relative)
            info.mtime = 0
            info.mode = path.stat().st_mode & 0o777
            duplicate = hashes.get(digest)
            if duplicate is None:
                hashes[digest] = relative
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
            else:
                info.type = tarfile.LNKTYPE
                info.linkname = duplicate
                bundle.addfile(info)
            members.append(
                dict(
                    path=relative,
                    bytes=len(data),
                    sha256=digest,
                    link_to=duplicate,
                )
            )
    expected = {member["path"]: member for member in members}
    with tarfile.open(target, "r:gz") as bundle:
        actual = bundle.getmembers()
        assert len(actual) == len(members)
        for member in actual:
            data = bundle.extractfile(member).read()
            entry = expected[member.name]
            assert len(data) == entry["bytes"], member.name
            assert hashlib.sha256(data).hexdigest() == entry["sha256"], member.name
    manifest = dict(
        source=str(args.source),
        archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        archive_bytes=target.stat().st_size,
        members=members,
        verified=True,
        definition="Every regular file is preserved with its content hash. "
        "Identical files use tar hard links, including repeated replay "
        "binaries. Environment credential fields are removed from archives. "
        "All members were reopened and checked, including hard-link targets.",
    )
    (args.destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "members"}))


if __name__ == "__main__":
    main()
