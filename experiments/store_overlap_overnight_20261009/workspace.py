# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse the validated environment and generated dependencies in a worktree."""

import argparse
import hashlib
import json
from pathlib import Path


def link_dependencies(source, destination):
    paths = [Path(".venv")]
    paths += [path.relative_to(source) for path in (source / "vllm").rglob("*.so")]
    paths += [
        Path("vllm/_version.py"),
        Path("vllm/third_party/flashmla/flash_mla_interface.py"),
        Path("vllm/third_party/deep_gemm/__init__.py"),
        Path("vllm/third_party/deep_gemm/envs.py"),
    ]
    paths += [
        Path("vllm/third_party/deep_gemm") / name
        for name in ("legacy", "mega", "testing", "utils", "include")
    ]
    records = []
    for relative in paths:
        target, path = source / relative, destination / relative
        assert target.exists(), target
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.symlink_to(target, target_is_directory=target.is_dir())
        assert path.resolve() == target.resolve(), path
        records.append(
            dict(
                relative=str(relative),
                resolved=str(target.resolve()),
                sha256=(
                    hashlib.sha256(target.read_bytes()).hexdigest()
                    if target.is_file()
                    else None
                ),
            )
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    records = link_dependencies(args.source, args.destination)
    args.manifest.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
