# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fingerprint completed evidence and archive the implementation after shutdown."""

import json
import subprocess
import time

from experiments.agentic_1m_20261012.common import DATA, ROOT, SOURCE, digest, save


def main():
    roots = []
    for root in sorted(ROOT.parent.glob("a1m[bf]*")):
        owner = root / "owner.json"
        if not owner.exists():
            continue
        info = json.loads(owner.read_text())
        if info["deadline"] not in (1791820841, 1791822821):
            continue
        assert (root / "stopped.json").exists(), root
        roots.append(root)
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    assert not occupied, occupied
    directory = ROOT / "archive"
    directory.mkdir(exist_ok=True)
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=SOURCE, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
    ).strip()
    base = subprocess.check_output(
        ["git", "rev-parse", "13ad6c6"], cwd=SOURCE, text=True
    ).strip()
    bundle = directory / "implementation.bundle"
    assert not bundle.exists(), "Preserve previous archives"
    subprocess.run(
        ["git", "bundle", "create", str(bundle), "HEAD", "^" + base],
        cwd=SOURCE,
        check=True,
    )
    with (directory / "implementation.patch").open("w") as stream:
        subprocess.run(
            ["git", "diff", base, "HEAD"], cwd=SOURCE, stdout=stream, check=True
        )
    paths = set(DATA.glob("*.json")) | set(DATA.glob("*.u32"))
    paths.update(ROOT.glob("*.json"))
    paths.add(ROOT / "frozen_predictor.json")
    for root in roots:
        paths.update(root.glob("*.json"))
        paths.update((root / "server").glob("*.json"))
        paths.update(root.glob("cells/*/result.json"))
        paths.update(root.glob("cells/*/spec.json"))
        paths.update(root.glob("cells/*/responses.jsonl"))
        paths.update(root.glob("cells/*/transfer_events.jsonl"))
    fingerprints = [
        dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))
        for path in sorted(paths)
        if path.exists()
    ]
    links = [
        dict(path=str(p), target=str(p.resolve()), sha256=digest(p))
        for p in (SOURCE / "vllm").glob("*.so")
    ]
    save(
        directory / "manifest.json",
        dict(
            time=time.time(),
            source=str(SOURCE),
            branch=branch,
            head=head,
            bundle_requires_base=base,
            bundle_sha256=digest(bundle),
            roots=[str(r) for r in roots],
            all_owned_stopped=True,
            gpu_processes=occupied,
            fingerprints=fingerprints,
            compiled_extensions=links,
            raw_logs=(
                "Preserved in place under the listed roots and campaign root; "
                "raw timelines and failed attempts are retained. HF weights "
                "remain outside the experiment tree and are not duplicated."
            ),
        ),
    )
    print(f"Archived implementation and {len(fingerprints)} evidence files", flush=True)


if __name__ == "__main__":
    main()
