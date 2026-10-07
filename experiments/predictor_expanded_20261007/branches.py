# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve each CPU architecture and follow-up candidate on its own branch."""

import hashlib
import json
import os
import subprocess

import study


def git(*args, cwd=study.SOURCE):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main():
    base = git("rev-parse", "HEAD")
    staging = study.ROOT / "branch_staging"
    if not staging.exists():
        git("worktree", "add", "--detach", str(staging), base)
    if not (staging / ".venv").exists():
        (staging / ".venv").symlink_to(study.SOURCE / ".venv", target_is_directory=True)
    plan = json.loads((study.ROOT / "improvement_plan.json").read_text())
    names = [("architecture", n) for n in study.arch.MODELS]
    names += [("improvement", n) for n in plan["candidates"] if n != "selected"]
    recorded = {}
    for family, name in names:
        branch = f"experiment/glm53-expanded-{family}-{name.replace('_', '-')}-20261007"
        exists = (
            subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
                cwd=study.SOURCE,
            ).returncode
            == 0
        )
        if exists:
            recorded[f"{family}:{name}"] = dict(
                branch=branch, commit=git("rev-parse", branch)
            )
            continue
        git("switch", "-c", branch, base, cwd=staging)
        parent = "folds" if family == "architecture" else "improvements"
        checkpoints = {}
        for seed in [73, 74, 75]:
            path = study.ROOT / parent / f"benchmark_wide_s{seed}/checkpoints.json"
            states = json.loads(path.read_text())
            selected = states[name]
            encoded = json.dumps(selected, sort_keys=True).encode()
            export = study.ROOT / "exports" / family / name / f"seed{seed}.json"
            study.arch.dump(export, selected)
            checkpoints[str(seed)] = dict(
                path=str(export),
                checkpoint_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),
                canonical_state_sha256=hashlib.sha256(encoded).hexdigest(),
            )
        manifest = dict(
            family=family,
            candidate=name,
            implementation=base,
            architecture=plan["architecture"] if family == "improvement" else name,
            checkpoints=checkpoints,
            inventory=json.loads((study.ROOT / "inventory.json").read_text()),
            cpu_only=True,
            inference_time_fitting=False,
            final_benchmark_used_for_selection=False,
            final_benchmark_is_reused=True,
        )
        path = staging / "experiments/predictor_expanded_20261007/variant.json"
        study.arch.dump(path, manifest)
        git("add", str(path.relative_to(staging)), cwd=staging)
        env = dict(
            os.environ, PATH=str(staging / ".venv/bin") + ":" + os.environ["PATH"]
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Codex",
                "-c",
                "user.email=noreply@openai.com",
                "commit",
                "-m",
                f"Record expanded predictor {family} {name}",
                "-m",
                "Co-authored-by: Codex <noreply@openai.com>",
                "-m",
                "Signed-off-by: Codex <noreply@openai.com>",
            ],
            cwd=staging,
            env=env,
            check=True,
        )
        recorded[f"{family}:{name}"] = dict(
            branch=branch, commit=git("rev-parse", "HEAD", cwd=staging)
        )
    study.arch.dump(study.ROOT / "branches.json", recorded)
    git("switch", "--detach", base, cwd=staging)
    git("worktree", "remove", "--force", str(staging))


if __name__ == "__main__":
    main()
