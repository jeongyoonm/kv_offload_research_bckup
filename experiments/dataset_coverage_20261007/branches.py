# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record each data optimization on a branch with immutable checkpoint hashes."""

import hashlib
import json
import os
import subprocess

import run


def git(*arguments, cwd=run.SOURCE):
    return subprocess.check_output(["git", *arguments], cwd=cwd, text=True).strip()


def main():
    selection = json.loads((run.ROOT / "study/selection.json").read_text())
    audit = json.loads((run.ROOT / "study/dataset_audit.json").read_text())
    base = git("rev-parse", "HEAD")
    staging = run.ROOT / "branch_staging"
    if not staging.exists():
        git("worktree", "add", "--detach", str(staging), base)
    if not (staging / ".venv").exists():
        (staging / ".venv").symlink_to(run.SOURCE / ".venv", target_is_directory=True)
    recorded = {}
    for variant in selection["variants"]:
        branch = f"experiment/glm53-dataset-{variant.replace('_', '-')}-20261007"
        existing = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=run.SOURCE,
        )
        if existing.returncode == 0:
            recorded[variant] = dict(branch=branch, commit=git("rev-parse", branch))
            continue
        git("switch", "-c", branch, base, cwd=staging)
        checkpoints = {
            str(seed): run.ROOT / "study/models" / variant / f"seed{seed}/serving.json"
            for seed in [73, 74, 75]
        }
        manifest = dict(
            variant=variant,
            implementation=base,
            architecture="Extra Trees base + histogram gradient boosting correction",
            recipe_unchanged=not variant.endswith("_mass_matched"),
            hyperparameters_unchanged=True,
            weighting=(
                "source_mass_matched" if variant.endswith("_mass_matched") else "legacy"
            ),
            input_features_unchanged=True,
            new_forecast_samples=audit["counts"][variant],
            one_matching_base_row_per_forecast=True,
            dataset_hashes=audit["hashes"],
            checkpoints={
                seed: dict(
                    path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()
                )
                for seed, p in checkpoints.items()
            },
        )
        path = staging / "experiments/dataset_coverage_20261007/variant.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
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
                f"Record {variant} dataset optimization and frozen checkpoints",
                "-m",
                "Co-authored-by: Codex <noreply@openai.com>",
                "-m",
                "Signed-off-by: Codex <noreply@openai.com>",
            ],
            cwd=staging,
            env=env,
            check=True,
        )
        recorded[variant] = dict(
            branch=branch, commit=git("rev-parse", "HEAD", cwd=staging)
        )
    run.save(run.ROOT / "study/branches.json", recorded)
    git("switch", "--detach", base, cwd=staging)
    git("worktree", "remove", "--force", str(staging))
    run.log("variant_branches_recorded", variants=len(recorded))


if __name__ == "__main__":
    main()
