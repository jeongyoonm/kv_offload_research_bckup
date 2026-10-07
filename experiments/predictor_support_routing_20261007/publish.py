# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify and preserve the support-routing experiment."""

import json
import os
import subprocess
import time

import numpy as np
import route
from threadpoolctl import threadpool_limits

r = route
p = r.importlib.import_module("publication")


def main():
    tasks = json.loads((r.e.DATA / "benchmark_tasks.json").read_text())
    pool = r.joblib.load(r.e.DATA / "benchmark_pool.joblib")
    os.environ["VLLM_DEFERRAL_FAST_FOREST"] = str(
        r.e.expanded.HIST / "fast_forest/build/libpredictor_forest_walk.so"
    )
    exports, runtime = {n: {} for n in r.KNOWN}, {}
    for task in [t for t in tasks if t["regime"] == "wide"]:
        states = p.state_set(task)
        support = json.loads(
            (r.ROOT / "folds" / task["name"] / "support.json").read_text()
        )
        x = r.e.features([pool["forecasts"][i] for i in task["test"]])
        with np.load(r.ROOT / "folds" / task["name"] / "predictions.npz") as f:
            expected = {n: f[n].copy() for n in r.KNOWN}
        for name, source in r.KNOWN.items():
            state = dict(
                signatures=support["signatures"],
                feature_indices=r.INDICES,
                known_model=source,
                expert=states[source],
                envelope=states["suffix_guard_envelope_boost"],
            )
            path = r.ROOT / "exports" / name / f"seed{task['seed']}.json"
            r.e.ARCH.dump(path, state)
            exports[name][str(task["seed"])] = dict(
                path=str(path), sha256=r.hashlib.sha256(path.read_bytes()).hexdigest()
            )
            model = r.Frozen(state, native=True)
            actual = np.array([model.predict_one(v) for v in x])
            error = float(np.max(np.abs(actual - expected[name])))
            assert error < 1e-4, (name, error)
            if task["seed"] != 73:
                continue
            calls = []
            with threadpool_limits(limits=1):
                for _ in range(5):
                    for row in x:
                        begin = time.perf_counter_ns()
                        model.predict_one(row)
                        calls.append((time.perf_counter_ns() - begin) / 1000)
            runtime[name] = dict(
                median_us=float(np.median(calls)),
                p95_us=float(np.quantile(calls, 0.95)),
                export_max_error_ms=error,
                feature_extraction_included=False,
            )
    r.e.ARCH.dump(r.ROOT / "exports.json", exports)
    r.e.ARCH.dump(r.ROOT / "runtime.json", runtime)
    cv = json.loads((r.ROOT / "validation_summary.json").read_text())
    final = json.loads((r.ROOT / "benchmark_summary.json").read_text())
    geometry = json.loads((r.ROOT / "geometry_summary.json").read_text())
    winner = json.loads((r.ROOT / "selection.json").read_text())["winner"]
    lines = [
        "# Training-support routing experiment",
        "",
        f"Validation-selected candidate: **{winner}**.",
        "",
        (
            "Route known aggregate compute geometry signatures to "
            "the fixed blend or sparse MLP, and unseen signatures to the "
            "suffix-guarded envelope. Signatures come exclusively from "
            "training. Observed signatures identify their "
            "sorted request compute geometry in this corpus; "
            "this does not prove uniqueness for arbitrary future shapes."
        ),
        "",
        (
            "| Model | Fresh CV MAE | CV macro MAE | Historical MAE | Reused "
            "replay MAE | Replay p95 |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in list(r.KNOWN) + ["envelope_sparse_mlp"]:
        v, h, f = (
            cv["episode"]["scores"]["fresh"][name],
            cv["episode"]["scores"]["historical"][name],
            final["overall"][name],
        )
        lines.append(
            f"| {name} | "
            f"{v['mae_ms']:.2f} | "
            f"{v['workload_macro_mae_ms']:.2f} | "
            f"{h['mae_ms']:.2f} | "
            f"{f['mae_ms']:.2f} | "
            f"{f['p95_absolute_ms']:.2f} |"
        )
    lines += [
        "",
        "Errors are milliseconds; figures average errors over three fitted seeds.",
        "",
        "## Generalization tradeoffs",
        "",
        "| Stress test | Unconditional blend | Selected router |",
        "|---|---:|---:|",
    ]
    for scope in ["long_context", "long_suffix", "large_batch", "heavy_transfers"]:
        s = cv[scope]["scores"]["fresh"]
        lines.append(
            f"| {scope} | "
            f"{s['envelope_sparse_mlp']['mae_ms']:.2f} | "
            f"{s[winner]['mae_ms']:.2f} |"
        )
    for mode in ["geometry_excluded", "count_matched"]:
        s = geometry[mode]["fresh"]
        lines.append(
            f"| {mode} | "
            f"{s['envelope_sparse_mlp']['mae_ms']:.2f} | "
            f"{s[winner]['mae_ms']:.2f} |"
        )
    lines += [
        "",
        (
            "The router improves episode validation and novel-geometry "
            "controls but can worsen transfer-range extrapolation. Its replay "
            "improvement is not established; the unconditional blend has "
            "slightly better replay MAE and p95. Both remain research "
            "candidates. No new GPU evaluation or serving integration was "
            "performed."
        ),
        "",
        (
            "Inference does not fit a model. Lookup cost is included in "
            "runtime.json; feature extraction is separate. Checkpoints, "
            "hashes, training support inventories, phase-specific scores, and "
            "paired validation bootstrap intervals are retained. The final "
            "replay set is reused, not an independent blind test, and did not "
            "select the router."
        ),
    ]
    (r.ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    base = p.git("rev-parse", "HEAD")
    recorded = {}
    for name in r.KNOWN:
        branch = f"experiment/glm53-support-{name.replace('_', '-')}-20261007"
        staging = r.ROOT / "branch_staging"
        p.git("worktree", "add", "-b", branch, str(staging), base)
        (staging / ".venv").symlink_to(r.SOURCE / ".venv", target_is_directory=True)
        manifest = (
            staging / "experiments/predictor_support_routing_20261007/variant.json"
        )
        r.e.ARCH.dump(
            manifest,
            dict(
                candidate=name,
                checkpoints=exports[name],
                implementation=base,
                no_inference_fitting=True,
                cpu_only=True,
                benchmark_is_reused=True,
                benchmark_used_for_selection=False,
                serving_integration_required=True,
            ),
        )
        p.git("add", str(manifest.relative_to(staging)), cwd=staging)
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
                f"Record support-routing predictor {name}",
                "-m",
                "Co-authored-by: Codex <noreply@openai.com>",
                "-m",
                "Signed-off-by: Codex <noreply@openai.com>",
            ],
            cwd=staging,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        recorded[name] = dict(
            branch=branch, commit=p.git("rev-parse", "HEAD", cwd=staging)
        )
        p.git("worktree", "remove", "--force", str(staging))
    r.e.ARCH.dump(r.ROOT / "branches.json", recorded)
    r.e.ARCH.dump(r.ROOT / "status.json", dict(phase="complete", finished=time.time()))
    print("support_publication_complete", flush=True)


if __name__ == "__main__":
    main()
