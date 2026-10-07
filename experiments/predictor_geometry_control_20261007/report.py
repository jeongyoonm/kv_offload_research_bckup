# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish paired geometry-coverage controls without selecting on replay."""

import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "geometry_study", Path(__file__).with_name("study.py")
)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
e = g.e


def arrays(tasks):
    pieces = []
    for task in tasks:
        with np.load(g.ROOT / "folds" / task["name"] / "predictions.npz") as f:
            a = {k: f[k] for k in f.files if k != "features"}
        with np.load(g.ROOT / "followups" / task["name"] / "predictions.npz") as f:
            np.testing.assert_array_equal(a["y"], f["y"])
            a.update({k: f[k].copy() for k in g.EXTRA})
        pieces.append(a)
    return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}


def main():
    tasks = json.loads((g.ROOT / "tasks.json").read_text())
    result, data = {}, {}
    names = e.ARCH.MODELS + g.EXTRA
    for mode in ["geometry_excluded", "count_matched"]:
        a = arrays([t for t in tasks if t["mode"] == mode])
        data[mode] = a
        result[mode] = {}
        for phase, mask in [
            ("overall", np.ones(len(a["y"]), dtype=bool)),
            ("fresh", a["phase"] == "fresh"),
            ("historical", a["phase"] != "fresh"),
        ]:
            rows = [
                dict(episode=x, workload=y)
                for x, y in zip(a["episode"][mask], a["workload"][mask])
            ]
            result[mode][phase] = {
                n: e.ARCH.metrics(a["y"][mask], a[n][mask], rows) for n in names
            }
    left, right = data["geometry_excluded"], data["count_matched"]
    np.testing.assert_array_equal(left["episode"], right["episode"])
    np.testing.assert_array_equal(left["y"], right["y"])
    # Episode bootstrap over all seeds together preserves pairing and does not
    # treat the three versions of each target as independent observations.
    paired = {}
    for n in names:
        mask = left["phase"] == "fresh"
        episode, inv = np.unique(left["episode"][mask], return_inverse=True)
        delta = np.abs(left[n][mask] - left["y"][mask]) - np.abs(
            right[n][mask] - right["y"][mask]
        )
        counts = np.bincount(inv)
        sums = np.bincount(inv, weights=delta)
        draw = np.random.default_rng(7307).integers(
            len(episode), size=(2000, len(episode))
        )
        values = sums[draw].sum(1) / counts[draw].sum(1)
        paired[n] = dict(
            exclusion_minus_control_mae_ms=float(delta.mean()),
            ci95_ms=np.quantile(values, [0.025, 0.975]).tolist(),
        )
    e.ARCH.dump(
        g.ROOT / "summary.json",
        dict(scores=result, paired=paired, cpu_only=True, benchmark_used=False),
    )
    lines = [
        "# Controlled geometry coverage study",
        "",
        (
            "Three outer episode folds and three model seeds. Each "
            "geometry-excluded training set contains no exact test batch "
            "geometry/dispatch-load-state combination. The control has exactly "
            "the same number of rows from each source and phase, sampled from "
            "the episode-purged training pool without using labels. Both base "
            "and correction use the same forecast rows (matched regime)."
        ),
        "",
        (
            "All nine original architectures are independently retuned on "
            "inner training episodes. Sparse MLP and envelope/blend follow-ups "
            "reuse these training-only capacities. This study uses no final "
            "replay labels and does not alter the frozen deployment candidate."
        ),
        "",
        (
            "| Model | Fresh exclusion MAE | Fresh count-matched MAE | "
            "Exclusion minus control |"
        ),
        "|---|---:|---:|---:|",
    ]
    for n in names:
        a, b = (
            result["geometry_excluded"]["fresh"][n],
            result["count_matched"]["fresh"][n],
        )
        lines.append(
            f"| {n} | "
            f"{a['mae_ms']:.2f} ms | "
            f"{b['mae_ms']:.2f} ms | "
            f"{paired[n]['exclusion_minus_control_mae_ms']:+.2f} ms |"
        )
    lines += [
        "",
        "## Limits",
        "",
        (
            "Exclusion is an aggressive geometry stress test: exact "
            "combinations are absent, although individual feature ranges may "
            "overlap. Removing familiar geometries changes the training "
            "distribution even after source/count matching. The control "
            "isolates that coverage change more closely than comparison with "
            "the larger original episode training pool; it does not make the "
            "two distributions identical."
        ),
        "",
        (
            "The test contains repeated runtime snapshots, and all snapshots "
            "from a held episode are purged from training. Bootstrap intervals "
            "group by episode and keep all three seeds paired. Accuracy "
            "averages errors over seeds, without a prediction ensemble. No "
            "throughput/TTFT claim follows from these CPU measurements."
        ),
        "",
        (
            "Training-row counts, source categories, geometry overlap and "
            "episode disjointness are in split_audit.json. Checkpoints, tuning "
            "trials, predictions and completion records are retained for every "
            "cell under folds/ and followups/."
        ),
    ]
    (g.ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    for n in ["trees_fixed", "mlp_direct", "envelope_sparse_mlp"]:
        print(
            n,
            "excluded",
            round(result["geometry_excluded"]["fresh"][n]["mae_ms"], 3),
            "control",
            round(result["count_matched"]["fresh"][n]["mae_ms"], 3),
            flush=True,
        )


if __name__ == "__main__":
    main()
