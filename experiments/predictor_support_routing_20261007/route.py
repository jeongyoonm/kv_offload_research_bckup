# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route familiar signatures to neural experts, novel ones to envelope."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "experiments/predictor_launch_envelope_20261007"))
e = importlib.import_module("envelope")
b = importlib.import_module("envelope_blend")
g = importlib.import_module("envelope_guard")
ev = importlib.import_module("envelope_evaluate")
report = importlib.import_module("report")
ROOT = Path("/mnt/localnvme/glm53_compute_support_20261007")
GEOMETRY = Path("/mnt/localnvme/glm53_geometry_control_20261007")
INDICES = list(range(9)) + [23, 24, 25]
KNOWN = {"compute_known_blend_else_envelope": "envelope_sparse_mlp"}


def signatures(rows):
    return {tuple(v) for v in e.features(rows)[:, INDICES]}


class Frozen:
    def __init__(self, state, native=False):
        self.indices = state.get("feature_indices", INDICES)
        self.known = {tuple(v) for v in state["signatures"]}
        self.envelope = g.Frozen(state["envelope"], native)
        self.expert = (
            b.Frozen(state["expert"], native)
            if state["known_model"] == "envelope_sparse_mlp"
            else e.improve.FrozenCandidate(state["expert"], native)
        )
        self.known_model = state["known_model"]

    def predict_one(self, x):
        if tuple(x[self.indices]) not in self.known:
            return self.envelope.predict_one(x)
        if self.known_model == "envelope_sparse_mlp":
            return self.expert.predict_one(x)
        return self.expert.predict_one(np.r_[x[:23], np.zeros(5), x[23:26]])


def apply(tasks, geometry=False):
    output = []
    pools = {}
    for task in tasks:
        path = task["dataset"]
        if path not in pools:
            pools[path] = joblib.load(path)
        pool = pools[path]
        train, test = (
            [pool["forecasts"][i] for i in task[k]] for k in ["train", "test"]
        )
        assert not {r["episode"] for r in train} & {r["episode"] for r in test}
        support = signatures(train)
        x = e.features(test)
        known = np.array([tuple(v) in support for v in x[:, INDICES]])
        if geometry:
            with np.load(
                GEOMETRY / "followups" / task["name"] / "predictions.npz"
            ) as f:
                data = {k: f[k] for k in f.files}
            with np.load(GEOMETRY / "folds" / task["name"] / "predictions.npz") as f:
                data["trees_fixed"] = f["trees_fixed"].copy()
        else:
            data = b.arrays([task])
        for name, source in KNOWN.items():
            data[name] = np.where(
                known, data[source], data["suffix_guard_envelope_boost"]
            )
        directory = ROOT / ("geometry" if geometry else "folds") / task["name"]
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "predictions.npz", **data)
        e.ARCH.dump(
            directory / "support.json",
            dict(
                signatures=[list(v) for v in sorted(support)],
                known_model_indices=INDICES,
                known_test_rows=int(known.sum()),
                test_rows=len(test),
                training_only=True,
            ),
        )
        output.append(data)
    return {k: np.concatenate([p[k] for p in output]) for k in output[0]}


def scores(data):
    result = {}
    names = (
        list(KNOWN)
        + list(KNOWN.values())
        + ["trees_fixed", "suffix_guard_envelope_boost"]
    )
    for phase, mask in [
        ("overall", np.ones(len(data["y"]), dtype=bool)),
        ("fresh", data["phase"] == "fresh"),
        ("historical", data["phase"] != "fresh"),
    ]:
        rows = [
            dict(episode=a, workload=c)
            for a, c in zip(data["episode"][mask], data["workload"][mask])
        ]
        result[phase] = (
            {n: e.ARCH.metrics(data["y"][mask], data[n][mask], rows) for n in names}
            if rows
            else {}
        )
    return result


def summarize(tasks):
    result = {}
    for scope in sorted({t["range_scope"] or "episode" for t in tasks}):
        members = [
            t
            for t in tasks
            if (t["range_scope"] or "episode") == scope and t["regime"] == "wide"
        ]
        data = apply(members)
        result[scope] = dict(
            scores=scores(data),
            paired_vs_blend={
                n: report.paired_interval(
                    dict(data, trees_fixed=data["envelope_sparse_mlp"]),
                    n,
                    data["phase"] == "fresh",
                    macro=True,
                )
                for n in KNOWN
            },
        )
    cv = result["episode"]["scores"]
    names = list(KNOWN) + ["envelope_sparse_mlp"]
    eligible = [
        n
        for n in names
        if cv["historical"][n]["mae_ms"]
        <= cv["historical"]["trees_fixed"]["mae_ms"] * 1.1
    ]
    winner = min(eligible, key=lambda n: cv["fresh"][n]["workload_macro_mae_ms"])
    e.ARCH.dump(ROOT / "validation_summary.json", result)
    e.ARCH.dump(
        ROOT / "selection.json",
        dict(
            winner=winner,
            criterion=(
                "fresh episode macro MAE; <=10% historical regression; "
                "existing blend remains eligible"
            ),
            benchmark_used_for_selection=False,
            benchmark_is_reused=True,
        ),
    )
    for n in names:
        print(
            n,
            "fresh",
            round(cv["fresh"][n]["mae_ms"], 3),
            "macro",
            round(cv["fresh"][n]["workload_macro_mae_ms"], 3),
            "historical",
            round(cv["historical"][n]["mae_ms"], 3),
            flush=True,
        )
    print("selected", winner, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--geometry", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.benchmark:
        checksum = hashlib.sha256((ROOT / "selection.json").read_bytes()).hexdigest()
        tasks = json.loads((e.DATA / "benchmark_tasks.json").read_text())
        data = apply([t for t in tasks if t["regime"] == "wide"])
        e.ARCH.dump(ROOT / "benchmark_summary.json", scores(data))
        assert (
            hashlib.sha256((ROOT / "selection.json").read_bytes()).hexdigest()
            == checksum
        )
        e.ARCH.dump(
            ROOT / "benchmark_opened.json",
            dict(selection_sha256=checksum, benchmark_is_reused=True),
        )
    elif args.geometry:
        tasks = json.loads((GEOMETRY / "tasks.json").read_text())
        result = {
            mode: scores(apply([t for t in tasks if t["mode"] == mode], geometry=True))
            for mode in ["geometry_excluded", "count_matched"]
        }
        e.ARCH.dump(ROOT / "geometry_summary.json", result)
    else:
        summarize(json.loads((e.DATA / "tasks.json").read_text()))
