# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report a robustness-oriented alternative without replacing the strict winner."""

import json

import improve
import numpy as np
import report
import study


def main():
    summary = json.loads((study.ROOT / "improvement_summary.json").read_text())
    cv = summary["episode:wide"]["scores"]
    proposal = json.loads((study.ROOT / "improvement_plan.json").read_text())
    limit = cv["historical"]["trees_fixed"]["mae_ms"] * 1.1
    eligible = [
        n for n in proposal["candidates"] if cv["historical"][n]["mae_ms"] <= limit
    ]
    best = min(eligible, key=lambda n: cv["fresh"][n]["workload_macro_mae_ms"])
    tolerance = 0.1
    tied = [
        n
        for n in eligible
        if cv["fresh"][n]["workload_macro_mae_ms"]
        <= cv["fresh"][best]["workload_macro_mae_ms"] + tolerance
    ]
    scopes = ["long_context", "long_suffix", "large_batch", "heavy_transfers"]
    range_scores = {
        n: float(
            np.mean(
                [
                    summary[f"{s}:wide"]["scores"]["fresh"][n]["mae_ms"]
                    / summary[f"{s}:wide"]["scores"]["fresh"]["trees_fixed"]["mae_ms"]
                    for s in scopes
                ]
            )
        )
        for n in tied
    }
    chosen = min(tied, key=lambda n: (range_scores[n], cv["historical"][n]["mae_ms"]))
    tasks = [
        t
        for t in json.loads((study.ROOT / "tasks.json").read_text())
        if t["range_scope"] is None and t["regime"] == "wide"
    ]
    pieces = []
    for task in tasks:
        with np.load(
            study.ROOT / "improvements" / task["name"] / "predictions.npz"
        ) as data:
            pieces.append({k: data[k] for k in data.files})
    data = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    comparison = dict(data, trees_fixed=data[best])
    result = dict(
        improvement=chosen,
        effective_architecture=improve.architecture_description(
            chosen,
            proposal["architecture"],
        ),
        benchmark_used_for_selection=False,
        strict_selection_not_replaced=True,
        fresh_macro_tolerance_ms=tolerance,
        rationale=(
            "A 0.013ms fresh-macro difference separated a pure neural model from "
            "its blend, below marker uncertainty and bootstrap uncertainty. "
            "Within 0.1ms, prefer observed range robustness, then historical MAE."
        ),
        tied_candidates=tied,
        normalized_range_mae=range_scores,
        paired_vs_strict_best={
            n: dict(
                fresh_macro=report.paired_interval(
                    comparison,
                    n,
                    data["phase"] == "fresh",
                    macro=True,
                ),
                historical=report.paired_interval(
                    comparison,
                    n,
                    data["phase"] != "fresh",
                ),
            )
            for n in tied
        },
        serving_integration_required=True,
        inference_time_training=False,
    )
    study.arch.dump(study.ROOT / "robust_selection.json", result)
    print(
        json.dumps({k: result[k] for k in ["improvement", "tied_candidates"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
