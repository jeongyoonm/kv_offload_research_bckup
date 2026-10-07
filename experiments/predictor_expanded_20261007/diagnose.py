# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Describe training coverage and source differences without benchmark labels."""

from collections import defaultdict

import joblib
import numpy as np
import study


def main():
    pool = joblib.load(study.ROOT / "pool.joblib")
    inventory = {}
    for kind, rows in pool.items():
        for source in ["historical", "fresh"]:
            selected = [r for r in rows if r["_source"] == source]
            x = study.arch.features(selected)
            inventory[f"{kind}:{source}"] = dict(
                samples=len(selected),
                episodes=len({r["episode"] for r in selected}),
                feature_min=x.min(0).tolist(),
                feature_max=x.max(0).tolist(),
                duration_quantiles_ms=np.quantile(
                    study.arch.target(selected), [0, 0.1, 0.5, 0.9, 1]
                ).tolist(),
            )
    groups = defaultdict(list)
    for row in pool["batches"]:
        key = (
            tuple(sorted(map(tuple, row["request_shapes"]))),
            tuple(row["load_state"]),
        )
        groups[key].append(row)
    differences = []
    for (shapes, loads), rows in groups.items():
        source_rows = {
            name: [r for r in rows if r["_source"] == name]
            for name in ["historical", "fresh"]
        }
        if not all(len({r["episode"] for r in rs}) >= 3 for rs in source_rows.values()):
            continue
        medians = {
            name: float(np.median(study.arch.target(rs)))
            for name, rs in source_rows.items()
        }
        differences.append(
            dict(
                shapes=shapes,
                dispatch_loads=loads,
                samples={n: len(rs) for n, rs in source_rows.items()},
                median_ms=medians,
                fresh_minus_historical_ms=medians["fresh"] - medians["historical"],
            )
        )
    study.arch.dump(
        study.ROOT / "coverage_diagnostics.json",
        dict(
            inventory=inventory,
            feature_names=study.arch.surface.FEATURE_NAMES,
            source_differences=sorted(
                differences, key=lambda r: -abs(r["fresh_minus_historical_ms"])
            ),
            marker_uncertainty={
                source: dict(
                    start_bound_width_p50_p95_max_ms=np.quantile(
                        [
                            max((z[3] - z[2]) / 1e6 for z in r["execution_ranks"])
                            for r in pool["forecasts"]
                            if r["_source"] == source
                        ],
                        [0.5, 0.95, 1],
                    ).tolist(),
                    lower_bound_lead_ge50ms=sum(
                        np.median([(z[2] - z[1]) / 1e6 for z in r["execution_ranks"]])
                        >= 50
                        for r in pool["forecasts"]
                        if r["_source"] == source
                    ).item(),
                )
                for source in ["historical", "fresh"]
            },
            benchmark_used=False,
            caveat=(
                "Matching unordered geometry and dispatch loads does not control "
                "transfer progress, token content, thermal state or launch "
                "configuration. "
                "Source differences cannot be interpreted as causal effects."
            ),
        ),
    )
    print(
        dict(source_comparable_groups=len(differences), inventory=inventory), flush=True
    )


if __name__ == "__main__":
    main()
