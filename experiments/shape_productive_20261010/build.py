# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze a shape-based earlier load-only exit with unchanged finish correction."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--efficiency", type=float, choices=(0.7, 0.8), required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    fit = json.loads(args.fit.read_text())
    direct = min(
        [c for c in fit["candidates"] if c["recipe"] == "attention_direct"],
        key=lambda c: c["workload_macro_mae"],
    )
    coefficients = dict(zip(direct["features"], direct["coefficients"]))
    work = [
        coefficients[name]
        for name in ("suffix_sum_per_61440", "prefix_index_pairs_per_1e9")
    ]
    assert all(c > 0 for c in work)
    calibration = copy.deepcopy(fit["calibration"])
    assert calibration["schema"] == 3
    calibration.update(
        schema=5,
        productive_load_policy=dict(
            schema=1,
            features=direct["features"],
            coefficients=direct["coefficients"],
            work_coefficients=work,
            minimum_efficiency=args.efficiency,
            minimum_ready_candidates=1,
            provenance=dict(
                duration_fit_sha256=hashlib.sha256(args.fit.read_bytes()).hexdigest(),
                recipe=direct["recipe"],
                alpha=direct["alpha"],
                held_mae_ms=direct["held"]["mae"],
                work_units="Positive suffix and prefix-index terms of the "
                "direct fit; normalized work rate, not measured GPU duty cycle.",
            ),
        ),
    )
    result = dict(
        calibration=calibration,
        selected_direct=direct,
        caveat="Finish-correction weights unchanged. Earlier exit only "
        "with no compute, at least one all-rank-ready candidate, "
        "and supported causal shapes. Original half-cohort exit "
        "and safety cap remain. Independent serving gate pending.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                efficiency=args.efficiency,
                recipe=direct["recipe"],
                alpha=direct["alpha"],
                held=direct["held"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
