# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze a direct duration fit in the existing shape-calibration transport."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    fit = json.loads(args.fit.read_text())
    direct = min(
        [c for c in fit["candidates"] if c["recipe"] == "attention_direct"],
        key=lambda c: c["workload_macro_mae"],
    )
    calibration = copy.deepcopy(fit["calibration"])
    assert calibration["schema"] == 3
    # raw_ms - (raw_ms - direct_ms) leaves only the direct duration estimate.
    calibration["features"] = direct["features"] + ["raw_duration_per_1000_ms"]
    calibration["coefficients"] = [-c for c in direct["coefficients"]] + [1000]
    calibration["provenance"].update(
        direct_fit_sha256=hashlib.sha256(args.fit.read_bytes()).hexdigest(),
        recipe=direct["recipe"],
        alpha=direct["alpha"],
        direct_mode="Within support and correction bounds, predicted duration "
        "equals the direct shape fit. Keep the old forest evaluation and "
        "original unsupported fallback unchanged.",
    )
    args.output.write_text(
        json.dumps(
            dict(
                calibration=calibration,
                selected_direct=direct,
                caveat="The existing correction clamp remains active. Record "
                "clipping explicitly before interpreting raw independence. "
                "No new labels or fitting; runtime and serving gates pending.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(recipe=direct["recipe"], held=direct["held"])))


if __name__ == "__main__":
    main()
