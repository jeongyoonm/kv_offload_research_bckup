# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze joint support from original training geometry without refitting."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("training", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--relative-margin", type=float, default=0.05)
    args = parser.parse_args()
    assert not args.output.exists() and 0 <= args.relative_margin <= 0.25
    fit = json.loads(args.fit.read_text())
    data = args.training.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    state = fit["calibration"]
    assert state["schema"] == 1
    assert state["provenance"]["input_sha256"] == digest
    rows = json.loads(data)
    lower = min(r["features"][4] / r["features"][7] for r in rows)
    upper = max(r["features"][5] / r["features"][6] for r in rows)
    state["schema"] = 2
    state["geometry_support"] = dict(
        suffix_per_context=[
            lower * (1 - args.relative_margin),
            upper * (1 + args.relative_margin),
        ]
    )
    state["provenance"]["joint_guard"] = dict(
        training_sha256=digest,
        relative_margin=args.relative_margin,
        description="Conservative minimum-suffix/maximum-context and "
        "maximum-suffix/minimum-context bounds from original training. "
        "This is a narrow support guard, not a general computation law.",
    )
    fit["guard"] = dict(
        refit=False,
        original_fit_sha256=hashlib.sha256(args.fit.read_bytes()).hexdigest(),
        coefficients_unchanged=True,
        selection="5% margin fixed before measured crossed results; "
        "no development/test geometry enters support fitting",
    )
    args.output.write_text(json.dumps(fit, indent=2) + "\n")
    print(json.dumps(state["geometry_support"]))


if __name__ == "__main__":
    main()
