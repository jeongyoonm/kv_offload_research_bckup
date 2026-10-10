# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze uncertainty and setup allowances without modifying duration weights."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.shape_margin_20261010.audit import shape_bin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quantile", type=float, choices=(0.75, 0.9), required=True)
    parser.add_argument(
        "--setup-target",
        choices=("entry_to_forward_ms", "seal_to_forward_cpu_ms"),
        default="entry_to_forward_ms",
    )
    args = parser.parse_args()
    assert not args.output.exists()
    original = json.loads(args.fit.read_text())
    audit = json.loads(args.audit.read_text())
    calibration = copy.deepcopy(original["calibration"])
    assert calibration["schema"] == 3
    finish_rows, setup_rows = audit["finish_rows"], audit["setup_rows"]

    def allowances(rows, value):
        bins = {}
        for row in rows:
            key = shape_bin(row["requests"], row["max_suffix"])
            bins.setdefault(key, []).append(row[value])
        return {
            key: max(0.0, float(np.quantile(values, args.quantile)))
            for key, values in bins.items()
            if len(values) >= 20
        }

    margin = dict(
        finish_allowance_ms=allowances(finish_rows, "error_ms"),
        setup_allowance_ms=allowances(setup_rows, args.setup_target),
        finish_fallback_ms=max(
            0.0, float(np.quantile([r["error_ms"] for r in finish_rows], args.quantile))
        ),
        setup_fallback_ms=float(
            np.quantile([r[args.setup_target] for r in setup_rows], args.quantile)
        ),
        extra_ms=2,
        bounds_ms=[10, 100],
    )
    calibration.update(schema=4, deadline_margin=margin)
    calibration["provenance"].update(
        margin_audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        duration_fit_sha256=hashlib.sha256(args.fit.read_bytes()).hexdigest(),
        margin_quantile=args.quantile,
        margin_setup_target=args.setup_target,
        margin_minimum_bin_samples=20,
        margin_mode="Grouped finish errors plus observed successor host setup; "
        "no future membership, no online training.",
    )
    result = dict(
        calibration=calibration,
        source_fit=str(args.fit),
        source_audit=str(args.audit),
        quantile=args.quantile,
        caveat="Empirical development allowances; CPU gate and new "
        "serving seeds required. Frozen duration coefficients "
        "are unchanged. Sparse bins use pooled allowances.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(margin, indent=2))


if __name__ == "__main__":
    main()
