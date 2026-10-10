# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check direct-fit transport algebra without importing the serving runtime."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    fit = json.loads(args.fit.read_text())
    calibration = fit["calibration"]
    direct = fit["selected_direct"]
    rows = []
    for source in calibration["provenance"]["sources"]:
        data = Path(source["path"]).read_bytes()
        data = gzip.decompress(data) if source["path"].endswith(".gz") else data
        rows.extend(r for r in json.loads(data) if r.get("adjacent_compute", True))
    differences, clipped = [], 0
    for row in rows:
        x = row["features"]
        q = np.array(
            [1, x[1] / 16, x[0] / 61440, x[3] / x[1] / 60000, x[5] / 4096, x[23] / 1e9]
        )
        duration = float(q @ direct["coefficients"])
        bias = float(np.append(q, row["raw_ms"] / 1000) @ calibration["coefficients"])
        bounded = np.clip(bias, *calibration["bias_bounds_ms"])
        clipped += bounded != bias
        if bounded == bias:
            differences.append(abs(row["raw_ms"] - bounded - duration))
    maximum = max(differences)
    assert maximum < 1e-9
    result = dict(
        rows=len(rows),
        clipped=int(clipped),
        maximum_algebra_error_ms=maximum,
        held_seed_mae_ms=direct["held"]["mae"],
        caveat="Transport algebra and development data only; this is not a "
        "held-out serving result. Correction clipping preserves legacy bounds.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
