# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combine independently frozen margin and productive-load rules without fitting."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("margin", type=Path)
    parser.add_argument("productive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    sources = [
        json.loads(p.read_text())["calibration"] for p in (args.margin, args.productive)
    ]

    def core(state):
        excluded = {"schema", "provenance", "deadline_margin", "productive_load_policy"}
        return {key: value for key, value in state.items() if key not in excluded}

    assert core(sources[0]) == core(sources[1])
    calibration = copy.deepcopy(sources[1])
    calibration.update(schema=6, deadline_margin=sources[0]["deadline_margin"])
    provenance = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (args.margin, args.productive)
    }
    result = dict(
        calibration=calibration,
        source_sha256=provenance,
        duration_core_identical=True,
        reason="Fresh-seed interaction test: shape margin can change "
        "pipeline cadence; productive early exit may avoid prolonged "
        "load-only half-target waits. No new fitting or changed duration weights.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                duration_core_identical=True,
            )
        )
    )


if __name__ == "__main__":
    main()
