# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remove optional loaded-server evaluations when the study time is exhausted."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path


def identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", type=int, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cutoff-utc", required=True)
    args = parser.parse_args()
    cutoff = datetime.fromisoformat(args.cutoff_utc.replace("Z", "+00:00")).timestamp()
    original = identity(args.owner)
    if original is None:
        raise ValueError("the experiment owner must exist before guarding its plan")
    while identity(args.owner) == original:
        if time.time() >= cutoff:
            temporary = args.plan.with_suffix(".guard.tmp")
            temporary.write_text("{}\n")
            os.replace(temporary, args.plan)
            args.plan.with_suffix(".guard.json").write_text(
                json.dumps(
                    dict(
                        event="optional_generalization_disabled_at_time_cutoff",
                        owner=args.owner,
                        cutoff=args.cutoff_utc,
                        time=time.time(),
                    ),
                    indent=2,
                )
                + "\n"
            )
            break
        time.sleep(2)


if __name__ == "__main__":
    main()
