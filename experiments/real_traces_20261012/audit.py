# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed-cell output, accounting, causality and predictor-provenance audit."""

import argparse
import json
from pathlib import Path

from experiments.real_traces_20261012.replay import CAMPAIGN, STATE_SHA256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", type=Path, nargs="+", required=True)
    args = p.parse_args()
    results = []
    for root in args.roots:
        for path in (root / "cells").glob("*/result.json"):
            r = json.loads(path.read_text())
            responses = [
                json.loads(x)
                for x in (path.parent / "responses.jsonl").read_text().splitlines()
            ]
            meta = json.loads((path.parent / "spec.json").read_text())
            checks = dict(
                accepted=r["accepted"],
                all_responses_valid=all(x["accepted"] for x in responses),
                response_count=len(responses) == r["expected_requests"],
                unique_request_ids=len({x["request"] for x in responses})
                == len(responses),
                context_bounds=all(
                    x["input_tokens"] + x["target_output_tokens"] <= 81920
                    for x in responses
                ),
                exact_input_counts=all(
                    x["input_tokens"] == x["usage"]["prompt_tokens"] for x in responses
                ),
                exact_output_counts=all(
                    x["target_output_tokens"] == x["usage"]["completion_tokens"]
                    for x in responses
                ),
                checkpoint=meta["predictor_sha256"] == STATE_SHA256,
                all_load_ranks=all(int(k) == 8 for k in r["h2d_rank_coverage"]),
                all_store_ranks=all(int(k) == 8 for k in r["d2h_rank_coverage"]),
            )
            metrics = r["metric_deltas"]
            for key, total in [
                ("vllm:request_success_total", len(responses)),
                ("vllm:request_prompt_tokens_sum", r["input_tokens"]),
                ("vllm:request_generation_tokens_sum", r["output_tokens"]),
            ]:
                checks[key] = metrics.get(key) == total
            if r["arm"] == "full":
                checks["frozen_predictor"] = r["predictor_unchanged"]
            if r["spec"]["dataset"] == "inferact":
                sessions = {x["session"] for x in responses}
                causal = True
                for sid in sessions:
                    rows = sorted(
                        (x for x in responses if x["session"] == sid),
                        key=lambda x: x["index"],
                    )
                    causal &= [x["index"] for x in rows] == list(range(len(rows)))
                    causal &= all(
                        b["scheduled_ns"] >= a["end_ns"] for a, b in zip(rows, rows[1:])
                    )
                checks["causal_session_order"] = causal
            results.append(
                dict(path=str(path), passed=all(checks.values()), checks=checks)
            )
    output = dict(
        cells=len(results), passed=all(r["passed"] for r in results), results=results
    )
    (CAMPAIGN / "closed_cell_audit.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(dict(cells=len(results), passed=output["passed"])))
    assert output["passed"], [r for r in results if not r["passed"]]


if __name__ == "__main__":
    main()
