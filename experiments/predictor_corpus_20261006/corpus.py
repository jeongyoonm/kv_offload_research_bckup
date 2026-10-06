# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normalize historical GLM TP8 measurements without mixing target definitions."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOTS = [
    "glm53_adaptive_seal_20261002",
    "glm53_poll_prediction_20261003",
    "glm53_full_predicted_wait_20261004",
    "glm53_coalesced_notify_20261004",
    "glm53_cached_pinned_20261005",
    "glm53_cached_pinned_60k_regression_20261006",
    "glm53_cached_pinned_batched_20261006",
    "glm53_compact_prep_20261006",
    "glm53_seal_deadline_20261006",
    "glm53_seal_calibration_20261006",
]


def metrics(errors):
    import numpy as np

    errors = np.asarray(errors, dtype=float)
    return dict(
        count=len(errors),
        mae_ms=float(np.mean(np.abs(errors))),
        bias_ms=float(np.mean(errors)),
        stddev_ms=float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
        p95_absolute_ms=float(np.quantile(np.abs(errors), 0.95)),
        p95_overprediction_ms=float(np.quantile(np.maximum(errors, 0), 0.95)),
    )


def features(row):
    import numpy as np

    shapes = row["request_shapes"]
    n = len(shapes)
    suffix = [s[0] for s in shapes]
    context = [s[1] for s in shapes]
    tokens = sum(suffix)
    return np.array(
        [
            tokens,
            n,
            sum(t * (c + t / 2) for t, c in shapes),
            sum(context),
            min(suffix),
            max(suffix),
            min(context),
            max(context),
            sum((t + 127) // 128 * 128 for t in suffix),
            row["load_state"][0],
            row["load_state"][1],
        ],
        dtype=float,
    )


FEATURE_NAMES = [
    "suffix_tokens",
    "requests",
    "attention_pairs",
    "context_sum",
    "suffix_min",
    "suffix_max",
    "context_min",
    "context_max",
    "padded128",
    "outstanding_loads_at_dispatch",
    "outstanding_load_tokens_at_dispatch",
]


def extract(event):
    if event.get("event") != "step_completion" or not event.get("selected"):
        return None, "not_compute"
    if not event.get("valid"):
        return None, "invalid_completion"
    shapes = event.get("request_shapes", [])
    if not shapes or any(len(s) != 2 or s[0] <= 1 or s[1] <= 0 for s in shapes):
        return None, "not_cached_prefill"
    if sum(s[0] for s in shapes) != sum(event["selected"].values()):
        return None, "token_accounting"
    execution = event.get("execution_ranks") or []
    preparation = event.get("ranks") or []
    if len(execution) != 8 or len(preparation) != 8:
        return None, "not_eight_ranks"
    for r, p in zip(execution, preparation):
        if (
            not r
            or not p
            or len(r) != 9
            or len(p) != 9
            or r[0] != event["step"]
            or p[0] != event["step"]
            or r[5] != 1
            or p[7] != 1
            or r[6] <= 0
            or not 0 < p[1] <= p[2] <= r[1] <= r[3] < r[8]
        ):
            return None, "marker_semantics"
    loads = event.get("load_state")
    if not isinstance(loads, list) or len(loads) != 2 or min(loads) < 0:
        return None, "missing_dispatch_loads"
    return dict(
        step=event["step"],
        dispatch_ns=event["dispatch_ns"],
        request_shapes=sorted(shapes),
        load_state=loads,
        duration_ms=max(r[6] for r in execution) / 1e6,
        rank_spread_ms=(max(r[6] for r in execution) - min(r[6] for r in execution))
        / 1e6,
        preparation_ms=max(p[2] - p[1] for p in preparation) / 1e6,
        execution_ranks=execution,
    ), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, files, seen = [], [], set()
    rejected = Counter()
    for root_name in ROOTS:
        root = Path("/mnt/localnvme") / root_name
        for path in sorted(root.rglob("engine.jsonl")):
            result_path = path.parent / "result.json"
            if not result_path.exists():
                rejected["no_cell_result"] += 1
                continue
            result = json.loads(result_path.read_text())
            if (
                not result.get("accepted")
                or result.get("gpu_hits")
                or result.get("preemptions")
                or result.get("runtime_error_lines")
            ):
                rejected["rejected_cell"] += 1
                continue
            workload = next(
                (
                    part
                    for part in path.parts
                    if len(part.split("_")) == 3
                    and all(
                        value[1:].isdigit() and value[0] == prefix
                        for value, prefix in zip(part.split("_"), "pfh")
                    )
                ),
                None,
            )
            if not workload:
                rejected["unknown_workload"] += 1
                continue
            policy_path = path.parent / "policy_setting.json"
            policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
            file_rows = []
            raw = path.read_bytes()
            for line in raw.splitlines():
                event = json.loads(line)
                row, reason = extract(event)
                if row is None:
                    if event.get("event") == "step_completion":
                        rejected[reason] += 1
                    continue
                key = (row["dispatch_ns"], row["step"])
                if key in seen:
                    rejected["duplicate_record"] += 1
                    continue
                seen.add(key)
                row.update(
                    source=root_name,
                    episode=str(path.parent),
                    workload=workload,
                    policy=path.parent.name,
                    policy_options=policy,
                    source_file=str(path),
                )
                file_rows.append(row)
            rows.extend(file_rows)
            files.append(
                dict(
                    path=str(path),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    samples=len(file_rows),
                    accepted_cell=result,
                )
            )
    with (args.output / "corpus.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    summary = dict(
        target="slowest TP8 CUDA execution event span, after preparation",
        training_filter="all requests cached prefill; eight valid ranks",
        feature_availability="load_state captured at dispatch before FIFO execute",
        samples=len(rows),
        episodes=len({r["episode"] for r in rows}),
        workloads=dict(Counter(r["workload"] for r in rows)),
        sources=dict(Counter(r["source"] for r in rows)),
        rejected=dict(rejected),
        feature_names=FEATURE_NAMES,
        files=files,
    )
    (args.output / "inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
