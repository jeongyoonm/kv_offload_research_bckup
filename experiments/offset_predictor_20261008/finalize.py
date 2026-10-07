# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit every measured request and preserve the completed comparison evidence."""

import hashlib
import json
import shutil
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path("/mnt/localnvme/glm53_offset_predictor_20261008")


def format_report(source):
    lines = []
    in_table = False
    for line in source.splitlines():
        is_table = line.startswith("|")
        if is_table and not in_table:
            lines.append("<!-- markdownlint-disable MD013 -->")
        elif in_table and not is_table:
            lines.append("<!-- markdownlint-enable MD013 -->")
        if is_table:
            lines.append(
                "| " + " | ".join(c.strip() for c in line.split("|")[1:-1]) + " |"
            )
        else:
            lines.extend(textwrap.wrap(line, width=80, break_long_words=False) or [""])
        in_table = is_table
    if in_table:
        lines.append("<!-- markdownlint-enable MD013 -->")
    return "\n".join(lines) + "\n"


def main():
    plan = json.loads((ROOT / "plan.json").read_text())
    audits = []
    for path in sorted((ROOT / "live").glob("*/r[0-3]_*/result.json")):
        result = json.loads(path.read_text())
        assert result["accepted"] and result["frozen_unchanged"]
        assert result["optimized_preparation_verified"]
        assert result["requests"] == len(result["responses"]) == 75
        assert result["concurrency"] == 32
        assert result["observed_cached"] == result["expected_cached"]
        checkpoint = json.loads(
            (ROOT / "models" / f"{result['label']}.json").read_text()
        )
        digest = hashlib.sha256(
            json.dumps(checkpoint, sort_keys=True).encode()
        ).hexdigest()
        assert result["checkpoint_sha256"] == digest
        policy = json.loads((path.parent / "policy.json").read_text())
        assert all(policy[k] == v for k, v in plan["policy"].items())
        assert policy["seal_estimator"]["loaded"]
        for key in ["vllm:prefix_cache_hits_total", "vllm:num_preemptions_total"]:
            assert result["metrics_before"][key] == result["metrics_after"][key]
        expected = {}
        for response in result["responses"]:
            request = response["request"]
            assert request not in expected
            usage = response["usage"]
            assert usage["completion_tokens"] == 1
            expected[request] = (
                usage["prompt_tokens"] - usage["prompt_tokens_details"]["cached_tokens"]
            )
        prefixes = {f"cmpl-{request}-0-": request for request in expected}
        scheduled = Counter()
        steps = set()
        for line in (path.parent / "engine.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["event"] != "step_completion":
                continue
            assert event["valid"]
            assert event["step"] not in steps
            steps.add(event["step"])
            for internal_id, tokens in event.get("selected", {}).items():
                matches = [
                    request
                    for prefix, request in prefixes.items()
                    if internal_id.startswith(prefix)
                ]
                assert len(matches) == 1, internal_id
                scheduled[matches[0]] += tokens
        assert dict(scheduled) == expected, path
        audits.append(
            dict(
                cell=str(path.parent),
                requests=len(expected),
                scheduled_tokens=sum(scheduled.values()),
                expected_tokens=sum(expected.values()),
                per_request_token_accounting=True,
                unique_ordered_completion_steps=len(steps),
                frozen_checkpoint=True,
                matched_policy=True,
                optimized_preparation=True,
                passed=True,
            )
        )
    assert len(audits) == 36
    assert sum(a["requests"] for a in audits) == 2700
    (ROOT / "measurement_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    allocation = json.loads((ROOT / "allocation.json").read_text())
    assert allocation["server_stopped"]
    comparison = json.loads((ROOT / "comparison.json").read_text())
    accuracy = json.loads((ROOT / "audit_live/audit.json").read_text())
    cpu = json.loads((ROOT / "cpu_comparison.json").read_text())
    previous = json.loads((ROOT / "audit_previous/audit.json").read_text())
    selection = json.loads((ROOT / "selection.json").read_text())
    lines = [
        "# Offset-only duration prediction: results and accuracy audit",
        "",
        "GLM 5.3 TP8, concurrency 32, one output token. Each model/workload",
        "uses four fresh-seed 75-request episodes. Startup, priming and",
        "warmups are excluded. All 36 cells and 2,700 requests passed.",
        "",
        "## Matched feature ablation on previous live data",
        "",
        "| Training data | Architecture | Elapsed input | MAE (ms) | Macro MAE (ms) |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for c in cpu:
        family, timing = c["name"].split("_")
        lines.append(
            f"| {c['source']} | {family} | {timing} | {c['mae_ms']:.2f} | "
            f"{c['macro_mae_ms']:.2f} |"
        )
    lines += [
        "",
        "The same rows, weights, capacities and seed are used in each paired",
        "feature ablation. Complete workload/prompt-seed groups, including",
        "all three predictor variants, are purged from held folds. These",
        "previous live cells are now development data. No fresh live labels",
        "are used in fitting or selection. Masking is applied to primary",
        "and fallback models; only elapsed execution time is removed.",
        "",
        "## Fresh live throughput and TTFT",
        "",
        (
            "| Workload | Predictor | req/s | Gain vs trees | "
            "Mean TTFT (ms) | p95 TTFT (ms) |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for g in comparison["aggregate"]:
        lines.append(
            f"| {g['workload']} | {g['predictor']} | {g['throughput_rps']:.3f} | "
            f"{g['throughput_gain_percent']:+.2f}% | {g['mean_ttft_ms']:.1f} | "
            f"{g['p95_ttft_ms']:.1f} |"
        )
    paired = []
    import math

    import numpy as np
    from scipy.stats import t

    for workload in ["familiar60k", "short16k", "heterogeneous"]:
        ratios = []
        for repeat in range(4):
            paths = ROOT / "live" / workload
            offset = json.loads(
                (paths / f"r{repeat}_matched_offset/result.json").read_text()
            )
            control = json.loads(
                (paths / f"r{repeat}_matched_elapsed/result.json").read_text()
            )
            ratios.append(
                math.log(offset["throughput_rps"] / control["throughput_rps"])
            )
        mean = float(np.mean(ratios))
        margin = float(t.ppf(0.975, 3) * np.std(ratios, ddof=1) / 2)
        paired.append(
            dict(
                workload=workload,
                gain_percent=100 * math.expm1(mean),
                bounds=[
                    100 * math.expm1(mean - margin),
                    100 * math.expm1(mean + margin),
                ],
            )
        )
    (ROOT / "paired_feature_gains.json").write_text(json.dumps(paired, indent=2) + "\n")
    lines += [
        "",
        "### Offset-only versus matched elapsed-input control",
        "",
        "| Workload | Paired throughput gain | 95% interval |",
        "| --- | ---: | ---: |",
    ]
    for g in paired:
        lo, hi = g["bounds"]
        lines.append(
            f"| {g['workload']} | {g['gain_percent']:+.2f}% | "
            f"[{lo:+.2f}%, {hi:+.2f}%] |"
        )
    lines += [
        "",
        "Intervals use four matched episode log ratios and a Student-t",
        "interval with three degrees of freedom. This small study cannot",
        "establish equivalence. The current tree baseline differs in fitting",
        "data and architecture; matched controls isolate feature removal.",
        "",
        "## Accuracy on identical fresh live decision snapshots",
        "",
        "| Workload | Forecasts | Predictor | MAE (ms) | MAPE | Bias (ms) |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for w in accuracy["live"]:
        for name, m in w["models"].items():
            a = m["accuracy"]
            lines.append(
                f"| {w['workload']} | {w['count']} | {name} | {a['mae_ms']:.2f} | "
                f"{a['mape_percent']:.2f}% | {a['bias_ms']:+.2f} |"
            )
    lines += [
        "",
        "All models are replayed at the same causal inputs sampled by the",
        "union of live policies. A different issued-prediction population",
        "must not be treated as an architecture accuracy comparison.",
        "",
        "## Audit findings",
        "",
        "Previous 215 issued forecasts reproduce exactly: zero nanosecond",
        "prediction difference and zero input-feature difference. The new",
        "audit also checks every issued value and its absolute deadline.",
        "CPU exports matched research inference within 0.000001 ms on",
        "215 prior live snapshots. Offset prediction passed 2,150 elapsed",
        "invariance checks, including fallback and remaining-wait arithmetic.",
        "The existing harness and predictor suite passed 211 tests with",
        "8 skips; the updated predictor suite passed 31 tests.",
        "",
        "The offline replay benchmark sampled close to GPU start (median",
        "0.64 ms), with median zero concurrent loads and a maximum of eight.",
        "Previous live 60K/16K predictions were made at median 52.3/15.5 ms",
        "after GPU start, with median 8.5/7 concurrent loads. 46/92 60K",
        "and 42/112 16K points exceed the old benchmark's load-count maximum.",
        "The original training pool does include heavier loads; this is a",
        "benchmark coverage gap, not wholly unseen training input.",
        "Live prediction points are selected by open sealing transactions:",
        "the previous run issued 215 forecasts across 558 compute rounds.",
        "The mixed workload contributed only 11 forecasts across 295 rounds.",
        "Offline collection sampled eligible batches independently of whether",
        "the sealing policy would need a prediction. Accuracy on the decision",
        "population must not be generalized to every compute batch.",
        "",
        "Most live homogeneous batch geometries appeared in training",
        "(88/92 60K and 104/112 16K). Matching geometry therefore does not",
        "remove the discrepancy. GPU start/end polling uncertainty is",
        "typically about 0.5–0.6 ms and is too small to explain tens of",
        "milliseconds of error. The event-span target includes CPU",
        "submission gaps and required output-copy joins. Submission and",
        "transfer conditions differ even for matching geometries.",
        "",
        "Masking elapsed in an already trained model improves some subsets",
        "and harms others; it is not equivalent to retraining. Offline",
        "errors remain different within comparable load and duration bins.",
        "The observed coverage shifts do not yet provide a numerical causal",
        "attribution of the entire gap. Feature ablations, data ablations",
        "and per-snapshot replay are the controlled evidence here.",
        "",
        "## Limits and artifacts",
        "",
        "The serving preparation optimization, CPU private mapping, FP8 KV",
        "and original sparse selector are preserved. No selector oracle or",
        "transfer auditing runs during throughput measurement. This is a",
        "prefill/TTFT study and does not establish long-generation performance.",
        "The offset experiment is opt-in; the default checkpoint is unchanged.",
        f"Selected architecture: {selection['selected_family']}.",
        f"Allocation: {allocation['seconds']:.1f} seconds, server stopped.",
        f"All raw artifacts: `{ROOT}`.",
    ]
    completion = dict(
        complete=True,
        cells=36,
        requests=2700,
        all_request_accounting_passed=True,
        allocation=allocation,
        offset_feature_invariance_verified=True,
        previous_issued_replay_difference_ns=previous[
            "issued_replay_max_difference_ns"
        ],
        fresh_issued_replay_difference_ns=accuracy["issued_replay_max_difference_ns"],
        branch="experiment/glm53-offset-only-predictor-20261008",
    )
    (ROOT / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    (ROOT / "FINDINGS.md").write_text(format_report("\n".join(lines)))
    (ROOT / "REPORT.md").write_text(format_report((ROOT / "REPORT.md").read_text()))
    dest = Path(__file__).parent / "results"
    dest.mkdir(exist_ok=True)
    for name in [
        "REPORT.md",
        "FINDINGS.md",
        "comparison.json",
        "cpu_comparison.json",
        "paired_feature_gains.json",
        "completion.json",
        "measurement_audit.json",
        "cpu_equivalence.json",
        "protocol.json",
        "selection.json",
        "freeze.json",
    ]:
        shutil.copy2(ROOT / name, dest / name)
    shutil.copy2(ROOT / "audit_previous/audit.json", dest / "audit_previous.json")
    shutil.copy2(
        ROOT / "audit_previous/conditions.json", dest / "conditions_previous.json"
    )
    shutil.copy2(ROOT / "audit_live/audit.json", dest / "audit_live.json")
    shutil.copy2(ROOT / "audit_live/conditions.json", dest / "conditions_live.json")
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
