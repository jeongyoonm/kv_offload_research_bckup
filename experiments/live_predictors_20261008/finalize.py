# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit every measured request and preserve the completed comparison evidence."""

import hashlib
import json
import shutil
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path("/mnt/localnvme/glm53_live_predictors_20261008")


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
    completion = dict(
        complete=True,
        measured_cells=36,
        measured_requests=2700,
        cpu_tests="211 passed, 8 skipped",
        cpu_export_equivalence_rows_per_model=5186,
        request_accounting_audit_passed=True,
        allocation=allocation,
        gpu_cleanup="Confirmed all eight GPUs at 0 MiB after driver exit code 0",
        branch="experiment/glm53-live-predictor-comparison-20261008",
        artifacts=str(ROOT),
        protocol="Four matched 75-request episodes, not one continuous stream",
    )
    (ROOT / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    comparison = json.loads((ROOT / "comparison.json").read_text())
    accuracy = json.loads((ROOT / "common_accuracy.json").read_text())
    lines = [
        "# Findings and verification",
        "",
        "All 36 measured cells passed: 2,700 requests, 300 per predictor/workload.",
        "Every request's scheduled token total equals its uncached prompt length.",
        "All cells preserve the frozen checkpoint and matched sealing policy,",
        "use optimized preparation on all ranks, and have no GPU prefix hits or",
        "preemptions. CPU checks passed (211 tests; 8 skipped), and the two new",
        "serving exports match research inference on 5,186 forecasts each",
        "within 0.000001 ms. The owned server stopped; all eight GPUs are free.",
        "",
        "## Throughput uncertainty",
        "",
        "| Workload | Predictor | Paired geometric gain | Paired 95% interval |",
        "|---|---|---:|---:|",
    ]
    for group in comparison["aggregate"]:
        if group["predictor"] == "trees":
            continue
        low, high = group["paired_95pct_gain_bounds"]
        lines.append(
            f"| {group['workload']} | {group['predictor']} | "
            f"{group['paired_throughput_gain_percent']:+.2f}% | "
            f"[{low:+.2f}%, {high:+.2f}%] |"
        )
    lines.extend(
        [
            "",
            "Intervals use four matched episode log throughput ratios and a",
            "Student-t interval (3 degrees of freedom). Every interval includes",
            "zero; these cells do not establish a throughput improvement or",
            "equivalence. Checkpoint coverage differs as well as architecture.",
            "",
            "## Accuracy on identical live snapshots",
            "",
            "| Workload | Predictor | Forecasts | MAE (ms) | MAPE | Bias (ms) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for metric in accuracy["metrics"]:
        lines.append(
            f"| {metric['workload']} | {metric['predictor']} | "
            f"{metric['count']} | {metric['mae_ms']:.2f} | "
            f"{metric['mape_percent']:.2f}% | {metric['bias_ms']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "Positive bias means predicting completion too late. Replay uses",
            "the same 215 causal snapshots for all models; these are sampled by",
            "the union of live policies, rather than an independent held-out",
            "workload distribution. The mixed workload has only 11 forecasts.",
            "The offline 4–5 ms error did not carry over to this live workload.",
            "The MLP modestly improves 60K MAE, but has worse short/mixed MAE.",
            "The blend is close to trees and has a positive completion bias.",
            "",
            "## Interpretation",
            "",
            "Keep the serving tree as the default. Neither new checkpoint",
            "improves accuracy and throughput consistently across workloads.",
            "For 60K, the MLP's aggregate +2.6% throughput accompanies a reduction",
            "in summed GPU gaps from 1,135 to 464 ms per episode, while compute",
            "rounds increase from 42 to 50 across the four episodes. The blend's",
            "-3.0% accompanies larger gaps (1,566 ms). For 16K, MLP gaps increase",
            "from 89 to 298 ms and aggregate throughput falls 2.0%. These",
            "associations do not isolate prediction error as the causal driver:",
            "changed sealing alters batches and CPU submission timing as well.",
            "Gap totals include load-only intervals, not just post-seal work.",
            "",
            "The next targeted study should collect forecasts at the live",
            "scheduler's actual decision times and split evaluation by complete",
            "episodes before fitting. Existing data show forecast timing and",
            "queue-state differences from offline collection; they do not yet",
            "prove a single root cause. Avoid selecting or retuning on these",
            "evaluation labels and then reporting the same cells as validation.",
            "",
            f"Raw traces, checkpoints and launch metadata: `{ROOT}`.",
            "The benchmark is prefill/TTFT focused: one output token, concurrency",
            "32. It does not establish long-generation throughput or model",
            "output equivalence under the original nondeterministic selector.",
        ]
    )
    (ROOT / "FINDINGS.md").write_text(format_report("\n".join(lines)))
    report_path = ROOT / "REPORT.md"
    # Repeated finalization must not accumulate formatting directives.
    report = "\n".join(
        line
        for line in report_path.read_text().splitlines()
        if not line.startswith("<!-- markdownlint-")
    )
    report_path.write_text(format_report(report))
    destination = Path(__file__).parent / "results"
    destination.mkdir(exist_ok=True)
    for name in [
        "REPORT.md",
        "FINDINGS.md",
        "comparison.json",
        "common_accuracy.json",
        "completion.json",
        "measurement_audit.json",
        "cpu_equivalence.json",
        "plan.json",
    ]:
        shutil.copy2(ROOT / name, destination / name)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
