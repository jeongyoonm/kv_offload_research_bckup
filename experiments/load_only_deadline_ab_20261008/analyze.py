# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit the four-arm load-only readiness and deadline comparison."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path("/mnt/localnvme/glm53_load_only_deadline_ab_20261008")
SOURCE = Path(__file__).resolve().parents[2]
os.environ["VLLM_DEFERRAL_FAST_FOREST"] = (
    "/mnt/localnvme/glm53_predictor_overnight_20261006/fast_forest/build/"
    "libpredictor_forest_walk.so"
)
for name in ["vllm", "vllm.v1", "vllm.v1.kv_offload"]:
    package = types.ModuleType(name)
    package.__path__ = [str(SOURCE / name.replace(".", "/"))]
    sys.modules[name] = package
from vllm.v1.kv_offload.seal_predictor import ComputeEstimator  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "offset_diagnostics", SOURCE / "experiments/offset_predictor_20261008/analyze.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    plan = read(ROOT / "plan.json")
    policies = ["all_off", "half_off", "all_on", "half_on"]
    states = [read(ROOT / "models" / f"{p}.json") for p in policies]
    off_state = states[0]
    assert all(state == off_state for state in states)
    digest = hashlib.sha256(json.dumps(off_state, sort_keys=True).encode()).hexdigest()
    model = ComputeEstimator.from_state_dict(
        off_state, off_state["context"]
    ).latency_surface
    assert model.excludes_elapsed
    cells, predictions, audit = [], [], []
    for path in sorted((ROOT / "live").glob("*/r[0-7]_*/result.json")):
        result = read(path)
        directory = path.parent
        assert result["accepted"] and result["frozen_unchanged"]
        assert result["checkpoint_sha256"] == digest
        assert result["requests"] == len(result["responses"]) == 75
        assert result["concurrency"] == 32
        assert result["expected_cached"] == result["observed_cached"]
        assert result["optimized_preparation_verified"]
        assert all(
            r["entries"] == result["compute_rounds"]
            for r in result["worker_preparation_coverage"].values()
        )
        policy = read(directory / "policy.json")
        assert policy["seal_estimator"]["loaded"]
        assert all(policy[k] == v for k, v in plan["policies"][result["label"]].items())
        for metric in ["vllm:prefix_cache_hits_total", "vllm:num_preemptions_total"]:
            assert result["metrics_before"][metric] == result["metrics_after"][metric]
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
        events = [
            json.loads(line)
            for line in (directory / "engine.jsonl").read_text().splitlines()
        ]
        completed, scheduled = {}, Counter()
        for event in events:
            if event["event"] != "step_completion":
                continue
            assert event["valid"] and event["step"] not in completed
            completed[event["step"]] = event
            for internal_id, tokens in event.get("selected", {}).items():
                matches = [
                    request
                    for prefix, request in prefixes.items()
                    if internal_id.startswith(prefix)
                ]
                assert len(matches) == 1
                scheduled[matches[0]] += tokens
        assert dict(scheduled) == expected
        late = read(directory / "late.json")["records"]
        cell_predictions = []
        for point in late:
            if (
                point["event"] != "seal_prediction"
                or point.get("predictor_input") is None
            ):
                continue
            a = completed[point["predecessor_step"]]
            ranks = a["execution_ranks"]
            start = max(r[3] for r in ranks)
            end = max(r[3] + r[6] for r in ranks)
            data = dict(
                a,
                native_predictor=True,
                forecast_ns=point["forecast_ns"],
                execution_snapshot=point["execution_snapshot"],
                forecast_load_features=point["predictor_input"][11:21],
            )
            replay, features = model.predict_with_features(data)
            assert replay == point["predicted_duration_ns"]
            assert np.max(np.abs(features - np.array(point["predictor_input"]))) < 1e-8
            row = dict(
                workload=result["workload"],
                policy=result["label"],
                repeat=result["repeat"],
                cell=str(directory),
                predecessor_step=a["step"],
                successor_step=point["step"],
                predicted_ms=replay / 1e6,
                actual_ms=(end - start) / 1e6,
                error_ms=(replay - end + start) / 1e6,
            )
            cell_predictions.append(row)
        load_only = []
        for sealed in [e for e in late if e["event"] == "membership_sealed"]:
            transport = sealed["transport"]
            if transport.get("has_compute") is False:
                percent = 50 if result["label"].startswith("half_") else 0
                n = transport["load_only_pending_candidates"]
                assert transport["load_only_ready_percent"] == percent
                target = (n * percent + 99) // 100 if percent else None
                assert transport["load_only_fraction_target"] == target
                if percent:
                    assert transport["ready_target"] == min(
                        target, n - len(transport["cancelled"])
                    )
                if transport.get("seal_reason") == "candidate_target_ready":
                    assert transport["partial_ready_count"] >= transport["ready_target"]
                    assert transport["statuses"][:8] == [1] * 8
                load_only.append(
                    dict(
                        step=sealed["step"],
                        pending=n,
                        target=transport["ready_target"],
                        reason=transport.get("seal_reason"),
                        activated=len(sealed["activated"]),
                        selected=len(sealed["selected"]),
                        ready=transport["partial_ready_count"],
                        unready=n
                        - len(transport["cancelled"])
                        - transport["partial_ready_count"],
                        wait_after_base_ms=(
                            transport["seal_ns"] - transport["base_ready_ns"]
                        )
                        / 1e6,
                    )
                )
            if transport.get("compute_seal_deadline_ns") is not None:
                p = transport["prediction"]
                assert (
                    transport["compute_seal_deadline_ns"]
                    == p["gpu_start_upper_ns"] + p["predicted_duration_ns"] - 5_000_000
                )
                assert transport["compute_seal_deadline_enabled"] == (
                    result["label"].endswith("_on")
                )
                assert transport["compute_seal_deadline_shadow"] == (
                    result["label"].endswith("_off")
                )
            if result["label"].endswith("_off"):
                assert transport.get("seal_reason") != "compute_seal_deadline"
        log = (directory / "measurement_server.log").read_text()
        runtime_errors = [
            line
            for line in log.splitlines()
            if any(
                pattern in line
                for pattern in ["Traceback", "ERROR", "CUDA error", "out of memory"]
            )
        ]
        compilation = [
            line
            for line in log.splitlines()
            if any(
                pattern in line.lower()
                for pattern in [
                    "compiling",
                    "jit compilation",
                    "building extension",
                    "torch.compile takes",
                ]
            )
        ]
        assert not runtime_errors, runtime_errors
        assert not compilation, compilation
        d = helper.diagnostics(directory)
        compute = sorted(
            [e for e in completed.values() if e.get("selected")],
            key=lambda e: max(r[3] for r in e["execution_ranks"]),
        )
        seals = {e["step"]: e for e in late if e["event"] == "membership_sealed"}
        forecast_times = [
            e["forecast_ns"] for e in late if e["event"] == "seal_prediction"
        ]
        first_forecast = min(forecast_times) if forecast_times else None
        d["initial_compute_batches_before_first_forecast"] = [
            len(e["selected"])
            for e in compute
            if first_forecast is None or seals[e["step"]]["time_ns"] < first_forecast
        ]
        d["all_compute_batch_sizes"] = [len(e["selected"]) for e in compute]
        categories = Counter()
        category_counts = Counter()
        for a, b in zip(compute, compute[1:]):
            end = max(r[3] + r[6] for r in a["execution_ranks"])
            start = max(r[3] for r in b["execution_ranks"])
            transport = seals[b["step"]]["transport"]
            category = (
                "load_only"
                if transport.get("has_compute") is False
                else "prediction_deadline"
                if transport.get("seal_reason") == "compute_seal_deadline"
                else "other"
            )
            categories[category] += max(0, start - end) / 1e6
            category_counts[category] += 1
        assert abs(sum(categories.values()) - d["gpu_gap_ms"]) < 1e-6
        d["gap_categories_ms"] = dict(categories)
        d["gap_category_counts"] = dict(category_counts)
        d["load_only_seals"] = load_only
        result["diagnostics"] = d
        cells.append(result)
        predictions.extend(cell_predictions)
        audit.append(
            dict(
                cell=str(directory),
                passed=True,
                requests=75,
                scheduled_tokens=sum(scheduled.values()),
                unique_ordered_completions=len(completed),
                frozen_checkpoint=True,
                matched_policy=True,
                native_forecasts_replayed=len(cell_predictions),
                runtime_errors=runtime_errors,
                measured_compilation=compilation,
            )
        )
    save(ROOT / "measurement_audit.json", audit)
    save(ROOT / "prediction_audit.json", predictions)
    save(ROOT / "audited_cells.json", cells)
    if args.audit_only:
        print(json.dumps(dict(cells=len(cells), forecasts=len(predictions))))
        return
    assert len(cells) == plan["measured_cells"]
    assert sum(c["requests"] for c in cells) == plan["measured_requests"]
    aggregates, contrasts = [], []
    comparisons = [
        ("half_ready_deadline_off", "all_off", "half_off"),
        ("half_ready_deadline_on", "all_on", "half_on"),
        ("deadline_wait_all", "all_off", "all_on"),
        ("deadline_half_ready", "half_off", "half_on"),
    ]
    for workload, repeats in plan["repeats"].items():
        selected = [c for c in cells if c["workload"] == workload]
        for repeat in range(repeats):
            block = [c for c in selected if c["repeat"] == repeat]
            assert len(block) == len(policies)
            assert len({c["seed"] for c in block}) == 1
            metric = "vllm:kv_offload_load_bytes_total"
            assert (
                len(
                    {
                        c["metrics_after"][metric] - c["metrics_before"][metric]
                        for c in block
                    }
                )
                == 1
            )
        for policy in policies:
            group = [c for c in selected if c["label"] == policy]
            assert len(group) == repeats
            ttft = [r["ttft_ms"] for c in group for r in c["responses"]]
            diagnostics = [c["diagnostics"] for c in group]
            load_seals = [s for d in diagnostics for s in d["load_only_seals"]]
            forecast_errors = [
                p["error_ms"]
                for p in predictions
                if p["workload"] == workload and p["policy"] == policy
            ]
            aggregates.append(
                dict(
                    workload=workload,
                    policy=policy,
                    episodes=len(group),
                    requests=sum(c["requests"] for c in group),
                    throughput_rps=sum(c["requests"] for c in group)
                    / sum(c["seconds"] for c in group),
                    mean_ttft_ms=statistics.mean(ttft),
                    p95_ttft_ms=float(np.quantile(ttft, 0.95)),
                    compute_rounds=sum(c["compute_rounds"] for c in group),
                    late_admissions=sum(c["late_admissions"] for c in group),
                    kv_ready_during_compute=sum(
                        c["kv_ready_during_compute"] for c in group
                    ),
                    recovered_compute_opportunities=sum(
                        c["recovered_compute_opportunities"] for c in group
                    ),
                    mean_episode_gap_ms=statistics.mean(
                        d["gpu_gap_ms"] for d in diagnostics
                    ),
                    mean_episode_compute_span_ms=statistics.mean(
                        d["gpu_compute_span_ms"] for d in diagnostics
                    ),
                    gap_categories_ms_per_episode={
                        name: statistics.mean(
                            d["gap_categories_ms"].get(name, 0) for d in diagnostics
                        )
                        for name in ["load_only", "prediction_deadline", "other"]
                    },
                    seal_reasons=dict(
                        sum(
                            (Counter(d["seal_reasons"]) for d in diagnostics), Counter()
                        )
                    ),
                    load_only_seals=len(load_seals),
                    load_only_seals_with_unready_candidates=sum(
                        s["unready"] > 0 for s in load_seals
                    ),
                    load_only_wait_after_base_ms=helper.describe(
                        [s["wait_after_base_ms"] for s in load_seals]
                    ),
                    load_only_selected_requests=helper.describe(
                        [s["selected"] for s in load_seals]
                    ),
                    load_only_pending_counts=helper.describe(
                        [s["pending"] for s in load_seals]
                    ),
                    load_only_targets=helper.describe(
                        [s["target"] for s in load_seals]
                    ),
                    issued_prediction_mae_ms=statistics.mean(
                        abs(e) for e in forecast_errors
                    )
                    if forecast_errors
                    else None,
                    issued_forecasts=len(forecast_errors),
                )
            )
        for name, reference, treatment in comparisons:
            pairs = []
            for repeat in range(repeats):
                a, b = [
                    next(
                        c for c in selected if c["label"] == p and c["repeat"] == repeat
                    )
                    for p in [reference, treatment]
                ]
                pairs.append(
                    dict(
                        repeat=repeat,
                        reference_rps=a["throughput_rps"],
                        treatment_rps=b["throughput_rps"],
                        gain_percent=100
                        * (b["throughput_rps"] / a["throughput_rps"] - 1),
                    )
                )
            ratios = [math.log(p["treatment_rps"] / p["reference_rps"]) for p in pairs]
            average = statistics.mean(ratios)
            margin = (
                t.ppf(0.975, repeats - 1)
                * statistics.stdev(ratios)
                / math.sqrt(repeats)
            )
            a, b = [
                next(
                    g
                    for g in aggregates
                    if g["workload"] == workload and g["policy"] == p
                )
                for p in [reference, treatment]
            ]
            contrasts.append(
                dict(
                    workload=workload,
                    comparison=name,
                    reference=reference,
                    treatment=treatment,
                    aggregate_gain_percent=100
                    * (b["throughput_rps"] / a["throughput_rps"] - 1),
                    paired_gain_percent=100 * math.expm1(average),
                    paired_95pct_bounds=[
                        100 * math.expm1(average - margin),
                        100 * math.expm1(average + margin),
                    ],
                    mean_ttft_reduction_percent=100
                    * (1 - b["mean_ttft_ms"] / a["mean_ttft_ms"]),
                    pairs=pairs,
                )
            )
    report = dict(
        complete=True,
        aggregate=aggregates,
        contrasts=contrasts,
        measured_cells=len(cells),
        measured_requests=sum(c["requests"] for c in cells),
        forecasts_replayed=len(predictions),
        checkpoint_sha256=digest,
        scope=plan["scope"],
        issued_accuracy_population="Policy-dependent; identical frozen checkpoint",
    )
    save(ROOT / "comparison.json", report)
    lines = [
        "# Load-only half-ready policy with and without predictor deadline",
        "",
        "GLM 5.3 NVFP4, TP8, concurrency 32, one output token.",
        "All four arms use the same frozen predictor and transfer/preparation path.",
        "",
        "<!-- markdownlint-disable MD013 -->",
        "",
        "| Workload | Policy | req/s | Mean TTFT ms | p95 TTFT ms | Compute rounds | "
        "GPU gap ms/episode | Load-only gap ms/episode |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in aggregates:
        lines.append(
            f"| {g['workload']} | {g['policy']} | {g['throughput_rps']:.3f} | "
            f"{g['mean_ttft_ms']:.1f} | {g['p95_ttft_ms']:.1f} | "
            f"{g['compute_rounds']} | {g['mean_episode_gap_ms']:.1f} | "
            f"{g['gap_categories_ms_per_episode']['load_only']:.1f} |"
        )
    lines.extend(
        [
            "",
            "| Workload | Comparison | Aggregate gain | Paired gain (95% interval) | "
            "Mean TTFT reduction |",
            "| --- | --- | ---: | --- | ---: |",
        ]
    )
    for c in contrasts:
        lo, hi = c["paired_95pct_bounds"]
        lines.append(
            f"| {c['workload']} | {c['comparison']} | "
            f"{c['aggregate_gain_percent']:+.2f}% | "
            f"{c['paired_gain_percent']:+.2f}% [{lo:+.2f}%, {hi:+.2f}%] | "
            f"{c['mean_ttft_reduction_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "<!-- markdownlint-enable MD013 -->",
            "",
            "Each arm/workload has four reset-and-reprimed 75-request episodes:",
            "300 requests total, not one continuous 300-request run. Twelve",
            "32-request warmups, startup, and priming are excluded. Position and",
            "immediate carryover are balanced across the four repeats. Confidence",
            "intervals use paired log throughput ratios with episodes as samples.",
            "Four repeats provide limited precision; intervals spanning zero do",
            "not establish a repeatable throughput benefit.",
            "",
            "GPU gaps use all-rank execution timing and exclude startup/final tail.",
            "Compute spans include submission stalls and output joins. Gap",
            "categories are classified by the successor seal's compute state.",
            "The fraction denominator is the frozen published candidate cohort",
            "(up to 16), not all outstanding transfers. Cancellations cap the",
            "target. Compute-active readiness rules remain unchanged.",
            "",
            "Audit: exact scheduled token accounting, unique ordered completion,",
            "equal transfer bytes within every four-arm block, CPU restoration",
            "counts, all-rank optimized preparation, frozen checkpoint replay,",
            "deadline/fraction arithmetic, and no measured compilation/errors.",
            "Original selector and model weights are preserved.",
            "",
            f"Artifacts: `{ROOT}`.",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
