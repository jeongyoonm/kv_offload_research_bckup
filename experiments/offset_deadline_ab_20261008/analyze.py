# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit matched deadline serving cells and report paired outcomes."""

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

ROOT = Path("/mnt/localnvme/glm53_offset_deadline_ab_20261008")
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
    plan = read(ROOT / "plan.json")
    off, on = "deadline_off", "deadline_on"
    off_state, on_state = [read(ROOT / "models" / f"{p}.json") for p in [off, on]]
    assert off_state == on_state
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
        for sealed in [e for e in late if e["event"] == "membership_sealed"]:
            transport = sealed["transport"]
            if transport.get("compute_seal_deadline_ns") is not None:
                p = transport["prediction"]
                assert (
                    transport["compute_seal_deadline_ns"]
                    == p["gpu_start_upper_ns"] + p["predicted_duration_ns"] - 5_000_000
                )
                assert transport["compute_seal_deadline_enabled"] == (
                    result["label"] == on
                )
                assert transport["compute_seal_deadline_shadow"] == (
                    result["label"] == off
                )
            if result["label"] == off:
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
    assert len(cells) == plan["measured_cells"]
    assert sum(c["requests"] for c in cells) == plan["measured_requests"]
    aggregates, pairs = [], []
    for workload, repeats in plan["repeats"].items():
        selected = [c for c in cells if c["workload"] == workload]
        for repeat in range(repeats):
            a, b = [
                next(c for c in selected if c["label"] == p and c["repeat"] == repeat)
                for p in [off, on]
            ]
            assert a["seed"] == b["seed"]
            metric = "vllm:kv_offload_load_bytes_total"
            bytes_a = a["metrics_after"][metric] - a["metrics_before"][metric]
            bytes_b = b["metrics_after"][metric] - b["metrics_before"][metric]
            assert bytes_a == bytes_b
            pairs.append(
                dict(
                    workload=workload,
                    repeat=repeat,
                    off_rps=a["throughput_rps"],
                    on_rps=b["throughput_rps"],
                    gain_percent=100 * (b["throughput_rps"] / a["throughput_rps"] - 1),
                    aggregate_h2d_bytes=bytes_a,
                )
            )
        ratios = [
            math.log(p["on_rps"] / p["off_rps"])
            for p in pairs
            if p["workload"] == workload
        ]
        avg = statistics.mean(ratios)
        margin = (
            t.ppf(0.975, repeats - 1) * statistics.stdev(ratios) / math.sqrt(repeats)
        )
        bounds = [100 * math.expm1(avg - margin), 100 * math.expm1(avg + margin)]
        for policy in [off, on]:
            group = [c for c in selected if c["label"] == policy]
            ttft = [r["ttft_ms"] for c in group for r in c["responses"]]
            errors = [
                p["error_ms"]
                for p in predictions
                if p["workload"] == workload and p["policy"] == policy
            ]
            aggregates.append(
                dict(
                    workload=workload,
                    policy=policy,
                    episodes=len(group),
                    requests=75 * len(group),
                    throughput_rps=sum(c["requests"] for c in group)
                    / sum(c["seconds"] for c in group),
                    mean_ttft_ms=statistics.mean(ttft),
                    p95_ttft_ms=float(np.quantile(ttft, 0.95)),
                    compute_rounds=sum(c["compute_rounds"] for c in group),
                    late_admissions=sum(c["late_admissions"] for c in group),
                    mean_episode_gap_ms=statistics.mean(
                        c["diagnostics"]["gpu_gap_ms"] for c in group
                    ),
                    mean_episode_gap_lower_ms=statistics.mean(
                        c["diagnostics"]["gpu_gap_lower_ms"] for c in group
                    ),
                    mean_episode_gap_upper_ms=statistics.mean(
                        c["diagnostics"]["gpu_gap_upper_ms"] for c in group
                    ),
                    seal_reasons=dict(
                        sum(
                            (Counter(c["diagnostics"]["seal_reasons"]) for c in group),
                            Counter(),
                        )
                    ),
                    issued_prediction_mae_ms=statistics.mean(abs(e) for e in errors)
                    if errors
                    else None,
                    issued_forecasts=len(errors),
                    paired_gain_percent=100 * math.expm1(avg),
                    paired_95pct_bounds=bounds,
                    batch_sizes=[
                        s for c in group for s in c["diagnostics"]["batch_requests"]
                    ],
                    gap_categories_ms_per_episode={
                        name: statistics.mean(
                            c["diagnostics"]["gap_categories_ms"].get(name, 0)
                            for c in group
                        )
                        for name in ["load_only", "prediction_deadline", "other"]
                    },
                    gap_category_counts=dict(
                        sum(
                            (
                                Counter(c["diagnostics"]["gap_category_counts"])
                                for c in group
                            ),
                            Counter(),
                        )
                    ),
                )
            )
    report = dict(
        complete=True,
        aggregate=aggregates,
        pairs=pairs,
        measured_cells=len(cells),
        measured_requests=sum(c["requests"] for c in cells),
        forecasts_replayed=len(predictions),
        checkpoint_sha256=digest,
        scope=plan["scope"],
        issued_accuracy_population="Policy-dependent; same checkpoint in both arms",
    )
    save(ROOT / "comparison.json", report)
    save(ROOT / "measurement_audit.json", audit)
    save(ROOT / "prediction_audit.json", predictions)
    lines = [
        "# Latest offset predictor: deadline enabled versus disabled",
        "",
        "GLM 5.3 NVFP4, TP8, concurrency 32, one output token. Both arms use",
        "the identical frozen offset predictor and optimized preparation/transfer.",
        "The disabled arm computes and records the deadline in shadow mode.",
        "",
        "<!-- markdownlint-disable MD013 -->",
        (
            "| Workload | Deadline off req/s | Deadline on req/s | Aggregate gain | "
            "Paired gain (95% interval) | Mean TTFT off/on ms | "
            "p95 TTFT off/on ms | GPU gap off/on ms per episode | "
            "Compute rounds off/on |"
        ),
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for workload in plan["repeats"]:
        a, b = [
            next(
                g for g in aggregates if g["workload"] == workload and g["policy"] == p
            )
            for p in [off, on]
        ]
        lo, hi = b["paired_95pct_bounds"]
        lines.append(
            f"| {workload} | {a['throughput_rps']:.3f} | {b['throughput_rps']:.3f} | "
            f"{100 * (b['throughput_rps'] / a['throughput_rps'] - 1):+.2f}% | "
            f"{b['paired_gain_percent']:+.2f}% [{lo:+.2f}%, {hi:+.2f}%] | "
            f"{a['mean_ttft_ms']:.1f} / {b['mean_ttft_ms']:.1f} | "
            f"{a['p95_ttft_ms']:.1f} / {b['p95_ttft_ms']:.1f} | "
            f"{a['mean_episode_gap_ms']:.1f} / {b['mean_episode_gap_ms']:.1f} | "
            f"{a['compute_rounds']} / {b['compute_rounds']} |"
        )
    lines.extend(
        [
            "<!-- markdownlint-enable MD013 -->",
            "",
            "Eight pairs for 60K and four pairs each for 16K and mixed workloads.",
            "Each episode has 75 requests and separate cache reset/priming.",
            "These are not uninterrupted 300-request cells. Six 32-request warmups",
            "and all startup/priming are excluded. Intervals use paired log throughput",
            "ratios and Student-t critical values; the pair is the sample unit.",
            "Mixed workloads have few predictor decisions. Forecast MAE populations",
            "depend on the policy and are not an architecture comparison.",
            "GPU spans include submission stalls and output joins. Gaps exclude",
            "startup and the final tail; they do not measure isolated kernel time.",
            "All requests passed CPU restoration, exact token accounting, unique",
            "ordered completion, and original output-token-count checks. No selector",
            "oracle, model/weight modification or DMA audit was used.",
            "",
            f"Full artifacts: `{ROOT}`.",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
