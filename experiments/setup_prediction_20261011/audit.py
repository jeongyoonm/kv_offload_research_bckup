# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit remaining setup, causal replay, and actual deadline tradeoffs."""

import argparse
import gzip
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "setup_runtime", SOURCE / "vllm/v1/kv_offload/setup_predictor.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
MODELS = Path("/mnt/localnvme/jeongyoon/setup_prediction_20261011/frozen_models")


def distribution(values):
    if not values:
        return dict(count=0)
    values = np.array(values)
    return dict(
        count=len(values),
        mean=float(values.mean()),
        median=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )


def errors(predictions, truths):
    if not truths:
        return dict(count=0)
    delta = np.array(predictions) - np.array(truths)
    return dict(
        count=len(delta),
        mae_ms=float(abs(delta).mean()),
        bias_ms=float(delta.mean()),
        covered_percent=float((delta >= 0).mean() * 100),
        shortfall_ms=float(np.maximum(-delta, 0).mean()),
        excess_ms=float(np.maximum(delta, 0).mean()),
        p95_absolute_ms=float(np.quantile(abs(delta), 0.95)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"]
    comparison = json.loads((args.root / "comparison.json").read_text())
    models = {
        name: runtime.SetupPredictor(
            json.loads((MODELS / (file + ".json")).read_text())
        )
        for name, file in (("learned_mean", "mean"), ("learned_q90", "uncertainty"))
    }
    by_model_id = {model.model_id: model for model in models.values()}
    batches, replay, live_forecasts, episodes, audit = [], [], [], [], Counter()
    for episode in summary["results"]:
        directory = (
            args.root
            / "live"
            / episode["workload"]
            / f"r{episode['repeat']}_{episode['label']}"
        )
        events = [
            json.loads(line)
            for line in (directory / "engine.jsonl").read_text().splitlines()
        ]
        computes = {
            e["step"]: e
            for e in events
            if e["event"] == "step_completion" and e.get("selected")
        }
        records = json.loads((directory / "late.json").read_text())["records"]
        ordered = [computes[step] for step in sorted(computes)]
        preceding = {b["step"]: a["step"] for a, b in zip(ordered, ordered[1:])}
        episodes.append(
            dict(
                workload=episode["workload"],
                arm=episode["label"],
                repeat=episode["repeat"],
                throughput_rps=episode["throughput_rps"],
                first_compute_load_state=ordered[0]["load_state"],
                compute_sizes=[len(event["selected"]) for event in ordered],
                seal_reasons=dict(
                    Counter(
                        event["transport"].get("seal_reason", "ordinary")
                        for event in records
                        if event["event"] == "membership_sealed"
                    )
                ),
                load_only_compute_seals=sum(
                    event["transport"].get("has_compute") is False
                    and event["step"] in computes
                    for event in records
                    if event["event"] == "membership_sealed"
                ),
            )
        )
        by_step = defaultdict(list)
        for event in records:
            if "step" in event:
                by_step[event["step"]].append(event)
        for step, points in by_step.items():
            forecasts = [p for p in points if p["event"] == "seal_prediction"]
            seal = next((p for p in points if p["event"] == "membership_sealed"), None)
            if step not in computes or not forecasts or seal is None:
                continue
            prediction = forecasts[0]
            if preceding.get(step) != prediction["predecessor_step"]:
                audit["nonadjacent_predecessor"] += 1
                continue
            a = computes.get(prediction["predecessor_step"])
            if a is None:
                continue
            b = computes[step]
            t = seal["transport"]
            sealed = t["seal_ns"]
            cpu, gpu, fifo = [], [], []
            for r, e, previous in zip(
                b["ranks"], b["execution_ranks"], a["execution_ranks"]
            ):
                origin = max(sealed, r[8])
                assert origin <= r[1] <= r[2] <= e[1] <= e[3]
                cpu.append((e[1] - origin) / 1e6)
                gpu.append(max(0, e[3] - max(e[1], previous[3] + previous[6])) / 1e6)
                fifo.append(max(0, r[8] - sealed) / 1e6)
            ready = max(c + g for c, g in zip(cpu, gpu))
            end = max(r[3] + r[6] for r in a["execution_ranks"])
            predicted_end = (
                prediction["gpu_start_upper_ns"] + prediction["predicted_duration_ns"]
            )
            deadline = t.get("compute_seal_deadline_ns")
            if deadline is None:
                continue
            row = dict(
                workload=episode["workload"],
                arm=episode["label"],
                repeat=episode["repeat"],
                step=step,
                target_cpu_ms=max(cpu),
                target_ready_ms=ready,
                gpu_dependency_ms=max(gpu),
                fifo_wait_ms=max(fifo),
                seal_lead_ms=(end - sealed) / 1e6,
                governing_allowance_ms=(predicted_end - deadline) / 1e6,
                A_error_ms=(predicted_end - end) / 1e6,
                deadline_relative_ideal_ms=(deadline - end) / 1e6 + ready,
                final_requests=len(b["selected"]),
                seal_reason=t["seal_reason"],
                actual_gpu_gap_ms=max(0, max(r[3] for r in b["execution_ranks"]) - end)
                / 1e6,
                rpc_available_before_A_end=all(r[8] <= end for r in b["ranks"]),
            )
            replayed = []
            acked = primary = loads = None
            for point in points:
                if point["time_ns"] > sealed:
                    continue
                if point["event"] == "preparation_ack_transition":
                    acked = point["candidates"]
                if point["event"] == "predictor_submission_progress":
                    primary = point["primary_snapshot"]
                    loads = point["forecast_load_features"]
                if point["event"] == "setup_prediction":
                    model = by_model_id[point["model_id"]]
                    result = model.predict_ms(point["inputs"])
                    assert (result is None and point["predicted_setup_ms"] is None) or (
                        abs(result - point["predicted_setup_ms"]) < 2e-6
                    )
                    audit["live_replay_exact"] += 1
                    audit["fallback"] += point["fallback"]
                    audit["applied"] += point["applied"]
                    live_forecasts.append(
                        dict(
                            workload=episode["workload"],
                            arm=episode["label"],
                            cost_us=point["cost_ns"] / 1000,
                            fallback=point["fallback"],
                            applied=point["applied"],
                        )
                    )
                    continue
                if point["event"] != "seal_preparation_prediction" or acked is None:
                    continue
                inputs = list(prediction["predictor_input"])
                if loads is not None:
                    inputs[11:21] = loads
                x = runtime.setup_features(
                    point["upcoming_shapes"], len(acked), inputs, primary
                )
                values = {name: m.predict_ms(x) for name, m in models.items()}
                replayed.append(
                    dict(
                        **row,
                        forecast_ns=point["time_ns"],
                        inputs=x.tolist() if x is not None else None,
                        shadow_predictions=values,
                        proposed_requests=len(point["upcoming_shapes"]),
                    )
                )
            for i, frame in enumerate(replayed):
                frame.update(first=i == 0, last=i == len(replayed) - 1)
                replay.append(frame)
            batches.append(row)
    levels = []
    for level in comparison["levels"]:
        selected = [
            r
            for r in batches
            if (r["workload"], r["arm"]) == (level["workload"], level["arm"])
        ]
        frames = [
            r
            for r in replay
            if (r["workload"], r["arm"]) == (level["workload"], level["arm"])
        ]
        live = [
            r
            for r in live_forecasts
            if (r["workload"], r["arm"]) == (level["workload"], level["arm"])
        ]
        forecast_metrics = {}
        for model in models:
            for position in ("first", "last"):
                subset = [
                    r
                    for r in frames
                    if r[position] and r["shadow_predictions"][model] is not None
                ]
                forecast_metrics[model + "_" + position] = errors(
                    [r["shadow_predictions"][model] for r in subset],
                    [r["target_ready_ms"] for r in subset],
                )
        ordering = {}
        for position in ("first", "last"):
            subset = [
                r["shadow_predictions"]
                for r in frames
                if r[position]
                and all(v is not None for v in r["shadow_predictions"].values())
            ]
            deltas = [r["learned_q90"] - r["learned_mean"] for r in subset]
            ordering[position] = dict(
                delta_ms=distribution(deltas),
                q90_below_mean_percent=100 * np.mean([v < 0 for v in deltas])
                if deltas
                else None,
            )
        levels.append(
            dict(
                **level,
                setup_accuracy=errors(
                    [r["governing_allowance_ms"] for r in selected],
                    [r["target_ready_ms"] for r in selected],
                ),
                shadow_accuracy=forecast_metrics,
                quantile_ordering=ordering,
                live_forecasts=dict(
                    count=len(live),
                    cost_us=distribution([r["cost_us"] for r in live]),
                    fallback_count=sum(r["fallback"] for r in live),
                    applied_count=sum(r["applied"] for r in live),
                ),
                setup={
                    k: distribution([r[k] for r in selected])
                    for k in (
                        "target_cpu_ms",
                        "target_ready_ms",
                        "gpu_dependency_ms",
                        "fifo_wait_ms",
                        "governing_allowance_ms",
                        "seal_lead_ms",
                        "A_error_ms",
                        "deadline_relative_ideal_ms",
                    )
                },
                rpc_available_before_A_end_percent=100
                * np.mean([r["rpc_available_before_A_end"] for r in selected])
                if selected
                else None,
            )
        )
    payload = dict(
        levels=levels,
        episodes=episodes,
        comparisons=comparison["comparisons"],
        audit=dict(audit),
        definitions="Setup removes ordered RPC waiting and adds residual GPU "
        "event-readiness delay beyond both B enqueue and preceding A end. "
        "This is a readiness proxy, not isolated GEMM start. Positive "
        "deadline_relative_ideal means a later deadline than A_end - observed "
        "B_setup; that counterfactual assumes setup could overlap A. Actual "
        "setup and future membership remain labels only. Shadow predictions "
        "are causal retrospective replays, not serving counterfactuals.",
    )
    (args.root / "setup_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.root / "setup_rows.json.gz").write_bytes(
        gzip.compress(
            json.dumps(dict(batches=batches, frames=replay)).encode(), mtime=0
        )
    )
    print(
        json.dumps(
            dict(
                levels=[
                    {
                        k: r[k]
                        for k in (
                            "workload",
                            "arm",
                            "throughput_rps",
                            "inter_compute_gap_ms",
                            "recovered_percent",
                            "setup_accuracy",
                        )
                    }
                    for r in levels
                ],
                audit=dict(audit),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
