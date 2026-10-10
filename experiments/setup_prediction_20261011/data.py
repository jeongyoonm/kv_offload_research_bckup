# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract prospective B proposals and setup labels with FIFO waiting removed."""

import argparse
import gzip
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "setup_runtime", SOURCE / "vllm/v1/kv_offload/setup_predictor.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, audit = [], Counter()
    for name in ("cs10", "dl10"):
        root = Path("/mnt/localnvme/jeongyoon") / name
        summary = json.loads((root / "summary.json").read_text())
        assert summary["complete"] and summary["stopped"]
        comparison = json.loads((root / "comparison.json").read_text())
        gaps = {
            (g["workload"], g["arm"], g["repeat"], g["step"]): g
            for g in comparison["gaps"]
        }
        for episode in summary["results"]:
            if name == "cs10" and episode["label"] != "best_envelope":
                continue
            assert episode["accepted"] and episode["frozen_unchanged"]
            directory = (
                root
                / "live"
                / episode["workload"]
                / (f"r{episode['repeat']}_{episode['label']}")
            )
            events = [
                json.loads(e)
                for e in (directory / "engine.jsonl").read_text().splitlines()
            ]
            computes = {
                e["step"]: e
                for e in events
                if e["event"] == "step_completion" and e.get("selected")
            }
            records = json.loads((directory / "late.json").read_text())["records"]
            frames = defaultdict(list)
            for record in records:
                if "step" in record:
                    frames[record["step"]].append(record)
            for step, points in frames.items():
                key = episode["workload"], episode["label"], episode["repeat"], step
                if key not in gaps or step not in computes:
                    continue
                gap, b = gaps[key], computes[step]
                previous = computes.get(gap["preceding_step"])
                seal = next(
                    (p for p in points if p["event"] == "membership_sealed"), None
                )
                forecasts = [p for p in points if p["event"] == "seal_prediction"]
                if not seal or not forecasts or previous is None:
                    continue
                forecast = forecasts[0]
                if forecast["predecessor_step"] != previous["step"]:
                    audit["nonadjacent_predecessor"] += 1
                    continue
                sealed = seal["transport"]["seal_ns"]
                ranks, execution = b["ranks"], b["execution_ranks"]
                if any(not r[8] or r[8] > r[1] or r[7] != 1 for r in ranks):
                    audit["missing_rpc_ready"] += 1
                    continue
                cpu, gpu, total, fifo = [], [], [], []
                for rank, (r, e, a) in enumerate(
                    zip(ranks, execution, previous["execution_ranks"])
                ):
                    begin = max(sealed, r[8])
                    assert begin <= r[1] <= r[2] <= e[1] <= e[3]
                    a_end = a[3] + a[6]
                    cpu.append((e[1] - begin) / 1e6)
                    gpu.append(max(0, e[3] - max(e[1], a_end)) / 1e6)
                    total.append(cpu[-1] + gpu[-1])
                    fifo.append(max(0, r[8] - sealed) / 1e6)
                acked, primary, loads = None, None, None
                per_batch = []
                for point in points:
                    if point["time_ns"] > sealed:
                        continue
                    if point["event"] == "preparation_ack_transition":
                        acked = point["candidates"]
                    if point["event"] == "predictor_submission_progress":
                        primary = point["primary_snapshot"]
                        loads = point["forecast_load_features"]
                    if point["event"] != "seal_preparation_prediction":
                        continue
                    if point["time_ns"] < forecast["forecast_ns"] or acked is None:
                        audit["missing_causal_preparation"] += 1
                        continue
                    shapes = point["upcoming_shapes"]
                    features = list(forecast["predictor_input"])
                    if loads is not None:
                        features[11:21] = loads
                    x = runtime.setup_features(shapes, len(acked), features, primary)
                    if x is None:
                        audit["invalid_features"] += 1
                        continue
                    split = (
                        "dev"
                        if name == "cs10" and episode["repeat"] == 1
                        else "final"
                        if name == "dl10" and episode["repeat"] == 1
                        else "train"
                    )
                    per_batch.append(
                        dict(
                            source=name,
                            workload=episode["workload"],
                            arm=episode["label"],
                            repeat=episode["repeat"],
                            group=f"{name}:{episode['workload']}:{episode['seed']}",
                            batch=f"{name}:{episode['workload']}:{episode['label']}:{episode['repeat']}:{step}",
                            split=split,
                            step=step,
                            forecast_ns=point["time_ns"],
                            seal_ns=sealed,
                            features=x.tolist(),
                            shapes=shapes,
                            target_cpu_ms=max(cpu),
                            target_ready_ms=max(total),
                            gpu_dependency_ms=max(gpu),
                            fifo_wait_ms=max(fifo),
                            raw_seal_to_start_ms=(max(e[3] for e in execution) - sealed)
                            / 1e6,
                            final_shapes=b["request_shapes"],
                        )
                    )
                for i, row in enumerate(per_batch):
                    row.update(
                        weight=1 / len(per_batch),
                        first=i == 0,
                        last=i == len(per_batch) - 1,
                    )
                    rows.append(row)
                audit["batches"] += bool(per_batch)
                audit["frames"] += len(per_batch)
    (args.output / "pool.json.gz").write_bytes(
        gzip.compress(json.dumps(rows).encode(), mtime=0)
    )
    counts = Counter(r["split"] for r in rows)
    report = dict(
        counts=dict(counts),
        audit=dict(audit),
        features=runtime.FEATURE_NAMES,
        targets=(
            "Per-rank CPU setup begins at max(seal, next-RPC-ready). "
            "This removes prior-RPC waiting. Target-ready adds GPU start delay "
            "beyond both B enqueue and preceding A GPU end; it is an observed "
            "event-readiness proxy, not isolated GEMM start."
        ),
        groups=(
            "Allocation/workload/seed; policies and every frame of a B batch "
            "remain grouped."
        ),
        features_causal=(
            "B shapes from recorded preparation forecasts; final_shapes are "
            "labels/audit only. A-load progress uses only prior recorded "
            "snapshots. No final membership, actual future completion, or "
            "fixed lead is a model input."
        ),
        limitations=(
            "Four cached workloads, same model and chunks; D2H submission state "
            "is not separately recorded as a causal feature. GPU residual can "
            "include later pipeline state and is tested as a distinct target."
        ),
    )
    (args.output / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
