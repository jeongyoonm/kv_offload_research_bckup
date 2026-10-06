# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Estimate preparation-path mismatch using audited TP8 execution traces."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "seal_predictor", SOURCE / "vllm/v1/kv_offload/seal_predictor.py"
)
predictor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predictor)


def read(path):
    return json.loads(path.read_text())


def signature(event):
    return tuple(sorted(tuple(s) for s in event["request_shapes"]))


def duration(event):
    return max(r[6] for r in event["execution_ranks"]) / 1e6


def metrics(errors):
    return dict(
        count=len(errors),
        bias_ms=statistics.mean(errors),
        mae_ms=statistics.mean(abs(e) for e in errors),
        stddev_ms=statistics.stdev(errors),
        p95_absolute_error_ms=float(np.quantile(np.abs(errors), 0.95)),
    )


def train(events):
    model = predictor.ComputeEstimator()
    for event in events:
        model.observe(event, event["execution_ranks"])
    assert model.coefficients is not None and len(model.samples) == len(events)
    model.frozen = True
    return model


def evaluate(model, events, mode):
    model.calibration_mode = mode
    return metrics([model.predict(e) / 1e6 - duration(e) for e in events])


def matched_training(data, repeats, seed):
    """Pair labels with identical feature sequences, within independent repeats."""
    random = np.random.default_rng(seed)
    pairs = []
    for repeat in repeats:
        groups = {mode: defaultdict(list) for mode in data}
        for mode in data:
            for event in data[mode][repeat]:
                groups[mode][signature(event)].append(event)
        for shape in sorted(set(groups["current"]) & set(groups["compact_pinned"])):
            count = min(len(groups[mode][shape]) for mode in data)
            selected = {
                mode: random.choice(len(groups[mode][shape]), count, replace=False)
                for mode in data
            }
            for index in range(count):
                pairs.append(
                    {mode: groups[mode][shape][selected[mode][index]] for mode in data}
                )
    random.shuffle(pairs)
    result = {mode: [pair[mode] for pair in pairs] for mode in data}
    assert [signature(e) for e in result["current"]] == [
        signature(e) for e in result["compact_pinned"]
    ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensitivity-seeds", type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "process_log.jsonl"

    def log(event, **fields):
        with log_path.open("a") as stream:
            stream.write(
                json.dumps(dict(event=event, time=time.time(), **fields)) + "\n"
            )

    log("started", argv=sys.argv)
    try:
        data = {mode: {} for mode in ("current", "compact_pinned")}
        input_hashes, cells, settings = {}, [], {}

        def load_events(directory, expected_requests):
            path = directory / "engine.jsonl"
            input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            events = [
                e
                for line in path.read_text().splitlines()
                if (e := json.loads(line))["event"] == "step_completion"
                and e.get("selected")
            ]
            assert sum(sum(e["selected"].values()) for e in events) == (
                expected_requests * 3072
            )
            for event in events:
                assert len(event["execution_ranks"]) == len(event["ranks"]) == 8
                assert all(s == (3072, 58368) for s in signature(event))
                assert len(event["selected"]) == len(event["request_shapes"])
                for execution, preparation in zip(
                    event["execution_ranks"], event["ranks"]
                ):
                    assert execution[0] == preparation[0] == event["step"]
                    assert execution[5] == preparation[7] == 1
                    assert execution[6] > 0 and execution[8] > execution[3] > 0
                    assert execution[1] >= preparation[2] > preparation[1] > 0
            return sorted(events, key=lambda e: e["dispatch_ns"])

        for cell in read(args.training_root / "cells.json"):
            mode = cell["mode"]
            if mode not in data:
                continue
            assert cell["accepted"] and not cell["gpu_hits"] and not cell["preemptions"]
            assert not cell["missing_execution_timing_steps"]
            assert not cell["measurement_jit_lines"] and not cell["runtime_error_lines"]
            assert not cell["unexpected_gpu_pids"]
            directory = Path(cell["path"])
            repeat = directory.name.split("_")[0]
            settings[mode, repeat] = read(directory / "policy_setting.json")
            data[mode][repeat] = load_events(directory, 80)
            cells.append(cell)
        repeats = sorted(data["current"])
        assert repeats == sorted(data["compact_pinned"]) == ["r0", "r1", "r2"]
        for repeat in repeats:
            a, b = settings["current", repeat], settings["compact_pinned", repeat]
            differences = {key for key in a if a[key] != b[key]}
            assert differences == {
                "late_cached_requests",
                "late_pinned_packing",
                "late_compact_tokens",
            }
            for key in differences:
                assert a[key] is False and b[key] is True
            pair = [c for c in cells if Path(c["path"]).name.startswith(repeat + "_")]
            assert len({c["h2d_bytes"] for c in pair}) == 1
            assert len({c["workload_manifest"] for c in pair}) == 1

        validation = []
        for cell in read(args.validation_root / "serving_comparison.json")["cells"]:
            validation.extend(load_events(Path(cell["path"]), 75))
        train_ids = {
            (e["dispatch_ns"], e["step"])
            for mode in data.values()
            for events in mode.values()
            for e in events
        }
        assert not train_ids & {(e["dispatch_ns"], e["step"]) for e in validation}
        assert len(validation) == 104
        state = read(args.validation_root / "trained_model.json")["state"]
        for path in (
            args.training_root / "cells.json",
            args.training_root / "canary/gpu/provenance.json",
            args.validation_root / "serving_comparison.json",
            args.validation_root / "trained_model.json",
        ):
            input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for cell in cells:
            path = Path(cell["path"]) / "policy_setting.json"
            input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        old_model = predictor.ComputeEstimator.from_state_dict(state, state["context"])
        old_model.frozen = True

        def comparison(training, held_out):
            models = {mode: train(events) for mode, events in training.items()}
            return {
                calibration: {
                    mode: evaluate(model, held_out, calibration)
                    for mode, model in models.items()
                }
                for calibration in ("legacy", "raw", "shape")
            }

        all_training = {
            mode: [e for repeat in repeats for e in blocks[repeat]]
            for mode, blocks in data.items()
        }
        unbalanced = comparison(all_training, validation)
        matched = matched_training(data, repeats, 0)
        balanced = comparison(matched, validation)
        cross_validation = []
        for held_out in repeats:
            training = matched_training(data, [r for r in repeats if r != held_out], 0)
            for mode in data:
                rows = comparison(training, data[mode][held_out])
                cross_validation.append(
                    dict(repeat=held_out, evaluation_path=mode, metrics=rows)
                )

        sensitivity = defaultdict(list)
        sensitivity_metrics = defaultdict(lambda: defaultdict(list))
        for seed in range(args.sensitivity_seeds):
            training = matched_training(data, repeats, seed)
            result = comparison(training, validation)
            for calibration, row in result.items():
                sensitivity[calibration].append(
                    row["current"]["mae_ms"] - row["compact_pinned"]["mae_ms"]
                )
                for mode, values in row.items():
                    sensitivity_metrics[calibration][mode].append(values)
        sensitivity = {
            mode: dict(
                seeds=len(values),
                mean_mismatch_penalty_ms=statistics.mean(values),
                min_ms=min(values),
                max_ms=max(values),
                p05_ms=float(np.quantile(values, 0.05)),
                p95_ms=float(np.quantile(values, 0.95)),
                definition=(
                    "MAE trained on current minus MAE trained on compact; "
                    "positive means mismatch hurts"
                ),
                scope="Training selection/order sensitivity, not a confidence interval",
                mean_metrics_by_training_path={
                    path: {
                        key: statistics.mean(row[key] for row in rows)
                        for key in (
                            "bias_ms",
                            "mae_ms",
                            "stddev_ms",
                            "p95_absolute_error_ms",
                        )
                    }
                    for path, rows in sensitivity_metrics[mode].items()
                },
            )
            for mode, values in sensitivity.items()
        }
        label_shifts = []
        for repeat in repeats:
            groups = {mode: defaultdict(list) for mode in data}
            for mode in data:
                for e in data[mode][repeat]:
                    groups[mode][signature(e)].append(e)
            for shape in sorted(set(groups["current"]) & set(groups["compact_pinned"])):
                row = dict(repeat=repeat, request_count=len(shape), paths={})
                for mode in data:
                    events = groups[mode][shape]
                    row["paths"][mode] = dict(
                        count=len(events),
                        mean_execution_ms=statistics.mean(duration(e) for e in events),
                        mean_pre_marker_preparation_ms=statistics.mean(
                            max(r[2] - r[1] for r in e["ranks"]) / 1e6 for e in events
                        ),
                    )
                row["compact_minus_current_execution_ms"] = (
                    row["paths"]["compact_pinned"]["mean_execution_ms"]
                    - row["paths"]["current"]["mean_execution_ms"]
                )
                label_shifts.append(row)
        report = dict(
            source_commit="8fd94fa",
            training_source=str(args.training_root),
            validation_source=str(args.validation_root),
            training_compute_batches={
                mode: len(events) for mode, events in all_training.items()
            },
            balanced_training_compute_batches={
                mode: len(events) for mode, events in matched.items()
            },
            held_out_compute_batches=len(validation),
            input_sha256=input_hashes,
            original_checkpoint={
                mode: evaluate(old_model, validation, mode)
                for mode in ("legacy", "raw", "shape")
            },
            warmed_training_unbalanced=unbalanced,
            warmed_training_balanced=balanced,
            leave_one_repeat_out=cross_validation,
            training_selection_sensitivity=sensitivity,
            matched_shape_execution_and_preparation=label_shifts,
            scope=(
                "Retrospective prediction-error comparison. Three alternating paired "
                "TP8 serving repeats share workload and policy except preparation "
                "flags. Balanced training has identical feature sequences and counts; "
                "request membership and overlap state are not forced identical. "
                "Latest calibration traces are fully held out; they enforce a 5ms "
                "deadline absent from training. No new GPU run or throughput "
                "comparison. Direct preparation duration is before the execution "
                "marker; concurrent "
                "preparation interference can remain inside execution spans. "
                "Random-seed ranges describe sensitivity rather than independent-run "
                "confidence."
            ),
        )
        (args.output / "comparison.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        log("completed", held_out_compute_batches=len(validation))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k
                    not in ("input_sha256", "matched_shape_execution_and_preparation")
                },
                indent=2,
            )
        )
    except BaseException as error:
        log("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
