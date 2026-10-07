# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test causal CUDA-marker enqueue lead separately from engine FIFO queue time."""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import improve
import joblib
import numpy as np
import study
from threadpoolctl import threadpool_limits

EXTRA_NAMES = [
    "gpu_marker_enqueue_gap_min_ms",
    "gpu_marker_enqueue_gap_max_ms",
    "gpu_marker_enqueue_gap_median_ms",
    "gpu_marker_enqueue_gap_std_ms",
    "rank_start_skew_ms",
]
MODES = {
    "gpu_enqueue_features": list(range(5)),
    "gpu_enqueue_gap_only": list(range(4)),
    "rank_skew_only": [4],
}


def register_schemas():
    # Research-only schemas; the serving metadata extractor is unchanged.
    for indices in MODES.values():
        extra = [EXTRA_NAMES[i] for i in indices]
        for names in [
            study.arch.surface.FEATURE_NAMES + extra,
            study.arch.surface.FEATURE_NAMES
            + study.arch.surface.FORECAST_NAMES
            + extra,
            study.arch.surface.FEATURE_NAMES
            + study.arch.surface.FORECAST_NAMES
            + extra
            + study.arch.surface.BASE_PREDICTION_NAMES,
        ]:
            if names not in study.arch.surface.FEATURE_SCHEMAS:
                study.arch.surface.FEATURE_SCHEMAS.append(names)


register_schemas()


def extra_features(rows):
    result = []
    for row in rows:
        ranks = row["execution_ranks"]
        assert len(ranks) == 8
        forecast = row.get("forecast_time_ns", max(r[3] for r in ranks))
        assert all(0 < r[1] <= r[3] <= forecast for r in ranks)
        gaps = sorted((r[3] - r[1]) / 1e6 for r in ranks)
        starts = [r[3] for r in ranks]
        mean = sum(gaps) / 8
        result.append(
            [
                gaps[0],
                gaps[-1],
                (gaps[3] + gaps[4]) / 2,
                (sum((v - mean) ** 2 for v in gaps) / 8) ** 0.5,
                (max(starts) - min(starts)) / 1e6,
            ]
        )
    return np.array(result)


def fit_variants(name, batches, rows, selected, seed, verify_rows=None):
    improve.configure()
    runtime = study.arch.runtime_features(rows)
    extra = extra_features(rows)
    names = study.arch.surface.FEATURE_NAMES + study.arch.surface.FORECAST_NAMES
    states, audit = {}, {}
    base_config = correction_config = None
    if name == "trees_fixed":
        base_config = study.arch.FIXED_BASE
        correction_config = study.arch.FIXED_CORRECTION
    elif not name.endswith("_direct"):
        family = "tree" if name == "trees_tuned" else name.split("_")[0]
        base_config = selected[f"base_{family}"]
        if not name.endswith("_base_only"):
            correction = "tree" if name == "trees_tuned" else name.split("_")[1]
            correction_config = selected[f"correction_{family}_{correction}"]
    base = oof = None
    if correction_config:
        base, _ = improve.fit(
            base_config,
            study.arch.features(batches),
            study.arch.target(batches),
            batches,
            seed,
        )
        oof, audit["base_oof"] = study.arch.cross_fit(base_config, batches, rows, seed)
    verified = verify_rows if verify_rows is not None else rows[:200]
    vx = np.column_stack(
        (study.arch.runtime_features(verified), extra_features(verified))
    )
    for mode, indices in MODES.items():
        extras = [EXTRA_NAMES[i] for i in indices]
        model_indices = list(range(23)) + [23 + i for i in indices]
        x = np.column_stack((runtime, extra[:, indices]))
        if name.endswith("_direct"):
            family = name.split("_")[0]
            model, _ = improve.fit(
                selected[f"direct_{family}"],
                x,
                study.arch.target(rows),
                rows,
                seed,
            )
            state = {"direct": study.arch.export_component(model, names + extras)}
            expected = study.arch.predict(model, vx[:, model_indices])
        elif correction_config:
            model, _ = improve.fit(
                correction_config,
                np.column_stack((x, oof)),
                study.arch.target(rows) - oof,
                rows,
                seed,
            )
            state = dict(
                base=study.arch.export_component(
                    base, study.arch.surface.FEATURE_NAMES
                ),
                correction=study.arch.export_component(
                    model, names + extras + study.arch.surface.BASE_PREDICTION_NAMES
                ),
            )
            bp = study.arch.predict(base, vx[:, :11])
            expected = bp + study.arch.predict(
                model, np.column_stack((vx[:, model_indices], bp))
            )
        else:
            model_indices = list(range(11)) + [23 + i for i in indices]
            x = np.column_stack(
                (study.arch.features(batches), extra_features(batches)[:, indices])
            )
            model, _ = improve.fit(
                base_config, x, study.arch.target(batches), batches, seed
            )
            state = {
                "direct": study.arch.export_component(
                    model, study.arch.surface.FEATURE_NAMES + extras
                )
            }
            expected = study.arch.predict(model, vx[:, model_indices])
        state["input_indices"] = model_indices
        states[mode] = dict(**improve.wrapped(state), gpu_enqueue_features=True)
        actual = improve.predict_state(states[mode], verified)
        error = float(np.max(np.abs(actual - np.maximum(1, expected))))
        assert error < 1e-4, (mode, error)
        audit[mode] = dict(export_equivalence_max_ms=error)
    return states, audit


def run_task(task):
    directory = study.ROOT / "enqueue" / task["name"]
    complete = directory / "completed.json"
    if complete.exists():
        return json.loads(complete.read_text())
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    proposal = json.loads((study.ROOT / "improvement_plan.json").read_text())
    pool = joblib.load(task["dataset"])
    train = [pool["forecasts"][i] for i in task["train"]]
    test = [pool["forecasts"][i] for i in task["test"]]
    held = {r["episode"] for r in test}
    batches = [r for r in pool["batches"] if r["episode"] not in held]
    if task["regime"] == "matched":
        batches = train
    assert not held.intersection(r["episode"] for r in train + batches)
    selected = json.loads(
        (study.ROOT / "folds" / task["name"] / "tuning.json").read_text()
    )["selected"]
    with threadpool_limits(limits=1):
        states, audit = fit_variants(
            proposal["architecture"], batches, train, selected, task["seed"], test
        )
        predictions = {n: improve.predict_state(s, test) for n, s in states.items()}
    parent = study.ROOT / "improvements" / task["name"]
    with np.load(parent / "predictions.npz") as f:
        arrays = {k: f[k] for k in f.files}
    arrays.update(predictions)
    temporary = directory / "merged.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(parent / "predictions.npz")
    existing = json.loads((parent / "checkpoints.json").read_text())
    existing.update(states)
    study.arch.dump(parent / "checkpoints.json", existing)
    study.arch.dump(directory / "checkpoints.json", states)
    result = dict(
        task=task["name"],
        seconds=time.perf_counter() - started,
        audit=audit,
        future_end_fields_used=False,
        test_training_episode_overlap=0,
    )
    study.arch.dump(complete, result)
    print(json.dumps(dict(event="enqueue_task_complete", **result)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rows = joblib.load(study.ROOT / "pool.joblib")["forecasts"]
    leads = extra_features(rows)[:, 2]
    summary = json.loads((study.ROOT / "improvement_summary.json").read_text())
    selection = json.loads((study.ROOT / "final_selection.json").read_text())
    historical = summary["episode:wide"]["scores"]["historical"][
        selection["improvement"]
    ]["mae_ms"]
    triggered = (leads >= 50).sum() >= 30 and historical > 5
    study.arch.dump(
        study.ROOT / "enqueue_plan.json",
        dict(
            triggered=bool(triggered),
            forecasts_with_lead_ge50ms=int((leads >= 50).sum()),
            historical_cv_mae_ms=historical,
            criterion=(
                "at least 30 training-pool leads >=50ms and historical CV MAE >5ms"
            ),
            benchmark_used_for_planning=False,
            input_contract=(
                "only immutable marker enqueue/start words observed by forecast time"
            ),
            no_future_end_fields=True,
            serving_integration_required=True,
            capacity_and_training_only_selected_configs_unchanged=True,
        ),
    )
    if not triggered:
        return
    tasks = json.loads((study.ROOT / "tasks.json").read_text())
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
    proposal = json.loads((study.ROOT / "improvement_plan.json").read_text())
    proposal["candidates"] = list(dict.fromkeys(proposal["candidates"] + list(MODES)))
    proposal["reasons"]["gpu_enqueue_features"] = (
        "Matching geometry showed shorter spans when GPU marker enqueue led "
        "physical start by 69-92ms. Compare actual marker lead against rank skew "
        "with existing geometry, transfer and engine queue features unchanged."
    )
    study.arch.dump(study.ROOT / "improvement_plan.json", proposal)
    improve.summarize(tasks, proposal)


if __name__ == "__main__":
    main()
