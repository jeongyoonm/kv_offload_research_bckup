# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for general-prefill extraction and initial causal snapshots."""

import importlib.util
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from matrix import build, feasible  # noqa: E402

from vllm.v1.kv_offload.dataset_shadow import (  # noqa: E402
    DatasetShadow,
    collection_compute_held,
)
from vllm.v1.kv_offload.latency_surface import batch_features  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "coverage_extract", SOURCE / "experiments/predictor_corpus_20261006/corpus.py"
)
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


def event():
    return dict(
        event="step_completion",
        step=7,
        valid=True,
        prefill_only=True,
        selected={"a": 128, "b": 512},
        request_shapes=[[128, 0], [512, 8192]],
        load_state=[2, 16384],
        cpu_cached_prefix_tokens=[0, 8192],
        dispatch_ns=5,
        execution_ranks=[[7, 30, 40, 50, 120, 1, 60, 110, 115]] * 8,
        ranks=[[7, 10, 20, 30, 100, 110, 120, 1, 0]] * 8,
    )


def test_accept_mixed_prefill_without_relaxing_decode_or_legacy_guard():
    e = event()
    assert corpus.extract(e)[0] is None
    row, reason = corpus.extract(e, cached_only=False)
    assert reason is None and np.isfinite(batch_features(row)).all()
    e["prefill_only"] = False
    assert corpus.extract(e, cached_only=False)[1] == "not_prefill_only"
    assert batch_features(dict(row, prefill_only=False)) is None
    e["prefill_only"] = True
    e["selected"]["a"] = 129
    assert corpus.extract(e, cached_only=False)[1] == "token_accounting"


def test_shadow_requires_all_rank_start_and_emits_once_without_seal():
    ranks = [[7, 1, 2, 3, 4, 1, 0, 0, 0] for _ in range(8)]
    timing = SimpleNamespace(
        steps={7: {"prefill_only": True}},
        execution_ranks=[None] * 8,
        read_execution=lambda _: ranks,
    )
    shadow = DatasetShadow.__new__(DatasetShadow)
    shadow.timing = timing
    shadow.events = queue.SimpleQueue()
    shadow.seen = set()
    ranks[-1][3] = 0
    shadow.sample()
    assert not shadow.drain()
    ranks[-1][3] = 3
    shadow.sample()
    result = shadow.drain()
    assert len(result) == 1 and result[0]["gpu_start_upper_ns"] == 3
    assert result[0]["time_ns"] >= 3
    shadow.sample()
    assert not shadow.drain()
    timing.steps = {8: {"prefill_only": True}}
    for r in ranks:
        r[0] = 8
        r[8] = 6
    shadow.sample()
    assert not shadow.drain()


def test_matrix_is_deterministic_and_holds_out_complete_combinations():
    m = build()
    assert m == build()
    assert all(feasible(f["shapes"]) for f in m["train"] + m["challenge"])
    train = {
        tuple(map(tuple, f["shapes"])) for f in m["train"] if f["block"] == "geometry"
    }
    held = {
        tuple(map(tuple, f["shapes"]))
        for f in m["challenge"]
        if f["block"] == "combination"
    }
    assert not train & held
    assert not set(m["splits"]["train_seeds"]) & set(m["splits"]["final_seeds"])


def test_collection_gates_allow_lookup_and_release_compute(monkeypatch, tmp_path):
    target = SimpleNamespace(request_id="cmpl-coverage-main-1", num_computed_tokens=0)
    background = SimpleNamespace(request_id="cmpl-coverage-bg-1", num_computed_tokens=0)
    assert not collection_compute_held(target)
    assert not collection_compute_held(background)
    gate = tmp_path / "hold"
    monkeypatch.setenv("VLLM_DATASET_BACKGROUND_PREFIX", "coverage-bg-")
    monkeypatch.setenv("VLLM_DATASET_TARGET_GATE", str(gate))
    gate.touch()
    assert not collection_compute_held(target)
    assert not collection_compute_held(background)
    target.num_computed_tokens = 8192
    background.num_computed_tokens = 16384
    assert collection_compute_held(target) and collection_compute_held(background)
    gate.unlink()
    assert not collection_compute_held(target) and collection_compute_held(background)


def test_shadow_and_live_options_satisfy_the_serving_configuration_contract():
    from run import options

    from vllm.v1.kv_offload.early_readiness import validate_experiment_options

    shadow = validate_experiment_options(options(False))
    live = validate_experiment_options(options(True))
    assert shadow["late_seal_policy"] == "event"
    assert not shadow["late_seal_native_predictor"]
    assert not shadow["late_seal_compute_deadline"]
    assert live["late_seal_policy"] == "predictive"
    assert live["late_seal_compute_deadline"]
    assert live["late_compact_tokens"] and shadow["late_compact_tokens"]
