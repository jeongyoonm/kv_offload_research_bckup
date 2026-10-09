# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit completed cells, launch invariants and predictor provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import regex as re

from experiments.store_overlap_overnight_20261009.summarize import BASE, RUNS

EXPECTED_LAUNCH = {
    "--tensor-parallel-size": "8",
    "--kv-cache-dtype": "fp8_e4m3",
    "--kv-cache-memory-bytes": "103079215104",
    "--max-model-len": "65536",
    "--max-num-batched-tokens": "61440",
    "--max-num-seqs": "64",
    "--block-size": "64",
}
CHECKPOINT_FILE_SHA = "048c50302c0405ce2dcad233f064d13088f9bf1282c018837237d854f9bc2572"


def audit(root):
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"] and summary["stopped"], root
    plan = json.loads((root / "plan.json").read_text())
    command = json.loads((root / "server/command.json").read_text())
    launch = {}
    for option, expected in EXPECTED_LAUNCH.items():
        value = command[command.index(option) + 1]
        assert value == expected, (root, option, value)
        launch[option] = value
    model = next(x for x in command if "models--Inferact--GLM-5.3-NVFP4" in x)
    assert model.endswith("ce67b36f3669192b5bb233819f0fda6c8a9837f8"), model
    results = summary["results"]
    assert results and all(r["accepted"] for r in results), root
    for row in results:
        assert len(row["responses"]) == row["requests"], root
        assert all(r["usage"]["completion_tokens"] == 1 for r in row["responses"])
        if row.get("full_prefix_residency_required", True):
            assert row["expected_cached"] == row["observed_cached"], root
    predictor = root / "loaded_predictor.json"
    semantic_sha = None
    if any("checkpoint_sha256" in r for r in results):
        assert plan["checkpoint_file_sha256"] == CHECKPOINT_FILE_SHA, root
        state = json.loads(predictor.read_text())
        semantic_sha = hashlib.sha256(
            json.dumps(state, sort_keys=True).encode()
        ).hexdigest()
        assert all(r["checkpoint_sha256"] == semantic_sha for r in results), root
        assert all(r["frozen_unchanged"] for r in results), root
        assert all(r["optimized_preparation_verified"] for r in results), root
    connector = root / "server/connector_config.json"
    extra = {}
    if connector.exists():
        config = json.loads(connector.read_text())
        extra = config.get("kv_connector_extra_config", {}) if config else {}
    if extra:
        assert extra["cpu_bytes_to_use"] == 2130300354560, root
        assert extra["cpu_cache_layout"] in ("private", "private_tensor"), root
        assert extra["async_h2d_submission"], root
        assert extra["async_h2d_chunk_bytes"] == 8388608, root
    log = (root / "server/server.log").read_text(errors="replace")
    capacities = sorted(set(re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)))
    assert capacities == ["2,160,960"], (root, capacities)
    return dict(
        root=str(root),
        model=model,
        launch=launch,
        cpu_bytes=extra.get("cpu_bytes_to_use"),
        cpu_layout=extra.get("cpu_cache_layout"),
        native_d2h=extra.get("async_d2h_submission", False),
        measured_cells=len(results),
        measured_requests=sum(r["requests"] for r in results),
        accepted=True,
        stopped=True,
        reported_gpu_tokens=capacities,
        checkpoint_file_sha=plan.get("checkpoint_file_sha256"),
        predictor_state_sha=semantic_sha,
        source=json.loads((root / "source_provenance.json").read_text()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-references", action="store_true")
    args = parser.parse_args()
    names = tuple(RUNS) + ("sg1", "sg2", "sg3", "sg4")
    records = []
    for name in names:
        root = BASE / name
        if (root / "summary.json").exists():
            records.append(audit(root))
    if args.require_references:
        assert {"sg2", "sg3", "sg4"}.issubset(
            {Path(r["root"]).name for r in records}
        ), "Required resident references are not complete"
        thp = json.loads((BASE / "sg4/gpu_worker_thp_policy.json").read_text())
        assert len(thp) == 8 and all(r["thp_enabled"] == 0 for r in thp.values())
    args.output.write_text(
        json.dumps(
            dict(
                consistent=True,
                records=records,
                definitions="Acceptance is cache, request, launch and telemetry "
                "validation, not whole-model accuracy. Predictor file hashes "
                "identify the frozen source checkpoint; state hashes identify "
                "its parsed runtime state after the reviewed H2D-context update.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(audited_allocations=len(records), consistent=True)))


if __name__ == "__main__":
    main()
