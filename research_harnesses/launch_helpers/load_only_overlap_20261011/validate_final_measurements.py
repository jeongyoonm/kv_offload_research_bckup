"""Verify closed serving evidence before report/archive completion is claimed."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    report = json.loads((ROOT / "final_report/report.json").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    assert report["complete"] or args.allow_partial
    assert report["expected_cells"] == 204 and 0 < report["cells"] <= 204
    if report["complete"]:
        assert report["cells"] == 204 and report["matched_arm_triplets"] == 68
    assert len(report["provenance"]) == 10
    for name, provenance in report["provenance"].items():
        assert provenance["stopped"], name
        if report["complete"]:
            assert provenance["complete"], name
        summary = json.loads((ROOT.parent / name / "summary.json").read_text())
        assert summary["stopped"]
        if report["complete"]:
            assert summary["complete"]
        assert all(row["accepted"] for row in summary["results"])

    cells = []
    full_coverage = []
    total_requests = 0
    for detail in report["timing_details"]:
        source = Path(detail["source"])
        data = json.loads(source.read_text())
        assert data["accepted"] and data["repeat"] != "warmup", source
        metrics = data["measured_metric_deltas"]
        assert metrics["vllm:request_prompt_tokens_sum"] == (
            metrics["vllm:prompt_tokens_cached_total"]
            + metrics["vllm:request_prefill_kv_computed_tokens_sum"]
        ), source
        assert metrics.get("vllm:num_preemptions_total", 0) == 0, source
        assert sum(data["realized_outcomes"].values()) == data["requests"]
        evolving = data["workload"]["family"] == "evolving"
        assert data["cache_reset_between_measurement_cohorts"] == (not evolving)
        if not evolving:
            assert data["realized_outcomes"] == {"cpu_hit": data["requests"]}
            context = data["workload"]["shapes"][0][0] if (
                data["workload"]["family"] == "homogeneous"
            ) else None
            if context is not None:
                assert all(n == context for n in data["cached_tokens"])
        for direction, verb in (("h2d", "load"), ("d2h", "store")):
            per_rank = data[f"{direction}_bytes_per_rank"]
            assert set(map(int, per_rank)) == set(range(8)), source
            assert sum(per_rank.values()) == data[f"{direction}_bytes_all_ranks"]
            assert data[f"{direction}_bytes_all_ranks"] == metrics[
                f"vllm:kv_offload_{verb}_bytes_total"
            ]
        assert all(n == 8 for n in data["store_rank_coverage"]), source
        if data["arm"] == "full":
            assert data["frozen_predictor_unchanged"] is True, source
            coverage = detail["worker_preparation_coverage"]
            assert coverage is not None and set(map(int, coverage)) == set(range(8))
            assert all(
                row["compute_entries"] == row["total_compute_steps"]
                and not row["unobserved_steps"]
                for row in coverage.values()
            ), source
            full_coverage.append({"source": str(source), "ranks": coverage})
        else:
            assert data["frozen_predictor_unchanged"] in (None, True), source
        total_requests += data["requests"]
        cells.append({
            "source": str(source), "arm": data["arm"],
            "requests": data["requests"], "family": data["workload"]["family"],
            "prompt_tokens": metrics["vllm:request_prompt_tokens_sum"],
            "computed_prompt_tokens": metrics["vllm:request_prefill_kv_computed_tokens_sum"],
            "cached_prompt_tokens": metrics["vllm:prompt_tokens_cached_total"],
        })
    assert len(cells) == report["cells"]
    assert total_requests == sum(c["requests"] for c in cells)
    assert len(full_coverage) == sum(c["arm"] == "full" for c in cells)
    expected_keys = set()
    actual_keys = set()
    for name, provenance in report["provenance"].items():
        plan = provenance["plan"]
        for spec in plan["specs"]:
            for repeat in range(plan["repeats"]):
                for concurrency in plan["concurrencies"]:
                    for arm in plan["arms"]:
                        key = (name, spec["id"], repeat, concurrency, arm)
                        assert key not in expected_keys
                        expected_keys.add(key)
    for detail in report["timing_details"]:
        data = json.loads(Path(detail["source"]).read_text())
        name = Path(detail["source"]).parents[3].name
        assert data["requests"] == report["provenance"][name]["plan"]["requests"]
        key = (name, data["workload"]["id"], data["repeat"], data["concurrency"], data["arm"])
        assert key in expected_keys and key not in actual_keys, key
        actual_keys.add(key)
    assert len(expected_keys) == 204 and len(actual_keys) == report["cells"]
    missing = sorted(expected_keys - actual_keys)
    assert bool(missing) == (not report["complete"])
    if report["complete"]:
        assert total_requests == 18720 and len(full_coverage) == 68
        for arm in ("baseline", "worker", "full"):
            assert sum(c["arm"] == arm for c in cells) == 68
            assert sum(c["arm"] == arm and c["family"] == "homogeneous" for c in cells) == 60
        for concurrency in (1, 32, 64):
            for arm in ("baseline", "worker", "full"):
                assert sum(r["family"] == "homogeneous" and r["concurrency"] == concurrency
                           and r["arm"] == arm for r in report["rows"]) == 16

    byte_audit = json.loads((ROOT / "block_audit_summary.json").read_text())
    assert byte_audit["byte_correctness_accepted"]
    assert not byte_audit["original_performance_acceptance"]
    assert {r["rank"] for r in byte_audit["ranks"]} == set(range(8))
    verified_bytes = sum(sum(r["verified_bytes"].values()) for r in byte_audit["ranks"])
    assert verified_bytes == 580666982400
    model = json.loads((ROOT / "all10_model_runtime_configuration_audit.json").read_text())
    assert len(model["roots"]) == 10
    assert all(not r["serving_runtime_differences"] for r in model["roots"])
    recovery = json.loads((ROOT / "recovery_configuration_audit.json").read_text())
    assert recovery["same_model_command"] and recovery["same_connector_configuration"]
    assert recovery["same_workload_and_policies"]
    assert recovery["sole_runtime_change"] == ["vllm/v1/kv_offload/late_admission_runtime.py"]
    checkpoint = ROOT / "predictor/frozen_A.json"
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checksum == "f159f32d226894457fb56b3195c47d4ff6e2a5d982d8d600b1970ea000294f96"

    result = {
        "utc": str(datetime.now(timezone.utc)), "measurement_gates_passed": True,
        "cells": cells, "main_cells": len(cells), "matched_triplets": report["matched_arm_triplets"],
        "planned_grid_complete": report["complete"], "missing_cells": missing,
        "measured_requests": total_requests, "full_rank_coverage": full_coverage,
        "byte_audit_verified_bytes": verified_bytes,
        "frozen_predictor_sha256": checksum,
        "scope": "Serving/accounting/closed all-rank entry evidence only. Entries observed "
        "are distinct from prepared/compact/pinned usage. Numerical accuracy for "
        "arbitrary workloads is not established. Archive integrity, final plot review "
        "and owned-process shutdown remain separate completion gates.",
    }
    (ROOT / "final_measurement_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS {len(cells)} accepted main cells, {total_requests} requests, "
          f"{len(full_coverage)} full cells with all8 ranks; missing={len(missing)}")


if __name__ == "__main__":
    main()
