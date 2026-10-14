# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit deadline serving cells and compare them with the shadow control."""

import argparse
import hashlib
import importlib.util
import json
import statistics
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def stats(values):
    return (
        dict(
            count=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            min=min(values),
            max=max(values),
        )
        if values
        else dict(count=0)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--control", default="compact_shadow")
    args = parser.parse_args()
    run = args.root / "canary/gpu"
    provenance = read(run / "provenance.json")
    source = Path(provenance["source"])
    for name, expected in provenance["source_sha256"].items():
        if name.startswith("vllm/") or name.startswith("tests/"):
            assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location(
        "deadline_poll_analysis",
        source / "experiments/poll_prediction_20261003/analyze.py",
    )
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    workers = [
        read(p)
        for p in (run / "server/timing/late_admission").glob("*.json")
        if read(p).get("role") == "worker"
    ]
    cells = []
    excluded_jit_warmups = []
    for path in sorted(run.glob("p*/*/result.json")):
        directory = path.parent
        result = analyzer.analyze(directory)
        if not result["accepted"] and result.get("measurement_jit_lines"):
            assert not result["runtime_error_lines"]
            assert not result["unexpected_gpu_pids"]
            excluded_jit_warmups.append(str(path))
            continue
        assert result["accepted"], path
        assert not read(directory / "server_cpu.json")["errors"]
        setting = read(directory / "policy_setting.json")
        assert setting["late_compact_tokens"] and setting["late_pinned_packing"]
        assert setting["late_cached_requests"] and setting["late_seal_freeze_estimator"]
        usage = read(directory / "usage.json")
        request_ids = {r["request_id"] for r in usage}
        assert len(request_ids) == len(usage)
        events = [
            json.loads(line)
            for line in (directory / "engine.jsonl").read_text().splitlines()
        ]
        dispatch = {e["step"]: e for e in events if e["event"] == "dispatch"}
        computed = [
            e for e in events if e["event"] == "step_completion" and e.get("selected")
        ]
        computed.sort(key=lambda e: e["dispatch_ns"])
        assert sum(sum(e["selected"].values()) for e in computed) == sum(
            r["prompt"] - r["cached"] for r in usage
        )
        assert all(len(e["execution_ranks"]) == 8 for e in computed)
        gaps, late_enqueue = [], []
        for before, after in zip(computed, computed[1:]):
            ar, br = before["execution_ranks"], after["execution_ranks"]
            lower = max(0, max(r[2] for r in br) - max(r[8] for r in ar)) / 1e6
            upper = max(0, max(r[3] for r in br) - max(r[7] for r in ar)) / 1e6
            gaps.append((lower, upper))
            if not any(
                before["step"] < s < after["step"] and not d.get("selected")
                for s, d in dispatch.items()
            ):
                late_enqueue.append(
                    max(0, after["dispatch_ns"] - max(r[8] for r in ar)) / 1e6
                )
        late = read(directory / "late.json")
        seals = [r for r in late["records"] if r["event"] == "membership_sealed"]
        for r in seals:
            t = r.get("transport", {})
            if t.get("compute_seal_deadline_ns") is None:
                continue
            p = t["prediction"]
            assert t["compute_seal_deadline_ns"] == (
                p["gpu_start_upper_ns"]
                + p["predicted_duration_ns"]
                - setting["late_seal_compute_deadline_lead_us"] * 1000
            )
            assert t["effective_seal_expiry_ns"] == min(
                t["compute_seal_deadline_ns"],
                t["base_ready_ns"] + setting["late_seal_safety_us"] * 1000,
            )
        reasons = Counter(
            r.get("transport", {}).get("seal_reason", "ordinary") for r in seals
        )
        hard = [
            r
            for r in seals
            if r.get("transport", {}).get("seal_reason") == "compute_seal_deadline"
        ]
        overshoots = [
            r["transport"]["compute_seal_deadline_overshoot_ns"] / 1e6 for r in hard
        ]
        completions = {e["step"]: e for e in computed}
        firings = []
        for r in hard:
            t = r["transport"]
            assert (
                t["compute_seal_deadline_enabled"]
                and t["seal_ns"] >= t["effective_seal_expiry_ns"]
            )
            assert t["statuses"][:8] == [1] * 8
            for w in t["adaptive_waits"]:
                assert w["requested_ns"] <= max(
                    0, t["effective_seal_expiry_ns"] - w["decision_ns"]
                )
            preceding = completions[t["prediction"]["predecessor_step"]]
            end_lower = max(x[7] for x in preceding["execution_ranks"])
            end_upper = max(x[8] for x in preceding["execution_ranks"])
            last_wait = t["adaptive_waits"][-1] if t["adaptive_waits"] else None
            firings.append(
                dict(
                    step=r["step"],
                    actual_seal_lead_lower_ms=(end_lower - t["seal_ns"]) / 1e6,
                    actual_seal_lead_upper_ms=(end_upper - t["seal_ns"]) / 1e6,
                    actual_enqueue_lead_lower_ms=(
                        end_lower - dispatch[r["step"]]["dispatch_ns"]
                    )
                    / 1e6,
                    actual_enqueue_lead_upper_ms=(
                        end_upper - dispatch[r["step"]]["dispatch_ns"]
                    )
                    / 1e6,
                    last_wait_requested_ms=(
                        last_wait["requested_ns"] / 1e6 if last_wait else None
                    ),
                    last_wait_actual_ms=(
                        (last_wait["end_ns"] - last_wait["begin_ns"]) / 1e6
                        if last_wait
                        else None
                    ),
                )
            )
        coverage = {}
        steps = {e["step"] for e in computed}
        for export in workers:
            if export["generation"] != late["generation"]:
                continue
            assert not export["overflow"]
            entries = [
                e
                for e in export["records"]
                if e["event"] == "worker_entry" and e.get("step") in steps
            ]
            coverage[export["rank"]] = dict(
                compute_entries=len(entries),
                compact=sum(e.get("compact_tokens", 0) for e in entries),
                pinned=sum(e.get("pinned_tokens", 0) for e in entries),
            )
        assert set(coverage) == set(range(8))
        assert all(
            v["compute_entries"] == len(computed)
            and v["compact"] > 0
            and v["pinned"] > 0
            for v in coverage.values()
        )
        cells.append(
            dict(
                cell=directory.name,
                path=str(directory),
                requests=len(usage),
                throughput=result["throughput"],
                mean_ttft_ms=result["mean_ttft_ms"],
                p95_ttft_ms=result["p95_ttft_ms"],
                compute_rounds=len(computed),
                late_admissions=result["late_admissions"],
                recovered_compute_opportunities=result[
                    "recovered_compute_opportunities"
                ],
                kv_ready_during_compute=result["kv_ready_during_compute"],
                missed_ready_at_worker_entry_with_seal_ack=result[
                    "missed_ready_at_worker_entry_with_seal_ack"
                ],
                h2d_bytes=result["h2d_bytes"],
                compute_span_ms=sum(
                    max(r[6] for r in e["execution_ranks"]) for e in computed
                )
                / 1e6,
                gap_lower_ms=sum(g[0] for g in gaps),
                gap_upper_ms=sum(g[1] for g in gaps),
                non_load_late_enqueue_lower_ms=sum(late_enqueue),
                seal_reasons=dict(reasons),
                deadline_overshoot_ms=stats(overshoots),
                deadline_firings=firings,
                worker_coverage=coverage,
            )
        )
    assert len({r["h2d_bytes"] for r in cells}) == 1
    controls = {
        r["cell"].split("_")[0]: r for r in cells if f"_{args.control}" in r["cell"]
    }
    for r in cells:
        control = controls[r["cell"].split("_")[0]]
        r["throughput_gain_pct"] = 100 * (r["throughput"] / control["throughput"] - 1)
        r["mean_ttft_reduction_pct"] = 100 * (
            1 - r["mean_ttft_ms"] / control["mean_ttft_ms"]
        )
    correctness = {
        name: read(run / name / "summary.json")
        for name in ["correctness", "correctness_distinct"]
    }
    assert all(r["passed"] for s in correctness.values() for r in s.values())
    report = dict(
        source_commit=provenance["commit"],
        cells=cells,
        excluded_jit_warmups=excluded_jit_warmups,
        correctness=correctness,
        scope=(
            "Cell gains use the control from the same repeat. CUDA execution "
            "spans include output-copy and host gaps, not just model kernels. "
            "One cell per lead does not establish a repeatable speedup."
        ),
    )
    (args.root / "serving_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(cells, indent=2))


if __name__ == "__main__":
    main()
