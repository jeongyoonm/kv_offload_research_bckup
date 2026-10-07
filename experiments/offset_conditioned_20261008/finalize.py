# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve the conditioned comparison with request and checkpoint audits."""

import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path("/mnt/localnvme/glm53_offset_conditioned_20261008")
helper_path = (
    Path(__file__).resolve().parents[1] / "offset_predictor_20261008/finalize.py"
)
spec = importlib.util.spec_from_file_location("offset_finalize", helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    audits = helper.audit_measurements(ROOT, 9, 675)
    allocation = json.loads((ROOT / "allocation.json").read_text())
    assert allocation["server_stopped"]
    conditioning = sorted((ROOT / "live").glob("*/rconditioning_*/result.json"))
    assert len(conditioning) == 9
    for path in conditioning:
        result = json.loads(path.read_text())
        assert result["accepted"] and result["requests"] == 32
        assert result["label"] == "trees"
    phase_pairs = []
    for path in conditioning:
        result = json.loads(path.read_text())
        repeat, name = result["repeat"].removeprefix("conditioning_").split("_", 1)
        measured = json.loads(
            (ROOT / "live/familiar60k" / f"r{repeat}_{name}/result.json").read_text()
        )
        phase_pairs.append(
            dict(
                repeat=int(repeat),
                predictor=name,
                common_tree_conditioning_rps=result["throughput_rps"],
                measured_rps=measured["throughput_rps"],
            )
        )
    comparison = json.loads((ROOT / "comparison.json").read_text())
    accuracy = json.loads((ROOT / "audit_live/audit.json").read_text())
    assert comparison["complete"] and accuracy["issued_replay_max_difference_ns"] == 0
    ratios = []
    for repeat in range(3):
        base = ROOT / "live/familiar60k"
        a = json.loads((base / f"r{repeat}_matched_offset/result.json").read_text())
        b = json.loads((base / f"r{repeat}_matched_elapsed/result.json").read_text())
        ratios.append(math.log(a["throughput_rps"] / b["throughput_rps"]))
    mean = float(np.mean(ratios))
    margin = float(t.ppf(0.975, 2) * np.std(ratios, ddof=1) / math.sqrt(3))
    paired = dict(
        gain_percent=100 * math.expm1(mean),
        bounds=[100 * math.expm1(mean - margin), 100 * math.expm1(mean + margin)],
    )
    completion = dict(
        complete=True,
        measured_cells=len(audits),
        measured_requests=675,
        excluded_conditioning_cells=9,
        excluded_conditioning_requests=288,
        all_request_accounting_passed=True,
        allocation=allocation,
        paired_offset_vs_elapsed=paired,
        issued_replay_difference_ns=accuracy["issued_replay_max_difference_ns"],
        conditioning_phase_pairs=phase_pairs,
    )
    (ROOT / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    lines = [
        "# Conditioned predictor comparison: GLM 5.3 TP8",
        "",
        "Each measured episode follows an excluded common-tree 32-request",
        "episode using the same workload and seed. Measured caches are then",
        "reset and primed separately. Three balanced rotations use fresh",
        "seeds 3301–3303. Checkpoints remain frozen from the primary study.",
        "Each predictor has 225 measured requests, for 675 in total.",
        "",
        "| Predictor | req/s | Mean TTFT (ms) | p95 TTFT (ms) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group in comparison["aggregate"]:
        lines.append(
            f"| {group['predictor']} | {group['throughput_rps']:.3f} | "
            f"{group['mean_ttft_ms']:.1f} | {group['p95_ttft_ms']:.1f} |"
        )
    lo, hi = paired["bounds"]
    lines += [
        "",
        f"Paired offset-only versus elapsed-input gain: {paired['gain_percent']:+.2f}%",
        f"with a 95% Student-t interval [{lo:+.2f}%, {hi:+.2f}%].",
        "Three episodes provide limited precision; this is not equivalence.",
        "",
        "## Conditioning phase check",
        "",
        "The conditioning checkpoint and procedure are identical across models.",
        "Its throughput is measured on 32 excluded requests, while each main",
        "episode uses 75; the absolute values are not equal-length comparisons.",
        "Variation among conditioning episodes exposes phase differences even",
        "without a predictor change. Conditioning does not guarantee identical",
        "physical state at the start of each measured episode.",
        "",
        (
            "| Repeat | Measured predictor | Common-tree conditioning req/s | "
            "Measured req/s |"
        ),
        "| ---: | --- | ---: | ---: |",
    ]
    for row in phase_pairs:
        lines.append(
            f"| {row['repeat']} | {row['predictor']} | "
            f"{row['common_tree_conditioning_rps']:.3f} | {row['measured_rps']:.3f} |"
        )
    lines += [
        "",
        "## Accuracy at common live snapshots",
        "",
        "| Predictor | Points | MAE (ms) | MAPE | Bias (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for workload in accuracy["live"]:
        for name, metrics in workload["models"].items():
            a = metrics["accuracy"]
            lines.append(
                f"| {name} | {workload['count']} | {a['mae_ms']:.2f} | "
                f"{a['mape_percent']:.2f}% | {a['bias_ms']:+.2f} |"
            )
    lines += [
        "",
        "All issued forecasts, feature snapshots, and deadlines replay exactly.",
        "All measured requests have exact uncached-token accounting and",
        "unique ordered completion steps. Frozen models, unchanged policy,",
        "all-rank compact/pinned preparation, CPU restoration, zero GPU prefix",
        "hits and zero preemptions are verified. GPU servers are stopped.",
        "",
        "This controls the immediately preceding workload episode; it does",
        "not identify the physical cause of any phase effect. No fresh live",
        "labels were used for training, selection, or checkpoint changes.",
        f"Raw traces and reports: `{ROOT}`.",
    ]
    (ROOT / "FINDINGS.md").write_text(helper.format_report("\n".join(lines)))
    (ROOT / "REPORT.md").write_text(
        helper.format_report((ROOT / "REPORT.md").read_text())
    )
    dest = Path(__file__).parent / "results"
    dest.mkdir(exist_ok=True)
    for name in [
        "FINDINGS.md",
        "REPORT.md",
        "comparison.json",
        "completion.json",
        "measurement_audit.json",
        "plan.json",
        "phase_audit.json",
    ]:
        shutil.copy2(ROOT / name, dest / name)
    shutil.copy2(ROOT / "audit_live/audit.json", dest / "audit_live.json")
    shutil.copy2(ROOT / "audit_live/conditions.json", dest / "conditions_live.json")
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
