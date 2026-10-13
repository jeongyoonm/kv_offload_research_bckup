# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare reuse delays and cache-write pressure before and after retiming."""

import json
import time
from pathlib import Path

from experiments.tracelab_arrival_intensity_20261013.prepare import ROOT
from experiments.tracelab_request_slots_20261013 import audit_cache


def main():
    offset = time.time() - time.monotonic()
    cases = {
        "original_baseline_r1.0": Path(
            "/mnt/localnvme/jeongyoon/tracelab_request_slots_20261013"
        ),
        "retimed_baseline_r0.2": ROOT / "episodes/primary_baseline_r0.2",
        "retimed_full_r0.2": ROOT / "episodes/primary_full_r0.2",
        "repaired_baseline_r0.2": ROOT / "episodes/repair_baseline_r0.2",
        "repaired_full_r0.2": ROOT / "episodes/repair_full_r0.2",
    }
    results = {}
    for label, directory in cases.items():
        arm = "full" if "full" in label else "baseline"
        if not (directory / "cells" / arm / "result.json").exists():
            continue
        audit_cache.ROOT = directory
        results[label] = audit_cache.audit(arm, offset)
    document = dict(
        cases=results,
        cpu_capacity_bytes=audit_cache.CPU_CAPACITY_BYTES,
        timing_assumption=(
            "Intervening store volume uses the current wall-minus-monotonic "
            "offset and assumes no wall-clock step. Request-phase CPU hits "
            "are matched directly by request IDs and recorded POST times."
        ),
        limitation=(
            "Write volume is a pressure proxy, not an exact eviction count. "
            "Retiming preserves the catalogue; different finite episodes can "
            "reach different later rounds and therefore different workloads. "
            "CPU-load requests include repeat loads after preemption. Byte "
            "accounting and initial-prefill token counters are distinct."
        ),
    )
    (ROOT / "LOCALITY_COMPARISON.json").write_text(
        json.dumps(document, indent=2) + "\n"
    )
    lines = [
        "TraceLab reuse-locality audit: measurement POST cohorts",
        "",
        (
            "Case | CPU-hit requests % | cached tokens % | followup queue mean s | "
            "tool wait mean s | intervening writes / CPU capacity median | "
            "followups with writes > capacity % | measured D2H TB"
        ),
    ]
    for label, result in results.items():
        measured = result["phases"]["measurement"]
        followups = measured["followup_times"]
        turnover = measured["store_bytes_between_followups_over_cpu_capacity"]
        fraction = measured[
            "fraction_positive_prefix_followups_with_writes_above_cpu_capacity"
        ]
        lines.append(
            f"{label} | {measured['cpu_hit_request_percent']:.2f} | "
            f"{measured['cached_prompt_token_percent']:.2f} | "
            f"{followups['slot_queue_seconds']['mean']:.2f} | "
            f"{followups['tool_wait_seconds']['mean']:.2f} | "
            f"{turnover['median']:.3f} | {fraction * 100:.2f} | "
            f"{measured['observed_d2h_bytes'] / 1e12:.3f}"
        )
    lines += ["", document["timing_assumption"], "", document["limitation"]]
    (ROOT / "LOCALITY_COMPARISON.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
