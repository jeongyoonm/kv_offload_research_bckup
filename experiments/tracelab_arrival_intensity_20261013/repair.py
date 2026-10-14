# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rerun both arms with matched short HTTP idle timeout after transport rejection."""

import hashlib
import json
import signal
import time
from pathlib import Path

from experiments.tracelab_arrival_intensity_20261013 import campaign as base
from experiments.tracelab_arrival_intensity_20261013.prepare import ROOT


def main():
    replay = base.replay
    replay.ROOT, replay.SOURCE = ROOT, base.SOURCE
    replay.BINARY = ROOT / "build_keepalive/release/session_runner"
    replay.RUNNER = Path(
        "/mnt/localnvme/jeongyoon/tracelab_arrival_intensity_src_20261013"
    )
    replay.setup()
    signal.signal(signal.SIGALRM, base.timeout_handler)
    signal.alarm(max(1, int(base.DEADLINE - time.time())))
    assert json.loads((ROOT / "keepalive_checks/result.json").read_text())["passed"]
    base.save(
        ROOT / "repaired_client_provenance.json",
        dict(
            runner_binary=str(replay.BINARY),
            runner_sha256=hashlib.sha256(replay.BINARY.read_bytes()).hexdigest(),
            runner_source=str(replay.RUNNER),
            http_pool_idle_timeout_seconds=1,
            retries=False,
            rate=0.2,
            warmup_seconds=600,
            measurement_seconds=900,
            deadline=base.DEADLINE,
            rationale=(
                "Reject failed transport episode. Use the same short idle "
                "timeout in both arms and preserve all trace contents/waits. "
                "Use 15min measurement after 10min warmup to reserve time "
                "for both reruns, initialization, and draining within 3h."
            ),
        ),
    )
    root = process = None
    results = []
    try:
        for arm, label in (("full", "fk"), ("baseline", "bk")):
            root, process = base.launch(arm, label)
            results.append(
                base.cell(root, process, arm, "0.2", f"repair_{arm}_r0.2", 600, 900)
            )
            base.stop(root, process)
            root = process = None
        base.save(
            ROOT / "completed.json",
            dict(
                time=time.time(),
                rate=0.2,
                matched_family="repair",
                repaired_client=True,
                results=[result["episode"] for result in results],
                prior_calibration_results=[
                    "pilot_baseline_r0.1",
                    "pilot_baseline_r0.2",
                    "primary_baseline_r0.2",
                ],
                rejected="rejected/primary_full_r0.2",
            ),
        )
        base.save(ROOT / "progress.json", dict(complete=True, phase="complete"))
        base.log("repaired_comparison_complete", results=len(results))
    except BaseException as error:
        base.save(
            ROOT / "failure.json",
            dict(time=time.time(), error=repr(error), completed_episodes=results),
        )
        base.log("repaired_comparison_failed", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        if process is not None:
            base.stop(root, process)


if __name__ == "__main__":
    main()
