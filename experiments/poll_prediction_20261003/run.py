"""Matched post-ack polling, preparation reserve, and hybrid wait experiments."""

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/localnvme/glm53_poll_prediction_20261003")
SOURCE = Path(__file__).resolve().parents[2]
HELPERS = SOURCE / "experiments/adaptive_seal_20261002"
sys.path.insert(0, str(HELPERS))
spec = importlib.util.spec_from_file_location("adaptive_run", HELPERS / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
run.ROOT = ROOT
run.SOURCE = SOURCE
run.POLICIES = {"worker": run.PRIME, "fixed": run.COMMON}
for ms in (1, 2, 5, 10):
    for family, policy in (
        ("readiness", "event_all_load4"),
        ("prediction", "predictive_conditioned"),
    ):
        run.POLICIES[f"{family}_{ms}ms"] = dict(
            run.POLICIES.get(policy, {}),
            **dict(
                run.COMMON,
                late_seal_policy="event" if family == "readiness" else "predictive",
                late_seal_target_candidates=16,
                late_seal_load_only_target_candidates=4 if family == "readiness" else 0,
                late_seal_condition_preparation=family == "prediction",
                late_seal_adaptive_poll_us=ms * 1000,
            ),
        )
for ms in (1, 5):
    run.POLICIES[f"recent_{ms}ms"] = dict(
        run.POLICIES[f"prediction_{ms}ms"], late_seal_recent_preparation=True
    )
for ms in (20, 50):
    run.POLICIES[f"hybrid_{ms}ms"] = dict(
        run.POLICIES["recent_1ms"],
        late_seal_adaptive_poll_us=2000,
        late_seal_hybrid_wait_us=ms * 1000,
        late_seal_hybrid_guard_us=50_000,
    )

import correctness  # noqa: E402

original_check = correctness.check


def check(directory, policies, *args, **kwargs):
    # Probe each changed mechanism and the largest interval before performance.
    selected = {
        k: v
        for k, v in policies.items()
        if k
        in {
            "worker",
            "fixed",
            "readiness_10ms",
            "prediction_10ms",
            "recent_1ms",
            "hybrid_50ms",
        }
    }
    kwargs["prefix_len"] = 11776
    original_check(directory, selected, *args, **kwargs)


correctness.check = check
if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "policies.json").write_text(json.dumps(run.POLICIES, indent=2) + "\n")
    with (ROOT / "process_log.jsonl").open("a") as log:
        log.write(
            json.dumps(
                {
                    "event": "driver_started",
                    "time": time.time(),
                    "argv": sys.argv,
                    "source": str(SOURCE),
                }
            )
            + "\n"
        )
    try:
        run.main()
    except BaseException as error:
        with (ROOT / "process_log.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "event": "driver_failed",
                        "time": time.time(),
                        "error": repr(error),
                    }
                )
                + "\n"
            )
        raise
    else:
        with (ROOT / "process_log.jsonl").open("a") as log:
            log.write(
                json.dumps({"event": "driver_completed", "time": time.time()}) + "\n"
            )
