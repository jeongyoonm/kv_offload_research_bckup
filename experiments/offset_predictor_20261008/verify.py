# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify serving exports and elapsed independence on held and fallback inputs."""

import copy
import importlib.util
import json
from pathlib import Path

import audit
from threadpoolctl import threadpool_limits

ROOT = Path("/mnt/localnvme/glm53_offset_predictor_20261008")
spec = importlib.util.spec_from_file_location(
    "offset_study", Path(__file__).with_name("study.py")
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def main():
    _, rows = study.live_rows()
    report = {}
    with threadpool_limits(limits=1):
        for name in ["matched_elapsed", "matched_offset"]:
            state = json.loads((ROOT / "models" / f"{name}.json").read_text())
            estimator = audit.ComputeEstimator.from_state_dict(state, state["context"])
            assert estimator.state_dict(state["context"]) == state
            model = estimator.latency_surface
            expected = study.predict(state["latency_surface"]["checkpoint"], rows)
            differences = []
            invariance_checks = 0
            for row, value in zip(rows, expected):
                data = audit.snapshot(
                    row, row["forecast_time_ns"], row["forecast_features"][:10]
                )
                prediction, features = model.predict_with_features(data)
                assert prediction is not None
                differences.append(abs(prediction / 1e6 - value))
                assert differences[-1] < 1e-4
                if name == "matched_offset":
                    start = max(r[3] for r in row["execution_ranks"])
                    for elapsed in [0, 5, 25, 50, 100, 1000]:
                        changed = dict(data, forecast_ns=start + elapsed * 1_000_000)
                        assert model.predict(changed) == prediction
                        # The absolute deadline stays fixed as elapsed increases.
                        deadline = start + prediction - 5_000_000
                        wait = max(0, deadline - changed["forecast_ns"])
                        assert wait == max(
                            0, prediction - elapsed * 1_000_000 - 5_000_000
                        )
                        invariance_checks += 1
                    # Outside-support fallback must also be elapsed independent.
                    outside = features.copy()
                    outside[0] = 1e9
                    baseline = model.predict_features_ms(outside)
                    for elapsed in [0, 5, 50, 1000]:
                        outside[21] = elapsed
                        assert model.predict_features_ms(outside) == baseline
                        invariance_checks += 1
                assert model.predict(dict(data, forecast_ns=1)) is None
                assert (
                    model.predict(
                        dict(data, execution_snapshot=data["execution_snapshot"][:-1])
                    )
                    is None
                )
            if name == "matched_offset":
                malformed = copy.deepcopy(state)
                malformed["latency_surface"]["checkpoint"].pop(
                    "excluded_feature_indices"
                )
                try:
                    audit.ComputeEstimator.from_state_dict(malformed, state["context"])
                except ValueError:
                    pass
                else:
                    raise AssertionError(
                        "Accepted an offset predictor without exclusion metadata"
                    )
            report[name] = dict(
                rows=len(rows),
                maximum_export_difference_ms=max(differences),
                elapsed_invariance_checks=invariance_checks,
                snapshot_round_trip=True,
                causal_snapshot_checks=True,
            )
    study.save("cpu_equivalence.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
