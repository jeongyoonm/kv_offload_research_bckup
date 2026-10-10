# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare frozen direct and residual estimates on identical unseen live frames."""

import argparse
import importlib.util
import json
from pathlib import Path

from experiments.shape_calibration_20261010.analyze import cell_audit, error_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "offline_shape_calibration", source / "vllm/v1/kv_offload/seal_calibration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    states = {
        name: json.loads(path.read_text())["calibration"]
        for name, path in (("direct", args.direct), ("residual", args.residual))
    }
    models = {k: module.BatchShapeCalibration(v) for k, v in states.items()}
    episodes = []
    for path in sorted((args.root / "live").glob("*/*/result.json")):
        result = json.loads(path.read_text())
        if result["repeat"] == "warmup":
            continue
        _, rows = cell_audit(path.parent, result)
        rows = [r for r in rows if r["adjacent_compute"]]
        errors = {name: [] for name in models}
        fallback = dict.fromkeys(models, 0)
        clipping = dict.fromkeys(models, 0)
        for row in rows:
            x, raw = row["features"], round(row["raw_ms"] * 1e6)
            for name, model in models.items():
                bias, unsupported = model.bias_ns(raw, x, 90_000_000)
                errors[name].append(max(1, raw - bias) / 1e6 - row["true_ms"])
                fallback[name] += unsupported
                if not unsupported:
                    query = module.shape_query(raw, x, model.features)
                    unbounded = float(query @ model.coefficients)
                    clipping[name] += (
                        not model.bounds[0] <= unbounded <= model.bounds[1]
                    )
        episodes.append(
            dict(
                workload=result["workload"],
                arm=result["label"],
                repeat=result["repeat"],
                estimates={
                    name: dict(
                        **error_metrics(v),
                        fallback=fallback[name],
                        clipped=clipping[name],
                    )
                    for name, v in errors.items()
                },
            )
        )
    payload = dict(
        episodes=episodes,
        scope="Frozen models, same unseen observed forecasts, no new "
        "fitting. This isolates estimate error on matched frames; "
        "it does not estimate serving throughput under changed seals.",
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
