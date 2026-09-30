# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small, immutable, experiment-only completion model; no device operations."""

import math
from dataclasses import dataclass

NS_MS = 1_000_000


@dataclass(frozen=True)
class CompletionEstimate:
    completion_ns: float | None
    model: str
    reason: str


def features(shape, requests, dispatch_ns, now_ns, ranks, *, progress=False):
    """Construct features using only the rank snapshot observed at decision time."""
    if not requests or len(requests) != shape[1]:
        raise ValueError("missing per-request shape")
    if any(len(r) != 2 or r[0] <= 0 or r[1] < 0 for r in requests):
        raise ValueError("invalid per-request shape")
    tokens = sum(r[0] for r in requests)
    if tokens != shape[2] or now_ns < dispatch_ns:
        raise ValueError("inconsistent batch shape or decision timestamp")
    # Normalize units before polynomial expansion; fit stores further scaling.
    x = [
        float(len(requests)),
        tokens / 1024,
        sum(n * c + n * (n + 1) / 2 for n, c in requests) / 1e6,
        sum(c for _, c in requests) / 1024,
        max(c for _, c in requests) / 1024,
        sum(n > 1 for n, _ in requests),
    ]
    base = list(x)
    x += [a * b for i, a in enumerate(base) for b in base[i:]]
    if progress:
        entries: list[float] = []
        prepared: list[float] = []
        for r in ranks:
            if r is None:
                continue
            if len(r) < 9 or r[6] > now_ns:
                raise ValueError("future or malformed rank snapshot")
            for index, values in ((1, entries), (2, prepared)):
                if r[index]:
                    if not dispatch_ns <= r[index] <= now_ns:
                        raise ValueError("invalid progress timestamp")
                    values.append((r[index] - dispatch_ns) / NS_MS)
        x += [
            (now_ns - dispatch_ns) / NS_MS,
            len(entries) / max(1, len(ranks)),
            len(prepared) / max(1, len(ranks)),
            min(entries, default=0),
            max(entries, default=0),
            min(prepared, default=0),
            max(prepared, default=0),
        ]
    return x


def evaluate(model, values):
    if len(values) != len(model["coefficients"]):
        raise ValueError("feature version mismatch")
    return model["intercept"] + sum(
        coefficient * (value - mean) / scale
        for coefficient, value, mean, scale in zip(
            model["coefficients"], values, model["mean"], model["scale"]
        )
    )


class BatchPredictor:
    def __init__(self, data):
        self.data = data
        self.reserve_ns = data["reserve_ns"]
        domain = data["domain"]
        if (
            data.get("feature_version") != 1
            or data.get("selected") not in ("shape", "progress")
            or not math.isfinite(self.reserve_ns)
            or self.reserve_ns < 0
            or domain["phase"] != "prefill"
            or domain["new_tokens"] != 1024
            or domain["context"] < 0
            or domain["max_requests"] < 1
            or domain["ranks"] != 8
        ):
            raise ValueError("invalid batch predictor domain")
        for name, model in data["models"].items():
            size = 34 if name == "progress" else 27
            if any(len(model[k]) != size for k in ("coefficients", "mean", "scale")):
                raise ValueError("invalid model dimensions")
            values = [model["intercept"], model["correction_ms"]]
            values += model["coefficients"] + model["mean"] + model["scale"]
            if not all(math.isfinite(v) for v in values) or min(model["scale"]) <= 0:
                raise ValueError("invalid model coefficients")
        if not {"shape", "progress", "prepared"} <= data["models"].keys():
            raise ValueError("missing fallback models")

    def estimate_completion(self, data, now, ranks):
        shape = data["shape"]
        requests = data.get("request_shapes", [])
        domain = self.data["domain"]
        if (
            shape[0] != domain["phase"]
            or not 1 <= shape[1] <= domain["max_requests"]
            or not requests
            or any(r != [domain["new_tokens"], domain["context"]] for r in requests)
        ):
            return CompletionEstimate(None, "none", "outside_domain")
        region = str(1 << (shape[1] - 1).bit_length())
        supports = self.data["support"].get(region, {})
        name = self.data["selected"]
        reason = "dispatch_fallback"
        if not supports.get("dispatch", False):
            name = f"target_{shape[1]}"
            if name not in self.data["models"]:
                return CompletionEstimate(
                    None, "none", "insufficient_dispatch_evidence"
                )
            reason = "targeted_shape_fallback"
        anchor = data["dispatch_ns"]
        if (
            name == "progress"
            and len(ranks) == domain["ranks"]
            and all(
                r is not None and len(r) >= 9 and anchor <= r[1] <= r[2] <= r[6] <= now
                for r in ranks
            )
            and supports.get("prepared", False)
        ):
            name = "prepared"
            anchor = max(r[2] for r in ranks)
            reason = "all_ranks_prepared"
        try:
            x = features(
                shape,
                requests,
                data["dispatch_ns"],
                now,
                ranks,
                progress=name == "progress",
            )
            model = self.data["models"][name]
            duration = evaluate(model, x) + model["correction_ms"]
            if not math.isfinite(duration) or duration <= 0:
                return CompletionEstimate(None, name, "invalid_duration")
            return CompletionEstimate(anchor + duration * NS_MS, name, reason)
        except (ValueError, TypeError, IndexError):
            return CompletionEstimate(None, name, "invalid_features")
