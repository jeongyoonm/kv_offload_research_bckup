"""Write final numerical tables only after all registered phases complete."""

import argparse
import json
import statistics
from pathlib import Path

CASES = ["p8192_f95_h100", "p61440_f95_h100", "p61440_f99_h100"]
LABELS = ["8K / 95%", "60K / 95%", "60K / 99%"]
PHASES = ["screen", "confirmation", "predicted_readiness"]


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def artifact_link(label, path):
    return f"- [{label}]({path})"


def main(root):
    complete = [json.loads((root / p / "complete.json").read_text()) for p in PHASES]
    if any(d["accepted"] != d["cells"] for d in complete):
        raise RuntimeError("Rejected cells require investigation before final findings")
    rows = json.loads((root / "summary/cells.json").read_text())
    lookup = {(r["phase"], r["workload"], r["policy"]): r for r in rows}
    policies = [
        "worker",
        "fixed",
        "readiness_1ms",
        "readiness_2ms",
        "readiness_5ms",
        "readiness_10ms",
        "prediction_1ms",
        "prediction_2ms",
        "prediction_5ms",
        "prediction_10ms",
        "recent_1ms",
        "recent_5ms",
        "hybrid_20ms",
        "hybrid_50ms",
    ]
    cells = sum(d["cells"] for d in complete)
    text = [
        "# GLM 5.3 TP8: polling and prediction experiments",
        "",
        (
            f"Completed {cells} accepted replay cells. "
            "Concurrency 32, one output token, "
            "ten candidate slots, FP8 KV, existing worker optimizations enabled. "
            "8K cells have 300 requests; 60K cells have 80. "
            "All replay requests have a CPU "
            "prefix hit; 95%/99% denotes the cached fraction of each prompt, "
            "rounded to "
            "64-token blocks. Worker-only retains async H2D, pipelining and early "
            "submission with late admission disabled; it is not unoptimized vLLM."
        ),
        "",
        "## Polling sweep: single screening seed",
        "",
        (
            "Throughput in requests/s. Each point is one replay. "
            "This is not a confidence interval. "
            "Pre-ACK polling remains 25 us; only adaptive post-ACK polling changes."
        ),
        "",
        table(
            ["Policy", *LABELS],
            [
                [p, *[f"{lookup[('screen', c, p)]['throughput']:.3f}" for c in CASES]]
                for p in policies
            ],
        ),
        "",
        "## Matched screening and fresh-seed confirmation",
        "",
        (
            "Cells show arithmetic mean requests/s / mean TTFT in ms, across seed 55 "
            "and seed 56. Confirmation reverses selected policy order. Two seeds do "
            "not establish a statistically reliable optimum."
        ),
        "",
    ]
    selected = [p for p in policies if ("confirmation", CASES[0], p) in lookup]
    combined = []
    for p in selected:
        values = [p]
        for c in CASES:
            pair = [lookup[(phase, c, p)] for phase in ["screen", "confirmation"]]
            values.append(
                f"{statistics.mean(r['throughput'] for r in pair):.3f} / "
                f"{statistics.mean(r['mean_ttft_ms'] for r in pair):.0f}"
            )
        combined.append(values)
    text += [
        table(["Policy", *LABELS], combined),
        "",
        "## Predicted initial wait followed by readiness polling",
        "",
        (
            "Separate server; matched seed 56. Up to 50 ms coarse sleeps, followed "
            "by 2 ms polling after the first predicted deadline minus 50 ms. The "
            "forecast controls checking frequency and cannot seal this variant. "
            "Seal on all active published candidates ready+acknowledged, preceding "
            "physical compute completion, or 2 s safety cap; load-only target=min(4, "
            "active candidates). Every wake refreshes progress; conditions during "
            "a sleep wait for the next wake. No forecast falls back to fine polling."
        ),
        "",
        table(
            ["Policy", *LABELS],
            [
                [
                    p,
                    *[
                        f"{lookup[('predicted_readiness', c, p)]['throughput']:.3f} / "
                        f"{lookup[('predicted_readiness', c, p)]['mean_ttft_ms']:.0f}"
                        for c in CASES
                    ],
                ]
                for p in [
                    "fixed",
                    "prediction_1ms",
                    "readiness_2ms",
                    "predicted_readiness_50ms",
                ]
            ],
        ),
        "",
        "## Interpretation limits and verification",
        "",
        (
            "- Each replay resets and independently primes the CPU cache, then "
            "resets GPU prefix state. Raw cache positions, exact transfer bytes, "
            "uncached token work, output counts, compilation/errors and GPU process "
            "ownership are validated. No replay may be silently dropped."
        ),
        (
            "- Known-answer and distinct-fact probes are stored in each phase. "
            "The new branch passes 69 CPU protocol/predictor tests and 12 selected "
            "option tests. Earlier polling/hybrid gates passed 67 CPU checks and 19 "
            "option checks. Two broader inherited fixture/artifact failures are "
            "retained and reproduced on the unchanged old evaluation source."
        ),
        (
            "- Prediction uses past observations only, but training history persists "
            "across policies and their priming passes. Reverse order helps assess "
            "sensitivity; it does not give each predictor an identical training state."
        ),
        (
            "- The recent reserve filters context regime and nearby batch shape, "
            "then takes the nearest 8 of the last 32 eligible samples, 75th percentile "
            "plus 10 ms. Context regime is not a definitive CPU-cache-origin flag. "
            "No nearby sample falls back to 10 ms and can underestimate preparation."
        ),
        (
            "- Polling CPU covers thread time in iterations leading to adaptive "
            "sleep; it excludes final/pre-ACK/fixed iterations and is not total "
            "engine CPU. Exposed hold is an all-rank timestamp bound, not an "
            "additive counterfactual throughput estimate. Admission denominators "
            "vary with resulting batch timing."
        ),
        (
            "- These are prefill/TTFT experiments with one output token; they "
            "do not establish performance for long generation or other concurrencies."
        ),
        "- All new policies remain opt-in and disabled by default.",
        "",
        "## Artifacts",
        "",
        artifact_link("Full tables", root / "summary/report.html"),
        artifact_link("Polling curves", root / "summary/polling_screen.png"),
        artifact_link(
            "Prediction versus readiness",
            root / "summary/predicted_readiness_comparison.png",
        ),
        f"- Process log: [{root / 'process_log.jsonl'}]({root / 'process_log.jsonl'})",
        artifact_link("Branches and native SHA manifests", root / "branches.json"),
        artifact_link(
            "Forecast diagnostics", root / "summary/prediction_diagnostics.csv"
        ),
        "",
    ]
    path = root / "summary/FINDINGS.md"
    path.write_text("\n".join(text))
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
