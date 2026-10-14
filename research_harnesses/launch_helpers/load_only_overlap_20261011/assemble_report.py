"""Build the human report from the closed, validated campaign summary."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / "final_report/report.json"
    report = json.loads(source.read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    assert report["complete"] or args.allow_partial
    assert report["expected_cells"] == 204
    if report["complete"]:
        assert report["cells"] == 204 and report["matched_arm_triplets"] == 68
    assert len(report["provenance"]) == 10
    assert all(
        p["stopped"] for p in report["provenance"].values()
    )
    rows = report["rows"]
    lines = [
        "GLM5.3 TP8: safe transfer readiness and workload ablation",
        f"Accepted campaign: {report['cells']} of 204 planned main cells; "
        f"{report['matched_arm_triplets']} matched arm triplets.",
        f"Planned grid complete: {report['complete']}.",
        f"Report assembled UTC: {datetime.now(timezone.utc)}",
        "",
        "IMPLEMENTATION AND ISOLATED FIX RESULTS",
        "Separate opt-in branches implement consistent load-only readiness targets,",
        "destination-specific H2D compute/zero/COW fences, vectorized bounded block",
        "bookkeeping, and retirement of completed store ownership without another load.",
        "Store source ownership remains protected. Both new runtime options default off.",
        "Original weights, sparse selector and numerical execution are retained.",
        "The final homogeneous sweep resumed 15 missing cells after a diagnostic",
        "trace-capacity failure. Recovery raises opt-in trace capacity from 4096",
        "to 65536; 37 accepted cells and all failed attempts remain preserved.",
        "See TRACE_CAPACITY_RECOVERY.txt and recovery_configuration_audit.json.",
        "",
        "Two 300-request repeats against the previous optimized implementations:",
        "mixed_supported: cached prefixes15552/58368, suffixes832/3072,",
        "100-prefix working set, concurrency32, max model length65536.",
        "Worker: 11.9391 -> 12.4311 req/s (+4.12%).",
        "Full: 12.8183 -> 13.2527 req/s (+3.39%). Mean/p95 TTFT improve 3.34%/11.49%.",
        "Full native inter-compute gaps: 702.5 -> 72.3 ms across the measured stream.",
        "Compute rounds rise 37 -> 39 and predictor MAE rises 5.62 -> 12.14 ms;",
        "the isolated gain is not explained by fewer rounds or better prediction.",
        "Load-only policy alone did not show a convincing throughput gain.",
        "Policy selection precedes the broader tests. See PHASE1_RESULTS.txt.",
        "",
        "BROADER HOMOGENEOUS COVERAGE",
        "Cached prefix: 4K/16K/32K/64K. Uncached prompt suffix: 128/512/2K/8K.",
        "All 16 shapes have concurrency32; outer1/64 coverage is staged as time allows.",
        "The matched shape count is reported below. Three profiled classes also have",
        "2/4/8/16. One 64-request CPU-hit episode per point and arm.",
        "Concurrency 64 is one burst, not a continuously replenished plateau.",
        f"The full illustrative360-cell grid is not claimed; accepted scope is {report['cells']}/204 planned.",
        "",
        "Geometric mean gains across the SAME matched shapes for both optimized arms:",
        "Concurrency  Arm       Shapes  Throughput gain   Mean TTFT reduction  p95 reduction",
    ]
    for concurrency in (1, 32, 64):
        baseline = {
            r["workload"]: r
            for r in rows
            if r["family"] == "homogeneous"
            and r["concurrency"] == concurrency
            and r["arm"] == "baseline"
        }
        assert len(baseline) == 16
        common = set(baseline)
        for candidate_arm in ("worker", "full"):
            common &= {r["workload"] for r in rows
                       if r["family"] == "homogeneous" and r["concurrency"] == concurrency
                       and r["arm"] == candidate_arm}
        assert common
        for arm in ("worker", "full"):
            group = [
                r for r in rows
                if r["family"] == "homogeneous"
                and r["concurrency"] == concurrency and r["arm"] == arm
                and r["workload"] in common
            ]
            assert len(group) == len(common)
            gains = []
            for metric in ("throughput_rps", "mean_ttft_ms", "p95_ttft_ms"):
                ratio = math.exp(sum(
                    math.log(r[metric] / baseline[r["workload"]][metric])
                    for r in group
                ) / len(group))
                gains.append(100 * (ratio - 1 if metric == "throughput_rps" else 1 - ratio))
            lines.append(
                f"{concurrency:11d}  {arm:9s}  {len(common):6d}  {gains[0]:+14.2f}%"
                f"  {gains[1]:+18.2f}%  {gains[2]:+12.2f}%"
            )
    lines += [
        "",
        "Dense curves are in DENSE_RESULTS.txt and dense_all_matched_curves.json.",
        "Balanced C32K/Q512 gains at higher concurrency, but regresses at 2/4/8.",
        "Transfer-heavy C64K/Q128 often regresses versus worker-only. At concurrency4,",
        "full has fewer rounds but paired-dispatch GPU/CPU envelopes average about",
        "142 ms versus 26 ms for singletons. Paired shapes exceed the fixed 128-token",
        "graph-capture ceiling. Its causal contribution has not been controlled.",
        "Candidate-free deadline seals wait about 7 microseconds, not a long hold.",
        "The 8K suffix exceeds the unchanged 4096-token late-candidate restriction;",
        "those cells do not test late admission of their long suffixes.",
        "",
        "HETEROGENEOUS AND EVOLVING CACHE RESULTS",
        "Two 300-request repeats per arm and workload; concurrency32.",
        "Workload                     Arm      req/s   vs baseline   mean TTFT ms  p95 ms",
    ]
    for workload in (
        "heterogeneous_equal", "heterogeneous_transfer_skew",
        "evolving_pool192", "evolving_pool384",
    ):
        for arm in ("baseline", "worker", "full"):
            r = next(r for r in rows if r["workload"] == workload and r["arm"] == arm)
            assert r["repeats"] == 2
            lines.append(
                f"{workload:28s} {arm:8s} {r['throughput_rps']:7.4f}"
                f" {r['throughput_rps_gain_percent']:+11.2f}%"
                f" {r['mean_ttft_ms']:13.2f} {r['p95_ttft_ms']:9.2f}"
            )
    lines += [
        "",
        "Relative to worker-only, full throughput gains are +4.27% (equal),",
        "+5.89% (transfer-skewed), +2.35% (pool192), and -1.26% (pool384).",
        "Equal-mixture full p95 TTFT is slightly worse than worker-only.",
        "Evolving runs retain both caches throughout each measured stream, with",
        "normal promotion, eviction and write-back under identical fixed budgets.",
        "Full pool192 outcomes: CPU65.5%, GPU19.67%, misses14.83%.",
        "Full pool384 outcomes: CPU13.33%, GPU19.67%, misses67.0%.",
        "Any CPU loading counts as a CPU hit, including partial GPU+CPU reuse.",
        "Outcomes describe the serving path, not arrival-time residency.",
        "The first pool384 full cell computes 2.44% more uncached prompt tokens",
        "than worker-only, while computed-token throughput is nearly identical.",
        "This supports a recomputation explanation, not a controlled causal claim.",
        "",
        "CORRECTNESS, MEASUREMENT AND LIMITATIONS",
        "633 scheduler CPU tests passed (22 skipped); 16 native GPU tests and",
        "ownership/API/timing checks passed. The separate TP8 byte audit verified",
        "580,666,982,400 bytes on all8 ranks, with zero mismatches/fingerprint changes.",
        "Its synchronous validation is excluded from throughput/preparation claims",
        "and does not establish arbitrary-workload model accuracy.",
        "All broad cells validate exact restored CPU prefixes where applicable,",
        "independent per-rank/aggregate transfer metrics, no preemptions, and",
        "prompt=cached+computed input tokens. Full predictor state is unchanged.",
        "Final all-rank preparation and archive/process checks are documented in",
        "completion_audit.json; observed entry coverage is distinct from prepared usage.",
        "",
        "Every arm shares private CPU-cache initialization outside measurement.",
        "Baseline uses ordinary host I/O; worker adds async chunked submission,",
        "early loads and store/allocator safeguards; full also adds preparation/sealing.",
        "CPU budget2,130,300,354,560 bytes server-wide; GPU KV96GiB/rank.",
        "Optimized H2D8MiB/depth2, D2H1GiB; predictor frozen_A, lead50ms, idle target4.",
        "Model length81920, max batched tokens61440, max sequences64, output1 token.",
        "Controlled TTFT = first token - max(client start, initial cohort release).",
        "Initial paused assembly, priming, warmup and post-cell drain/reset are excluded.",
        "Later closed-loop requests retain ordinary client start clocks.",
        "Means/ranges are descriptive: one homogeneous repeat and two mixed repeats.",
        "GPU forward envelopes include CPU launch holes; CPU and GPU envelopes",
        "overlap and cannot be added as independent costs. Baseline native GPU gaps",
        "are unobserved. Exposed load-only hold includes necessary DMA waiting",
        "and is not all removable overhead. Missing native readiness gives bounds",
        "instead of an imputed exact admission ratio.",
        "",
        f"Source summary: {source}",
        f"Scientific PNG/PDF plots and CSV: {ROOT / 'final_report'}",
        f"Code/data archives and manifests: {ROOT / 'archives'}",
        "See LATENCY_DEFINITION.txt, PHASE1_RESULTS.txt, EVOLVING_RESULTS.txt,",
        "HETEROGENEOUS_RESULTS.txt and final accounting/coverage audits for detail.",
    ]
    if not report["complete"]:
        audit = json.loads((ROOT / "final_measurement_audit.json").read_text())
        lines += ["", "UNFINISHED PLANNED CELLS", "Deadline-limited staged coverage; no imputation or replacement of results."]
        lines += [str(key) for key in audit["missing_cells"]]
    (ROOT / "REPORT.txt").write_text("\n".join(lines) + "\n")
    print(ROOT / "REPORT.txt")


if __name__ == "__main__":
    main()
