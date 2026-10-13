# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build comparison plots and verify the completed capacity-sweep deliverables."""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

from experiments.tracelab_arrival_intensity_20261013.prepare import content_hash
from experiments.tracelab_capacity_sweep_20261013 import analyze

ROOT = analyze.ROOT


def output_iterations(result, samples):
    begin = result["replay_start_timestamp"] + result["warmup_seconds"] + 60
    end = result["replay_start_timestamp"] + 1500
    selected = []
    for sample in samples:
        if sample["episode"] != result["episode"] or not begin <= sample["time"] <= end:
            continue
        counters = {
            field: sum(
                value
                for name, value in sample["values"].items()
                if name.startswith("vllm:iteration_tokens_total_" + field + "{")
            )
            for field in ("count", "sum")
        }
        if counters["count"]:
            selected.append((sample["time"], counters))
    if len(selected) < 2:
        return dict(available=False, reason="Insufficient iteration counter samples")
    first, last = selected[0], selected[-1]
    seconds = last[0] - first[0]
    count = last[1]["count"] - first[1]["count"]
    tokens = last[1]["sum"] - first[1]["sum"]
    assert count >= 0 and tokens >= 0
    return dict(
        available=True,
        start=first[0],
        end=last[0],
        observed_seconds=seconds,
        target_seconds=840,
        sufficient_coverage=seconds >= 800,
        output_iterations=count,
        reported_computed_plus_generated_tokens=tokens,
        iterations_per_second=count / seconds,
        reported_tokens_per_iteration=tokens / count if count else None,
        reported_tokens_per_second=tokens / seconds,
        scope=(
            "Existing API iteration histogram; prompt compute is accounted on "
            "first output, not per chunk. These are output-side iterations, not "
            "a substitute for native GPU-forward counts or scheduled batch shape. "
            "Common target is the last840s of the900s measurement because iteration "
            "sampling was added35s into the first baseline measurement. Actual "
            "endpoints lie inside that target and are reported without extrapolation."
        ),
    )


def plot(pairs, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = sorted({pair["rate"] for pair in pairs})
    fig, axes = plt.subplots(
        2, len(rates), figsize=(4.3 * len(rates), 6.8), squeeze=False
    )
    for column, rate in enumerate(rates):
        selected = sorted(
            (pair for pair in pairs if pair["rate"] == rate and pair["repeat"] == 0),
            key=lambda pair: pair["concurrency"],
        )
        x = [pair["concurrency"] for pair in selected]
        for arm, label, color, marker in (
            ("baseline", "Baseline", "#597594", "o"),
            ("full", "Full prototype", "#bf6434", "s"),
        ):
            axes[0, column].plot(
                x,
                [pair[arm]["throughput_rps"] for pair in selected],
                label=label,
                color=color,
                marker=marker,
                linewidth=1.8,
            )
        axes[0, column].set_title(f"{rate:.2f} new sessions/s")
        gains = [pair["throughput_gain_pct"] for pair in selected]
        axes[1, column].plot(x, gains, color="#bf6434", marker="s", linewidth=1.8)
        axes[1, column].axhline(0, color="0.55", linewidth=0.8)
        for a, b in zip(x, gains):
            axes[1, column].annotate(
                f"{b:+.1f}%",
                (a, b),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
            )
        for row in (0, 1):
            axes[row, column].set_xticks(x)
            axes[row, column].set_xlabel("In-flight model calls")
            axes[row, column].grid(axis="y", alpha=0.2)
            axes[row, column].spines[["top", "right"]].set_visible(False)
        axes[0, column].set_ylim(bottom=0)
        if gains:
            axes[1, column].set_ylim(min(0, min(gains)) - 3, max(0, max(gains)) + 5)
    axes[0, 0].set_ylabel("Completed requests/s")
    axes[1, 0].set_ylabel("Throughput gain (%)")
    axes[0, -1].legend(frameon=False, fontsize=9)
    fig.suptitle("TraceLab · GLM 5.3 NVFP4 · TP8", fontsize=13)
    fig.text(
        0.5,
        0.015,
        "600s warmup + 900s measurement; fixed cache budgets.\n"
        "Point estimates; optional confirmation reported separately.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(destination / f"capacity_sweep.{extension}", dpi=180)
    plt.close(fig)


def finalize():
    assert (ROOT / "completed.json").exists(), "Campaign has not completed"
    assert not (ROOT / "failure.json").exists(), "Inspect and resolve campaign failure"
    pairs = analyze.write_report()
    results = analyze.results()
    plan = json.loads((ROOT / "plan.json").read_text())
    keys = {(pair["rate"], pair["concurrency"], pair["repeat"]) for pair in pairs}
    assert {(0.2, limit, 0) for limit in (32, 40, 48)} <= keys
    paired_episodes = {
        pair[arm]["episode"] for pair in pairs for arm in ("baseline", "full")
    }
    unpaired = [r["episode"] for r in results if r["episode"] not in paired_episodes]
    rejected = []
    for path in sorted((ROOT / "rejected").glob("*/exclusion.json")):
        rejected.append(dict(path=str(path), **json.loads(path.read_text())))
    assert len(results) == 2 * len(pairs) + len(unpaired)
    assert all(
        result["accepted"] and all(result["checks"].values()) for result in results
    )
    assert all(all(result["audit"]["accounting_checks"].values()) for result in results)
    manifest = json.loads((analyze.PREVIOUS / "dataset_manifest.json").read_text())
    verified_hashes = {}
    for rate in ("0.15", "0.2", "0.25"):
        path = ROOT / "workloads" / f"rate_{rate}.csv"
        observed = content_hash(path)
        assert observed == manifest["original_content"]
        verified_hashes[rate] = observed
    registered = {}
    connector_by_arm = {}
    for directory in sorted((ROOT / "servers").iterdir()):
        stopped = json.loads((directory / "stopped.json").read_text())
        assert stopped["returncode"] == 0
        config = json.loads((directory / "server/config.json").read_text())
        assert config["kv_connector_extra_config"]["cpu_bytes_to_use"] == 3028287488000
        arm = "baseline" if directory.name.startswith("b") else "full"
        if arm in connector_by_arm:
            assert config == connector_by_arm[arm]
        connector_by_arm[arm] = config
        command = json.loads((directory / "server/command.json").read_text())
        for flag, expected in (
            ("--tensor-parallel-size", "8"),
            ("--max-model-len", "1048576"),
            ("--max-num-batched-tokens", "61440"),
            ("--max-num-seqs", "64"),
            ("--kv-cache-memory-bytes", "103079215104"),
            ("--kv-cache-dtype", "fp8_e4m3"),
        ):
            assert command[command.index(flag) + 1] == expected
        assert command[command.index("serve") + 1] == (
            "/mnt/localnvme/huggingface/hub/models--Inferact--GLM-5.3-NVFP4/"
            "snapshots/ce67b36f3669192b5bb233819f0fda6c8a9837f8"
        )
        registrations = json.loads((directory / "registration_audit.json").read_text())
        assert {int(rank) for rank, _, code in registrations if int(code) == 0} == set(
            range(8)
        )
        registered[directory.name] = sum(int(size) for _, size, _ in registrations)
        assert registered[directory.name] == 3028011286528
    policy_by_arm = {}
    for result in results:
        directory = ROOT / "episodes" / result["episode"] / "cells" / result["arm"]
        spec = json.loads((directory / "spec.json").read_text())
        command = spec["command"]
        assert spec["request_slots"] == result["measurement_request_slots"]
        assert (
            int(command[command.index("--max-in-flight-requests") + 1])
            == (result["measurement_request_slots"])
        )
        assert command[command.index("--run-seconds") + 1] == "1500"
        assert spec["warmup_seconds"] == 600 and spec["measurement_seconds"] == 900
        assert not spec["tool_wait_scaling"] and spec["session_cap"] is None
        assert spec["cpu_cache_bytes"] == plan["cpu_cache_bytes"]
        assert spec["gpu_cache_bytes_per_rank"] == plan["gpu_cache_bytes_per_rank"]
        assert spec["predictor_sha256"] == plan["predictor_sha256"]
        arm = result["arm"]
        if arm in policy_by_arm:
            assert spec["policy"] == policy_by_arm[arm]
        policy_by_arm[arm] = spec["policy"]
        assert (
            hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()
            == (plan["runner_sha256"])
        )
    source = Path(__file__).resolve().parents[2]
    assert not subprocess.check_output(
        ["git", "diff", "595b8bf", "--", "vllm"], cwd=source, text=True
    )
    gpu_pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).split()
    owned_gpu_pids = []
    aliases = [
        json.loads((directory / "path_alias.json").read_text())["alias"]
        for directory in (ROOT / "servers").iterdir()
    ]
    for pid in gpu_pids:
        environment = (Path("/proc") / pid / "environ").read_bytes()
        if any(alias.encode() in environment for alias in aliases):
            owned_gpu_pids.append(pid)
    assert not owned_gpu_pids
    plot(pairs, ROOT / "plots")
    summary = dict(
        time=time.time(),
        matched_pairs=len(pairs),
        cells=len(results),
        unpaired_cells=unpaired,
        rejected_attempts=rejected,
        checks_verified=True,
        trace_content_checks=verified_hashes,
        registered_cpu_bytes=registered,
        all_owned_servers_stopped=True,
        all_gpus_idle=not gpu_pids,
        owned_gpu_pids=owned_gpu_pids,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        serving_source_unchanged=True,
        comparisons=[
            {
                key: value
                for key, value in pair.items()
                if key not in ("baseline", "full")
            }
            for pair in pairs
        ],
        allocation_end=json.loads((ROOT / "allocation.json").read_text())["deadline"],
    )
    samples = [json.loads(line) for line in (ROOT / "cache_samples.jsonl").open()]
    summary["output_iteration_diagnostics"] = {
        result["episode"]: output_iterations(result, samples) for result in results
    }
    (ROOT / "output_iteration_diagnostics.json").write_text(
        json.dumps(summary["output_iteration_diagnostics"], indent=2) + "\n"
    )
    base_pairs = [pair for pair in pairs if pair["rate"] == 0.2 and pair["repeat"] == 0]
    original = json.loads(
        (analyze.PREVIOUS / "results/repair_baseline_r0.2.json").read_text()
    )
    original_full = json.loads(
        (analyze.PREVIOUS / "results/repair_full_r0.2.json").read_text()
    )
    lines = [
        "TraceLab capacity sweep: verified final analysis",
        "",
        f"Completed {len(results)} accepted cells / {len(pairs)} matched pairs.",
        f"Unpaired accepted controls (no inferred gain): {unpaired}",
        f"Rejected incomplete attempts (not performance samples): {rejected}",
        "All submitted requests drained, success/token/retirement/rank accounting",
        "checks passed; physical H2D/D2H bytes matched server counters. All owned",
        "servers stopped. Workload nonarrival contents rehashed and verified.",
        "",
        "At 0.2 new sessions/s:",
        "Calls | baseline req/s | full req/s | gain % | CPU-load requests B/F % |",
        "        mean HTTP TTFT B/F ms | p95 HTTP TTFT B/F ms | preemptions B/F",
    ]
    for pair in base_pairs:
        b, f = pair["baseline"], pair["full"]
        lines.append(
            f"{pair['concurrency']} | {b['throughput_rps']:.5f} | "
            f"{f['throughput_rps']:.5f} | {pair['throughput_gain_pct']:+.2f} | "
            f"{b['cache_profile']['measurement']['cpu_hit_request_percent']:.2f}/"
            f"{f['cache_profile']['measurement']['cpu_hit_request_percent']:.2f} | "
            f"{b['ttft_ms']['mean']:.2f}/{f['ttft_ms']['mean']:.2f} | "
            f"{b['ttft_ms']['p95']:.2f}/{f['ttft_ms']['p95']:.2f} | "
            f"{b['metric_deltas'].get('vllm:num_preemptions_total', 0):.0f}/"
            f"{f['metric_deltas'].get('vllm:num_preemptions_total', 0):.0f}"
        )
    old_gain = (original_full["throughput_rps"] / original["throughput_rps"] - 1) * 100
    lines += [
        "",
        (
            "Earlier 32-call pair (separate, fresh server each cell): "
            f"{original['throughput_rps']:.5f} -> "
            f"{original_full['throughput_rps']:.5f} "
            f"req/s (+{old_gain:.2f}%)."
        ),
        "Current experiment reuses servers within each arm group but resets logical",
        "caches and applies the same 600s warmup to every cell. Compare replication",
        "as sensitivity evidence; do not treat a selected confirmation as unbiased.",
        "",
        "All arrival-rate and replication results: REPORT.txt / comparisons.json.",
        "Per-cell realized provider/input/prefix/append/recomputed/output/tool-wait",
        "mix and available GPU execution envelopes: episodes/*/cells/*/audit.json.",
        "Physical transfers, late admissions and preemptions cover whole episodes;",
        "latency and hit ratios cover measurement POSTs. Do not divide counts across",
        "different cohorts to infer an admission opportunity recovery ratio.",
        "",
        "Limits: synthetic token content follows native TraceLab reconstruction,",
        "not original source prompt identities. Recorded token lengths and tool waits",
        "are preserved; zero append/output lengths use native minimum-one handling.",
        "Calls replay with actual generated token IDs carried forward, so observed",
        "output content/routing can differ across arms. Faster closed-loop progress",
        "advances farther through sessions and changes the finite-window mix.",
        "Growing queues are not indefinite open-loop stability. Point estimates",
        "do not establish a confidence interval or an asymptotic throughput ceiling.",
        "GPU utilization is a busy-time sample, not peak compute efficiency.",
        "Full-only native execution envelopes are not a paired kernel-level trace.",
    ]
    summary["mean_cell_wall_seconds"] = statistics.mean(
        result["cell_wall_seconds"] for result in results
    )
    (ROOT / "FINAL_ANALYSIS.txt").write_text("\n".join(lines) + "\n")
    (ROOT / "FINAL_ANALYSIS.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.watch:
        while not any(
            (ROOT / name).exists() for name in ("completed.json", "failure.json")
        ):
            time.sleep(30)
        owner = json.loads((ROOT / "owner.json").read_text())
        process = Path("/proc") / str(owner["pid"]) / "cmdline"
        while process.exists():
            try:
                if b"tracelab_capacity_sweep_20261013." not in process.read_bytes():
                    break
            except FileNotFoundError:
                break
            time.sleep(2)
    print(json.dumps(finalize(), indent=2), flush=True)


if __name__ == "__main__":
    main()
