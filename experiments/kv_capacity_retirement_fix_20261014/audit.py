# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize preserved capacity diagnostics and matched serving measurements."""

import collections
import json
import subprocess

import regex as re

from experiments.kv_capacity_retirement_fix_20261014.campaign import ROOT


def main():
    diagnostics = {}
    for server in sorted((ROOT / "servers").iterdir()):
        log = server / "server/server.log"
        if not log.exists():
            continue
        engine_pids = set(re.findall(r"EngineCore[^)]*pid=(\d+)", log.read_text()))
        counts = collections.Counter()
        samples = collections.defaultdict(list)
        preemptions = []
        for pid in engine_pids:
            for path in (server / "server/load_timeline").glob(f"{pid}.*.jsonl"):
                for line in path.open("rb"):
                    if not line.startswith(b'{"event": "capacity_'):
                        continue
                    event = json.loads(line)
                    kind = event["event"]
                    counts[kind] += 1
                    if len(samples[kind]) < 8:
                        samples[kind].append(event)
                    if kind == "capacity_preempt":
                        preemptions.append(event)
        diagnostics[server.name] = dict(
            counts=dict(counts),
            samples=dict(samples),
            preemptions=preemptions,
            source_config=json.loads((server / "server/config.json").read_text()),
            scope="Whole server lifetime; each measured cell resets logical caches",
        )
    (ROOT / "capacity_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    pairs_path = ROOT / "performance_comparisons.json"
    pairs = json.loads(pairs_path.read_text()) if pairs_path.exists() else []
    lines = [
        "KV capacity / retirement fix: GLM5.3 TP8 TraceLab evaluation",
        "Original = recent full prototype before the capacity fix.",
        "600s warmup + 900s measurement; recorded lengths/tools preserved.",
        "Point estimates, not statistical equivalence or causal speedup proofs.",
        "",
        "C | original/fixed req/s | throughput change | "
        "mean HTTP TTFT ms | p95 HTTP TTFT ms | preemptions",
    ]
    for pair in pairs:
        old, fixed = pair["original"], pair["fixed"]
        lines.append(
            f"{pair['concurrency']} | {old['throughput_rps']:.5f}/"
            f"{fixed['throughput_rps']:.5f} | "
            f"{pair['throughput_change_pct']:+.2f}% | "
            f"{old['ttft_ms']['mean']:.2f}/{fixed['ttft_ms']['mean']:.2f} | "
            f"{old['ttft_ms']['p95']:.2f}/{fixed['ttft_ms']['p95']:.2f} | "
            f"{pair['preemptions_original']:.0f}/{pair['preemptions_fixed']:.0f}"
        )
    lines += ["", "Diagnostics (server lifetime):"]
    for name, item in diagnostics.items():
        lines.append(f"{name}: {item['counts']}")
    lines += ["", "CPU verification:"]
    if (ROOT / "cpu_checks.json").exists():
        lines.append((ROOT / "cpu_checks.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    lines += [
        f"Source HEAD: {commit}",
        "No model, selector, predictor, hold, chunk-size or cache-budget changes.",
        "All-rank transfer/request/token checks are in each accepted result/audit.",
        "Execution envelopes include CPU submission within forward; "
        "not kernel-only GPU busy time.",
        "Preemptions/transfers cover warmup+measurement+drain; "
        "throughput uses completion window.",
        "Finite-window closed-loop progress changes the request mix; "
        "see each audit.json.",
    ]
    (ROOT / "REPORT.txt").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                pairs=len(pairs),
                diagnostics={k: v["counts"] for k, v in diagnostics.items()},
            )
        )
    )


if __name__ == "__main__":
    main()
