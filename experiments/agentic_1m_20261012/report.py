# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched results only, with workload scope and unmeasured effects explicit."""

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.agentic_1m_20261012.common import DATA, ROOT, SOURCE, save


def collect():
    rows = []
    for root in ROOT.parent.glob("a1m[bf]*"):
        for path in root.glob("cells/*/result.json"):
            result = json.loads(path.read_text())
            if result["spec"].get("measurement") == "capability":
                continue
            result["root"] = str(root)
            rows.append(result)
    return rows


def main():
    directory = ROOT / "report"
    directory.mkdir(exist_ok=True)
    cells = collect()
    grouped = defaultdict(dict)
    for cell in cells:
        grouped[cell["spec"]["id"]][cell["arm"]] = cell
    pairs = []
    for name, group in grouped.items():
        if set(group) != {"baseline", "full"}:
            continue
        b, f = group["baseline"], group["full"]
        assert b["dataset_sha256"] == f["dataset_sha256"]
        assert b["token_sha256"] == f["token_sha256"]
        assert b["spec"] == f["spec"]
        assert b["requests"] == f["requests"]
        assert b["input_tokens"] == f["input_tokens"]
        assert b["output_tokens"] == f["output_tokens"]
        if not b["accepted"] or not f["accepted"]:
            continue
        pair = dict(
            id=name,
            dataset=b["spec"]["dataset"],
            concurrency=b["spec"]["concurrency"],
            repeat=b["spec"].get("repeat", 0),
            requests=b["requests"],
            baseline_rps=b["throughput_rps"],
            full_rps=f["throughput_rps"],
            throughput_gain_pct=100 * (f["throughput_rps"] / b["throughput_rps"] - 1),
            full_late_admission_events=f["protocol_counts"].get(
                "membership_sealed_activated", 0
            ),
        )
        for arm, result in (("baseline", b), ("full", f)):
            for key in (
                "ttft_ms",
                "scheduled_ttft_ms",
                "driver_queue_ms",
                "latency_ms",
            ):
                for metric in ("mean", "p50", "p95", "p99"):
                    pair[f"{arm}_{key}_{metric}"] = result[key][metric]
            source = result.get("token_source_deltas", {})
            for kind in ("local_cache_hit", "external_kv_transfer", "local_compute"):
                pair[f"{arm}_{kind}_pct"] = (
                    100 * source.get(kind, 0) / result["input_tokens"]
                )
            pair[f"{arm}_cpu_request_pct"] = (
                100 * result["realized_outcomes"].get("cpu_hit", 0) / result["requests"]
            )
            pair[f"{arm}_h2d_bytes_all_ranks"] = result["h2d_bytes_all_ranks"]
            pair[f"{arm}_d2h_bytes_all_ranks"] = result["d2h_bytes_all_ranks"]
            pair[f"{arm}_preemptions"] = result["metric_deltas"].get(
                "vllm:num_preemptions_total", 0
            )
        for metric in ("mean", "p50", "p95", "p99"):
            pair[f"ready_ttft_{metric}_reduction_pct"] = 100 * (
                1
                - pair[f"full_scheduled_ttft_ms_{metric}"]
                / pair[f"baseline_scheduled_ttft_ms_{metric}"]
            )
        pairs.append(pair)
    pairs.sort(key=lambda r: (r["dataset"], r["concurrency"], r["repeat"]))
    save(directory / "pairs.json", pairs)
    save(directory / "all_cells.json", cells)
    if pairs:
        with (directory / "pairs.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    workloads = {}
    for path in DATA.glob("*.json"):
        data = json.loads(path.read_text())
        workloads[path.stem] = {k: v for k, v in data.items() if k != "sessions"}
    save(directory / "workload_scope.json", workloads)
    text = [
        "GLM5.3 NVFP4 TP8,1,048,576 total context, recorded output lengths.",
        f"Matched accepted pairs:{len(pairs)}. "
        f"Accepted measured cells:{sum(c['accepted'] for c in cells)}.",
        "Inferact:128 seeded sessions, full recorded histories; "
        "completion-paced with32/16/8 HTTP cap.",
        "TraceLab:structural proxy,32 context-stratified sessions/provider, "
        "fixed contiguous source windows up to64 calls/session. "
        "No original text/prefix identities available.",
        "Both arms use equal96GiB/rank GPU and2.13TB server-wide CPU budgets; "
        "natural cache evolution, no forced provider-hit ratio.",
        "Original selector/model weights/numerics retained. "
        "No model quality or bitwise equivalence claim.",
        "GPU gaps/predictor errors and KV-ready opportunity denominators require "
        "diagnostic records; cumulative admissions alone cannot establish them.",
        "Single-pair and two-pair variation is not a confidence interval. "
        "Whole finite-episode throughput includes cold start and serialized tails.",
        "",
        "Dataset | concurrency | repeat | baseline_rps | full_rps | gain% | "
        "readyTTFTmean reduction% | CPU request% baseline/full",
    ]
    for p in pairs:
        text.append(
            f"{p['dataset']} | {p['concurrency']} | {p['repeat']} | "
            f"{p['baseline_rps']:.4f} | {p['full_rps']:.4f} | "
            f"{p['throughput_gain_pct']:+.2f} | "
            f"{p['ready_ttft_mean_reduction_pct']:+.2f} | "
            f"{p['baseline_cpu_request_pct']:.2f}/"
            f"{p['full_cpu_request_pct']:.2f}"
        )
    (directory / "REPORT.txt").write_text("\n".join(text) + "\n")
    if pairs:
        keys = sorted(set((p["dataset"], p["concurrency"]) for p in pairs))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        labels = [
            name.replace("tracelab_", "TraceLab ").replace("_windows", "") + f"\nC={n}"
            for name, n in keys
        ]
        x = np.arange(len(keys))
        width = 0.36
        for ax, metric, title, units in [
            (axes[0], "rps", "Serving throughput", "requests/s"),
            (axes[1], "scheduled_ttft_ms_mean", "Mean ready-to-first-token", "seconds"),
        ]:
            for i, arm in enumerate(("baseline", "full")):
                values = [
                    np.mean(
                        [
                            p[f"{arm}_{metric}"]
                            for p in pairs
                            if (p["dataset"], p["concurrency"]) == key
                        ]
                    )
                    for key in keys
                ]
                if units == "seconds":
                    values = np.asarray(values) / 1000
                ax.bar(
                    x + (i - 0.5) * width,
                    values,
                    width,
                    label="Baseline" if i == 0 else "Full prototype",
                )
            ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
            ax.set_title(title)
            ax.set_ylabel(units)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(directory / "comparison.png", dpi=180)
        fig.savefig(directory / "comparison.pdf")
        plt.close(fig)
    save(
        directory / "report_provenance.json", dict(source=str(SOURCE), pairs=len(pairs))
    )
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    main()
