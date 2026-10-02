"""Controlled GLM TP8 prompt/cache matrix for seal policy experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/mnt/localnvme/glm53_adaptive_seal_20261002")
SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/bin"))
import cpu_offload_benchmark as benchmark  # noqa: E402
import glm53_early_kv_experiment as early  # noqa: E402
import glm53_reproduce as glm  # noqa: E402

COMMON = dict(
    copy_pipeline_depth=2,
    early_submission=True,
    cooperative_response_progress=True,
    deferral_policy="timing",
    late_admission_mode=4,
    late_admission_markers=True,
    zero_token_stage_markers=True,
    late_candidate_slots=10,
    late_seal_delay_us=200_000,
    late_seal_extension_us=100_000,
    late_seal_when_all_candidates_ready=True,
    late_partial_min_pending_candidates=8,
    late_partial_ready_percent=50,
)
PRIME = dict(
    COMMON,
    late_admission_mode=0,
    late_candidate_slots=4,
    late_seal_delay_us=0,
    late_seal_extension_us=0,
    late_seal_when_all_candidates_ready=False,
    late_partial_min_pending_candidates=0,
    late_partial_ready_percent=0,
)
POLICIES = {
    "predictive": dict(COMMON, late_seal_policy="predictive"),
    "predictive80": dict(
        COMMON, late_seal_policy="predictive", late_seal_prediction_scale_percent=80
    ),
    "predictive120": dict(
        COMMON, late_seal_policy="predictive", late_seal_prediction_scale_percent=120
    ),
    "worker": PRIME,
    "fixed0": dict(
        COMMON,
        late_seal_delay_us=0,
        late_seal_extension_us=0,
        late_seal_when_all_candidates_ready=False,
        late_partial_min_pending_candidates=0,
        late_partial_ready_percent=0,
    ),
    "fixed200ext100": COMMON,
    "fixed100": dict(COMMON, late_seal_delay_us=100_000, late_seal_extension_us=0),
    "fixed300": dict(COMMON, late_seal_delay_us=300_000, late_seal_extension_us=0),
}


def save(path, value):
    glm.save(path, value)


def configure(directory, options, label):
    for _ in range(60):
        try:
            result = json.loads(
                glm.call(
                    "/experimental/kv-load-scheduling",
                    {"early_load_completion": True, "experiment_options": options},
                )
            )
            assert all(result.get(k) == v for k, v in options.items()), result
            save(directory / (label + ".json"), result)
            return result
        except Exception as error:
            if getattr(error, "code", None) != 409:
                raise
            time.sleep(0.5)
    raise RuntimeError("policy configuration did not drain")


def profile(directory, dataset, concurrency, count, phase):
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(ROOT / "aiperf_venv/bin/python"),
        "-m",
        "aiperf",
        "profile",
        "--model",
        "glm53-compat",
        "--tokenizer",
        str(glm.MODEL),
        "--url",
        glm.BASE,
        "--endpoint-type",
        "completions",
        "--streaming",
        "--custom-dataset-type",
        "inputs-json",
        "--input-file",
        str(dataset),
        "--dataset-sampling-strategy",
        "sequential",
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(count),
        "--random-seed",
        "55",
        "--use-server-token-count",
        "--export-level",
        "raw",
        "--artifact-dir",
        str(directory / phase),
        "--profile-export-prefix",
        phase,
    ]
    save(directory / (phase + "_command.json"), command)
    begin = time.monotonic_ns()
    with (directory / (phase + ".log")).open("w") as log:
        subprocess.run(command, stdout=log, stderr=log, check=True, timeout=3600)
    save(
        directory / (phase + "_wall.json"),
        dict(begin_ns=begin, end_ns=time.monotonic_ns()),
    )
    result = json.loads((directory / phase / (phase + ".json")).read_text())
    assert not result.get("error_summary"), result.get("error_summary")
    assert result["request_latency"]["count"] == count
    return result


def make_workload(directory, length, fraction, request_hit_percent, count):
    if (directory / "manifest.json").exists():
        return json.loads((directory / "manifest.json").read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(glm.MODEL), local_files_only=True)
    prefix = int(length * fraction / 100) // 64 * 64
    prime, replay, manifest = benchmark.generate_datasets(
        tokenizer,
        "glm53-compat",
        count,
        count,
        prefix_tokens=prefix,
        suffix_tokens=length - prefix,
        seed=55,
    )
    hits = count * request_hit_percent // 100
    # Interleave cache misses to exercise mixed compute/transfer batches.
    hit_positions = [
        i for i in range(count) if (i * hits) // count != ((i + 1) * hits) // count
    ]
    prime["data"] = [prime["data"][i] for i in hit_positions]
    for dataset in (prime, replay):
        for entry in dataset["data"]:
            entry["payloads"][0].update(seed=55, add_special_tokens=False)
    for name, value in (("prime", prime), ("replay", replay)):
        save(directory / (name + ".json"), value)
    manifest.update(
        prompt_tokens=length,
        prefix_tokens=prefix,
        cached_prompt_percent=fraction,
        request_hit_percent=request_hit_percent,
        hit_positions=hit_positions,
        prime_count=hits,
        request_count=count,
    )
    save(directory / "manifest.json", manifest)
    save(
        directory / "sha256.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.glob("*.json")
        },
    )
    return manifest


def usage(path):
    rows = []
    for line in path.read_text().splitlines():
        item = json.loads(line)
        packets = [
            packet
            for response in item.get("responses", [])
            for packet in response.get("packets", [])
        ]
        packets += item.get("response", {}).get("data", [])
        for packet in packets:
            try:
                value = json.loads(packet["value"])
            except (ValueError, KeyError, TypeError):
                continue
            if value.get("usage"):
                u = value["usage"]
                rows.append(
                    dict(
                        request_id=item["request_headers"]["X-Request-ID"],
                        prompt=u["prompt_tokens"],
                        cached=u.get("prompt_tokens_details", {}).get(
                            "cached_tokens", 0
                        ),
                    )
                )
    return rows


def cell(directory, workload, options, server_dir, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    configure(directory, PRIME, "prime_setting")
    glm.reset_cache(directory, "reset_all", external=True)
    if manifest["prime_count"]:
        profile(directory, workload / "prime.json", 8, manifest["prime_count"], "prime")
    glm.reset_cache(directory, "reset_gpu", external=False)
    configure(directory, options, "policy_setting")
    before = glm.metrics(directory, "metrics_before")
    trace_files = list((server_dir / "timing").glob("engine-*.jsonl"))
    offsets = {p.name: p.stat().st_size for p in trace_files}
    late_before = set((server_dir / "timing/late_admission").glob("*.json"))
    from measurement_guard import measure_guard

    with measure_guard(directory, server_dir) as guard:
        replay = profile(
            directory, workload / "replay.json", 32, manifest["request_count"], "replay"
        )
    after = glm.metrics(directory, "metrics_after")
    configure(directory, PRIME, "drain_setting")
    for p in (server_dir / "timing").glob("engine-*.jsonl"):
        with p.open() as stream:
            stream.seek(offsets.get(p.name, 0))
            (directory / "engine.jsonl").write_text(stream.read())
    for p in set((server_dir / "timing/late_admission").glob("*.json")) - late_before:
        data = json.loads(p.read_text())
        if data.get("role") == "engine":
            save(directory / "late.json", data)
    raw = usage(directory / "replay/replay_raw.jsonl")
    save(directory / "usage.json", raw)
    expected = Counter(
        {
            manifest["prefix_tokens"]: manifest["prime_count"],
            0: manifest["request_count"] - manifest["prime_count"],
        }
    )
    expected += Counter()  # Remove zero-count entries.
    observed = Counter(x["cached"] for x in raw)
    gpu_hits = after["vllm:prefix_cache_hits"] - before["vllm:prefix_cache_hits"]
    preemptions = after["vllm:num_preemptions"] - before["vllm:num_preemptions"]
    result = dict(
        path=str(directory),
        throughput=replay["request_throughput"]["avg"],
        mean_ttft_ms=replay["time_to_first_token"]["avg"],
        p95_ttft_ms=replay["time_to_first_token"]["p95"],
        p99_ttft_ms=replay["time_to_first_token"]["p99"],
        cached_distribution=dict(observed),
        expected_cached_distribution=dict(expected),
        gpu_hits=gpu_hits,
        preemptions=preemptions,
        h2d_bytes=after["vllm:kv_offload_load_bytes"]
        - before["vllm:kv_offload_load_bytes"],
        **guard,
        accepted=not any(guard.values())
        and len(raw) == manifest["request_count"]
        and observed == expected
        and gpu_hits == 0
        and preemptions == 0,
    )
    save(directory / "result.json", result)
    print("CELL", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--policies", default="worker,fixed0,fixed200ext100,fixed100,fixed300"
    )
    parser.add_argument(
        "--workloads",
        default="8192:50:100,8192:95:100,16384:75:100,30720:50:100,30720:95:100,16384:95:50",
    )
    parser.add_argument("--requests", type=int, default=160)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--workload-request-counts")
    parser.add_argument("--distinct-fact-probe", action="store_true")
    args = parser.parse_args()
    run = ROOT / args.name
    run.mkdir(parents=True, exist_ok=True)
    glm.SOURCE = SOURCE
    glm.SCRIPTS = ROOT / "code"
    glm.config = early.config

    def experiment_popen(command, *positional, **keywords):
        if isinstance(command, list) and "--max-model-len" in command:
            command = list(command)
            command[command.index("--max-model-len") + 1] = str(args.max_model_len)
            if "--async-scheduling" not in command:
                command.append("--async-scheduling")
            save(run / "server/launched_command.json", command)
        return early.async_scheduler_popen(command, *positional, **keywords)

    subprocess.Popen = experiment_popen
    save(
        run / "provenance.json",
        dict(
            source=str(SOURCE),
            source_sha256={},
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
            ).strip(),
        ),
    )
    save(
        run / "protocol.json",
        vars(args) | dict(source=str(SOURCE), gpu_gib=96, cpu_gib=1984, concurrency=32),
    )
    os.environ.update(
        VLLM_COOPERATIVE_RESPONSE_PROGRESS="1",
        VLLM_DEFERRAL_TIMING_DIR=str(run / "server/timing"),
        VLLM_DEFERRAL_EXECUTION_TIMING="1",
        VLLM_DEFERRAL_LOAD_AUDIT="1",
        TMPDIR=str(ROOT / "tmp"),
    )
    (ROOT / "tmp").mkdir(exist_ok=True)
    settings = SimpleNamespace(
        root=run,
        cpu_gib=1984,
        block=64,
        batch=61440,
        gpu_gib=96,
        requests=args.requests,
        stage="experiment",
    )
    workloads = [tuple(map(int, x.split(":"))) for x in args.workloads.split(",")]
    policies = args.policies.split(",")
    counts = (
        [int(value) for value in args.workload_request_counts.split(",")]
        if args.workload_request_counts
        else [args.requests] * len(workloads)
    )
    if len(counts) != len(workloads) or any(count <= 0 for count in counts):
        raise ValueError("positive request count required for each workload")
    if any(length + 1 > args.max_model_len for length, _, _ in workloads):
        raise ValueError("prompt plus output exceeds configured model length")
    if any(
        count * length * 381600 > 1984 * 1024**3
        for (length, _, _), count in zip(workloads, counts)
    ):
        raise ValueError("unique prompt cohort exceeds physical CPU cache capacity")
    results = {}
    with glm.server(run / "server", "tp8", "A", settings):
        from correctness import check

        check(
            run / "correctness",
            {name: POLICIES[name] for name in policies},
            configure,
            glm,
            PRIME,
            prefix_len=58368 if args.max_model_len >= 65536 else 11776,
        )
        if args.distinct_fact_probe:
            from correctness import check_distinct

            check_distinct(
                run / "correctness_distinct",
                {name: POLICIES[name] for name in policies},
                configure,
                glm,
                PRIME,
                prefix_len=58368 if args.max_model_len >= 65536 else 11776,
            )
        for (length, fraction, hit_percent), request_count in zip(workloads, counts):
            name = f"p{length}_f{fraction}_h{hit_percent}"
            work = ROOT / "workloads" / f"{name}_n{request_count}"
            manifest = make_workload(work, length, fraction, hit_percent, request_count)
            for repeat in range(args.repeats):
                order = policies if repeat % 2 == 0 else list(reversed(policies))
                for policy in order:
                    target = run / name / f"r{repeat}_{policy}"
                    if (target / "result.json").exists():
                        results[str(target)] = json.loads(
                            (target / "result.json").read_text()
                        )
                        continue
                    print("START", target, flush=True)
                    for attempt in range(3):
                        measured = (
                            target
                            if not attempt
                            else target.with_name(target.name + f"_warmretry{attempt}")
                        )
                        result = cell(
                            measured, work, POLICIES[policy], run / "server", manifest
                        )
                        results[str(measured)] = result
                        save(run / "results.json", results)
                        if result["accepted"]:
                            break
                        if not result.get("measurement_jit_lines"):
                            raise RuntimeError(f"Invalid measured cell: {measured}")
                    else:
                        raise RuntimeError(f"Repeated first-use compilation: {target}")
    save(
        run / "complete.json",
        dict(cells=len(results), accepted=sum(x["accepted"] for x in results.values())),
    )


if __name__ == "__main__":
    main()
