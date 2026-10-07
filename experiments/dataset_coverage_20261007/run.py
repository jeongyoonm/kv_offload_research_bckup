# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Budgeted real-server prefill cohorts and causal shadow-data collection."""

import argparse
import concurrent.futures
import contextlib
import ctypes
import importlib.util
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from matrix import build

SOURCE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/localnvme/glm53_dataset_coverage_20261007")
BASE = "http://127.0.0.1:18407"
MODEL = (
    "/mnt/localnvme/huggingface/hub/models--Inferact--GLM-5.3-NVFP4/"
    "snapshots/ce67b36f3669192b5bb233819f0fda6c8a9837f8"
)
HIST = Path("/mnt/localnvme/glm53_predictor_overnight_20261006")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def log(event, **fields):
    record = dict(event=event, time=time.time(), **fields)
    with (ROOT / "process_log.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def call(path, payload=None, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode()
        return json.loads(raw) if raw else None


def textcall(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as response:
        return response.read().decode()


def counters():
    result = {}
    for line in textcall("/metrics").splitlines():
        if line.startswith("#") or not line:
            continue
        key, value = line.split()[:2]
        key = key.split("{")[0]
        if key in (
            "vllm:prefix_cache_hits_total",
            "vllm:num_preemptions_total",
            "vllm:kv_offload_load_bytes_total",
        ):
            result[key] = result.get(key, 0) + float(value)
    return result


def reset(external):
    for _ in range(60):
        if call(f"/reset_prefix_cache?reset_external={str(external).lower()}", {})[
            "success"
        ]:
            return
        time.sleep(0.1)
    raise RuntimeError("cache reset did not drain")


def options(active=False):
    original = Path(
        "/mnt/localnvme/glm53_predictor_overnight_20261006/dense_cached/micro_warm/p16384_f95_h100/n8/p16384_f95_h100/r0_micro/policy_setting.json"
    )
    result = json.loads(original.read_text())
    result.pop("early_load_completion", None)
    result.pop("schedule_delay_ms", None)
    result.update(
        late_seal_policy="predictive" if active else "event",
        late_seal_compute_deadline=active,
        late_seal_compute_deadline_shadow=False,
        late_seal_compute_deadline_lead_us=5000 if active else 0,
        late_seal_native_predictor=active,
        late_seal_freeze_estimator=True,
        late_seal_condition_preparation=active,
        late_seal_recent_preparation=active,
        late_seal_prediction_to_readiness=active,
        late_seal_full_prediction_wait=active,
        late_seal_coalesced_notifications=active,
        late_seal_hybrid_wait_us=50000 if active else 0,
    )
    return result


def configure(active=False, state=None):
    payload = dict(early_load_completion=True, experiment_options=options(active))
    if state is not None:
        payload["seal_estimator"] = dict(action="load", state=state)
    for _ in range(60):
        try:
            return call("/experimental/kv-load-scheduling", payload)
        except urllib.error.HTTPError as error:
            if error.code != 409:
                raise
            time.sleep(0.1)
    raise RuntimeError("policy update could not drain")


def prompt(context, suffix, seed):
    from transformers import AutoTokenizer

    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rng = random.Random(seed)

    def tokens(count, marker):
        if not count:
            return []
        pieces = [marker, "\n"]
        ids = []
        while len(ids) < count:
            pieces.extend(f" kv{rng.randrange(1_000_000_000)}" for _ in range(2048))
            ids = TOKENIZER.encode("".join(pieces), add_special_tokens=False)
        return ids[:count]

    prefix = tokens(context, f"COVERAGE_PREFIX_{seed:020d}")
    tail = tokens(suffix, f"COVERAGE_SUFFIX_{seed:020d}")
    return prefix, tail


def completion(tokens, name):
    started = time.monotonic_ns()
    payload = dict(
        model="glm53-compat",
        prompt=tokens,
        max_tokens=1,
        temperature=0,
        seed=0,
        ignore_eos=True,
        request_id=name,
    )
    result = call("/v1/completions", payload)
    return dict(
        request=name, started_ns=started, ended_ns=time.monotonic_ns(), response=result
    )


def snapshots():
    return {
        p: p.stat().st_size for p in (ROOT / "server/timing").glob("engine-*.jsonl")
    }


def traces(offsets, directory):
    configure()
    chunks = []
    for path in (ROOT / "server/timing").glob("engine-*.jsonl"):
        with path.open() as f:
            f.seek(offsets.get(path, 0))
            chunks.append(f.read())
    (directory / "engine.jsonl").write_text("".join(chunks))
    return [json.loads(line) for chunk in chunks for line in chunk.splitlines()]


def cohort(family, seed, repeat, split, *, retries=0):
    directory = (
        ROOT
        / "cells"
        / split
        / family["block"]
        / family["id"]
        / f"s{seed}_r{repeat}_a{retries}"
    )
    if (directory / "result.json").exists():
        return json.loads((directory / "result.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    save(directory / "family.json", dict(family, seed=seed, repeat=repeat, split=split))
    reset(True)
    requests = []
    primes = []
    for i, (s, c) in enumerate(family["shapes"]):
        prefix, tail = prompt(c, s, seed * 100000 + repeat * 1000 + i)
        hit = i * family["hit_percent"] // 100 != (i + 1) * family["hit_percent"] // 100
        requests.append(
            (prefix + tail, f"coverage-main-{seed}-{repeat}-{i}", c if hit else 0)
        )
        if c and hit:
            primes.append(prefix + [31700] * 64)
    background = []
    for i in range(family["background_loads"]):
        p, t = prompt(16384, 128, seed * 200000 + repeat * 1000 + i)
        primes.append(p + [31701] * 64)
        background.append((p + t, f"coverage-bg-{seed}-{repeat}-{i}"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for j in range(0, len(primes), 8):
            pending = [
                pool.submit(completion, p, f"coverage-prime-{seed}-{repeat}-{i}")
                for i, p in enumerate(primes[j : j + 8])
            ]
            for f in pending:
                f.result()
        reset(False)
        configure()
        before = counters()
        offsets = snapshots()
        call("/pause?mode=keep&clear_cache=false", {})
        gate = ROOT / "target_hold"
        if background:
            gate.touch()
        started = time.monotonic_ns()
        main = [pool.submit(completion, p, name) for p, name, _ in requests]
        bg = []
        try:
            deadline = time.monotonic() + 60
            target = len(main)
            while time.monotonic() < deadline:
                status = call("/experimental/deferral-cohort", timeout=10)
                if (
                    status["paused"]
                    and status["requests"] == target
                    and not status["queued_steps"]
                ):
                    break
                time.sleep(0.005)
            else:
                raise RuntimeError("cohort arrival barrier timed out")
            if background:
                call("/resume", {})
                while time.monotonic() < deadline:
                    status = call("/experimental/deferral-cohort", timeout=10)
                    if status["dataset_target_ready"] == len(main):
                        break
                    time.sleep(0.001)
                else:
                    raise RuntimeError("target KV never became ready at preload gate")
                bg = [pool.submit(completion, p, name) for p, name in background]
                while bg and time.monotonic() < deadline:
                    status = call("/experimental/deferral-cohort", timeout=10)
                    if status["dataset_background_submitted"] == len(bg):
                        break
                    time.sleep(0.001)
                else:
                    raise RuntimeError("background transfers were not submitted")
                if family["background_timing"] == "old":
                    time.sleep(0.01)
                gate.unlink()
            save(directory / "cohort_barrier.json", status)
            call("/resume", {})
            responses = [f.result(timeout=120) for f in main]
            ended = time.monotonic_ns()
        finally:
            gate.unlink(missing_ok=True)
            call("/resume", {})
            if bg:
                call("/abort_requests", {})
            for f in bg:
                with contextlib.suppress(Exception):
                    f.result(timeout=30)
    events = traces(offsets, directory)
    after = counters()
    actual = []
    for e in events:
        if e["event"] == "step_completion" and e.get("selected"):
            actual.extend(e.get("request_shapes", []))
    expected = sorted(
        [[len(tokens) - cached, cached] for tokens, _, cached in requests]
    )
    observed = sorted(actual)
    observed_cache = Counter(
        r["response"]["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        for r in responses
    )
    expected_cache = Counter(c for _, _, c in requests)
    totals = Counter()
    for e in events:
        if e["event"] == "step_completion":
            for req, n in e.get("selected", {}).items():
                totals[req] += n
    expected_totals = sorted(len(p) - cached for p, _, cached in requests)
    chunked = sum(expected_totals) > 61440
    accepted = (
        (
            sorted(totals.values()) == expected_totals
            if chunked
            else observed == expected
        )
        and observed_cache == expected_cache
        and after.get("vllm:prefix_cache_hits_total", 0)
        == before.get("vllm:prefix_cache_hits_total", 0)
        and after.get("vllm:num_preemptions_total", 0)
        == before.get("vllm:num_preemptions_total", 0)
        and all(
            "coverage-bg-" not in req
            for e in events
            if e["event"] == "step_completion"
            for req in e.get("selected", {})
        )
    )
    done = {
        e["step"]: e
        for e in events
        if e["event"] == "step_completion" and e.get("valid")
    }
    eligible = set()
    for e in events:
        if e["event"] != "dataset_forecast" or e["predecessor_step"] not in done:
            continue
        ranks = done[e["predecessor_step"]].get("execution_ranks", [])
        if len(ranks) == 8 and all(
            0 < r[2] <= r[3] <= e["time_ns"] and r[3] - r[2] <= 2000000 for r in ranks
        ):
            eligible.add(e["predecessor_step"])
    result = dict(
        path=str(directory),
        accepted=accepted,
        actual=observed,
        expected=expected,
        cached_distribution=dict(observed_cache),
        expected_cached=dict(expected_cache),
        elapsed_seconds=(ended - started) / 1e9,
        metrics_before=before,
        metrics_after=after,
        responses=responses,
        background_requests=len(bg),
        valid_forecasts=len(eligible) if accepted else 0,
        chunked=chunked,
    )
    save(directory / "result.json", result)
    log(
        "cell",
        family=family["id"],
        block=family["block"],
        split=split,
        accepted=accepted,
        seconds=result["elapsed_seconds"],
    )
    return result


def server_command():
    old = Path(
        "/mnt/localnvme/glm53_predictor_overnight_20261006/dense_cached/canary/gpu/server/command.json"
    )
    command = json.loads(old.read_text())
    command[1] = str(Path(__file__).resolve())
    command[2:2] = ["--exec-server"]
    command[command.index("--port") + 1] = "18407"
    command[command.index("--max-model-len") + 1] = "65536"
    cfg = json.loads(command[command.index("--kv-transfer-config") + 1])
    cfg["kv_connector_extra_config"]["cpu_cache_layout"] = "private"
    command[command.index("--kv-transfer-config") + 1] = json.dumps(cfg)
    if "--async-scheduling" not in command:
        command.append("--async-scheduling")
    return command, cfg


def stop_server(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)


def launch():
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if occupied:
        raise RuntimeError(f"GPUs occupied: {occupied}")
    command, cfg = server_command()
    env = dict(
        os.environ,
        PYTHONPATH=str(SOURCE),
        HF_HOME="/mnt/localnvme/huggingface",
        HF_HUB_OFFLINE="1",
        VLLM_SERVER_DEV_MODE="1",
        PYTHONHASHSEED="0",
        CUDA_HOME="/usr/local/cuda-13.0",
        VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS="0",
        VLLM_DEFERRAL_TIMING_DIR=str(ROOT / "server/timing"),
        VLLM_DEFERRAL_EXECUTION_TIMING="1",
        VLLM_PREDICTOR_DATASET_SHADOW="1",
        VLLM_COOPERATIVE_RESPONSE_PROGRESS="1",
        VLLM_DEFERRAL_PREDICTOR_PROGRESS="1",
        VLLM_PREDICTOR_GENERAL_PREFILL="1",
        VLLM_DATASET_BACKGROUND_PREFIX="coverage-bg-",
        VLLM_DATASET_TARGET_GATE=str(ROOT / "target_hold"),
        VLLM_DEFERRAL_NATIVE_OBSERVER=str(ROOT / "libseal_execution_observer.so"),
        VLLM_DEFERRAL_FAST_FOREST=str(
            HIST / "fast_forest/build/libpredictor_forest_walk.so"
        ),
        VLLM_CACHE_ROOT="/mnt/localnvme/cpu_cache_init_20261007/glm_canary_final/vllm_cache",
        FLASHINFER_WORKSPACE_BASE="/mnt/localnvme/cpu_cache_init_20261007/glm_canary_final/flashinfer_workspace",
        TMPDIR=str(ROOT / "tmp"),
    )
    env["PATH"] = (
        str(SOURCE / ".venv/bin") + ":/usr/local/cuda-13.0/bin:" + env.get("PATH", "")
    )
    (ROOT / "server").mkdir(exist_ok=True)
    (ROOT / "tmp").mkdir(exist_ok=True)
    save(ROOT / "server/command.json", command)
    save(ROOT / "server/connector_config.json", cfg)
    save(
        ROOT / "server/environment.json",
        {
            k: v
            for k, v in env.items()
            if k.startswith(("VLLM_", "FLASHINFER_", "HF_", "CUDA_", "PYTHON"))
        },
    )
    previous_log = ROOT / "server/server.log"
    if previous_log.exists():
        previous_log.rename(ROOT / "server" / f"server-{time.time_ns()}.log")
    f = previous_log.open("w")
    proc = subprocess.Popen(
        command, env=env, stdout=f, stderr=f, start_new_session=True, cwd=SOURCE
    )
    f.close()
    save(ROOT / "server/process.json", dict(pid=proc.pid, started=time.time()))
    return proc


def normalize():
    spec = importlib.util.spec_from_file_location(
        "coverage_corpus", SOURCE / "experiments/predictor_corpus_20261006/corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    rejected = Counter()
    for path in sorted((ROOT / "cells").rglob("result.json")):
        result = json.loads(path.read_text())
        if not result["accepted"]:
            continue
        audit_path = path.parent / "measurement_audit.json"
        if audit_path.exists() and not json.loads(audit_path.read_text())["usable"]:
            rejected["measurement_guard"] += 1
            continue
        meta = json.loads((path.parent / "family.json").read_text())
        workload = meta["id"]
        if meta["block"] == "familiar":
            suffix, context = meta["shapes"][0]
            total = suffix + context
            fraction = 99 if context / total > 0.98 else 95
            workload = f"p{total}_f{fraction}_h100"
        for line in (path.parent / "engine.jsonl").read_text().splitlines():
            e = json.loads(line)
            row, reason = module.extract(e, cached_only=False)
            if row is None:
                if e["event"] == "step_completion":
                    rejected[reason] += 1
                continue
            ranks = row["execution_ranks"]
            if any(not 0 < r[2] <= r[3] or r[3] - r[2] > 2000000 for r in ranks):
                rejected["rank_clock"] += 1
                continue
            row.update(
                episode=str(path.parent),
                source_file=str(path.parent / "engine.jsonl"),
                workload=workload,
                block=meta["block"],
                split=meta["split"],
                cpu_cached_prefix_tokens=e.get("cpu_cached_prefix_tokens"),
            )
            rows.append(row)
    from cold import rows as cold_rows

    rows.extend(cold_rows(module))
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    (data / "batches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run(
        [
            str(HIST / "eval-venv/bin/python"),
            str(SOURCE / "experiments/predictor_runtime_20261006/forecast_corpus.py"),
            "--corpus",
            str(data / "batches.jsonl"),
            "--output",
            str(data),
        ],
        check=True,
    )
    save(
        data / "batch_inventory.json", dict(samples=len(rows), rejected=dict(rejected))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    matrix = build()
    if (ROOT / "matrix.json").exists():
        assert json.loads((ROOT / "matrix.json").read_text()) == matrix
    else:
        save(ROOT / "matrix.json", matrix)
    if args.normalize:
        normalize()
        return
    proc = None
    started = time.monotonic()
    previous_allocation = ROOT / "allocation.json"
    used = (
        json.loads(previous_allocation.read_text()).get("total_seconds", 0)
        if previous_allocation.exists()
        else 0
    )
    if used >= 14400:
        raise RuntimeError("TP8 allocation budget exhausted")
    watchdog = None
    try:
        proc = launch()
        watchdog = threading.Timer(max(1, 14400 - used - 60), stop_server, args=(proc,))
        watchdog.daemon = True
        watchdog.start()
        log("server_started", pid=proc.pid)
        while time.monotonic() - started < 1500:
            if proc.poll() is not None:
                raise RuntimeError("server exited before health")
            try:
                call("/health", timeout=2)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(2)
        else:
            raise RuntimeError("startup exceeded pilot allowance")
        historical = json.loads((HIST / "boosted_residual/model.json").read_text())[
            "state"
        ]
        save(ROOT / "server/policy.json", configure(state=historical))
        log("server_ready", seconds=time.monotonic() - started)
        pilot = [
            matrix["train"][0],
            matrix["train"][6],
            next(f for f in matrix["train"] if f["block"] == "composition"),
            next(
                f
                for f in matrix["train"]
                if f["block"] == "transfer" and f["background_loads"] == 2
            ),
        ]
        for i, f in enumerate(pilot):
            result = cohort(f, 601, i, "pilot")
            if not result["accepted"]:
                raise RuntimeError(f"pilot rejected: {result['path']}")
        normalize()
        log("pilot_complete")
        if args.pilot_only:
            return
        blocks = ["geometry", "composition", "transfer", "familiar"]
        counts = Counter()
        positions = Counter()
        targets = dict(geometry=400, composition=400, transfer=400, familiar=1200)
        families = {b: [f for f in matrix["train"] if f["block"] == b] for b in blocks}
        while used + time.monotonic() - started < 9000 and any(
            counts[b] < targets[b] for b in blocks
        ):
            b = min(
                (b for b in blocks if counts[b] < targets[b]),
                key=lambda b: counts[b] / targets[b],
            )
            i = positions[b]
            f = families[b][i % len(families[b])]
            repeat = i // len(families[b])
            result = cohort(f, 701 + repeat % 3, repeat, "train")
            counts[b] += result["valid_forecasts"]
            positions[b] += 1
            if positions[b] % len(families[b]) == 0:
                save(
                    ROOT / "collection_progress.json",
                    dict(counts=counts, targets=targets, positions=positions),
                )
            if positions[b] > 400 and counts[b] == 0:
                raise RuntimeError(f"no causal forecasts collected for {b}")
        normalize()
        log("collection_complete")
        for split, seeds in [
            ("development", [801, 802, 803]),
            ("final", [901, 902, 903]),
        ]:
            for seed in seeds:
                for i, f in enumerate(matrix["challenge"] + matrix["train"][::8]):
                    if used + time.monotonic() - started >= (
                        10350 if split == "development" else 11700
                    ):
                        break
                    cohort(f, seed, i, split)
                else:
                    continue
                break
        normalize()
        log("shadow_test_complete")
    except BaseException as error:
        log("failed", error=repr(error))
        raise
    finally:
        if watchdog is not None:
            watchdog.cancel()
        if proc is not None:
            stop_server(proc)
        save(
            ROOT / "allocation.json",
            dict(
                seconds=time.monotonic() - started,
                total_seconds=used + time.monotonic() - started,
                limit_seconds=14400,
                server_stopped=proc is None or proc.poll() is not None,
            ),
        )
        log("stopped", seconds=time.monotonic() - started)


TOKENIZER = None

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--exec-server":
        if ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0):
            raise OSError(ctypes.get_errno(), "PR_SET_THP_DISABLE")
        os.execvp(sys.argv[2], sys.argv[2:])
    main()
