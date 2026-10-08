# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-cohort GLM TP8 calibration with native copies in each model worker."""

import argparse
import ctypes
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
MODEL = (
    "/mnt/localnvme/huggingface/hub/models--Inferact--GLM-5.3-NVFP4/"
    "snapshots/ce67b36f3669192b5bb233819f0fda6c8a9837f8"
)
GEOMETRY = {
    "short": dict(context=4096, suffix=64, batch=8),
    "medium": dict(context=15552, suffix=512, batch=8),
    "long": dict(context=58368, suffix=2048, batch=8),
}
CONFIGS = [(c, d) for c in (8, 16, 32, 64, 128) for d in (1, 2)]


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def make_plan(args):
    configs = [(128, 2), (16, 2)] if args.pilot else CONFIGS
    if args.configs:
        configs = [tuple(map(int, s.split(":"))) for s in args.configs]
    assert all(c > 0 and d in (1, 2) for c, d in configs)
    shapes = ["short"] if args.pilot else list(GEOMETRY)
    modes = ["concurrent", "after_enqueued"] if args.confirmation else ["concurrent"]
    cells = []
    for repeat in range(args.repeats):
        order = [(c, d, p) for c, d in configs for p in ("4k", "16k", "60k")]
        random.Random(9109 + repeat).shuffle(order)
        cells.extend(
            dict(
                repeat=repeat,
                shape=None,
                mode="copy_only",
                chunk_mib=c,
                depth=d,
                profile=p,
                jobs=4,
            )
            for c, d, p in order
        )
    for shape in shapes:
        for repeat in range(args.repeats):
            cells.append(
                dict(
                    repeat=repeat,
                    shape=shape,
                    mode="compute_only",
                    chunk_mib=128,
                    depth=2,
                    profile="4k",
                    jobs=4,
                )
            )
            order = [
                (c, d, p, m)
                for c, d in configs
                for p in ("4k", "16k", "60k")
                for m in modes
            ]
            random.Random(9209 + repeat).shuffle(order)
            cells.extend(
                dict(
                    repeat=repeat,
                    shape=shape,
                    mode=m,
                    chunk_mib=c,
                    depth=d,
                    profile=p,
                    jobs=4,
                )
                for c, d, p, m in order
            )
    for index, cell in enumerate(cells, 1):
        cell["id"] = index
    return dict(
        cells=cells,
        geometry=GEOMETRY,
        configs=configs,
        shapes=shapes,
        repeats=args.repeats,
        modes=modes,
        seed=9109,
        scope="Real cached GLM forward, independent reconstructed KV copies; "
        "GPU-resident prefixes, fixed cohorts, no scheduler late admission; "
        "not serving throughput or full model-output equivalence.",
    )


def environment(root):
    if ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "PR_SET_THP_DISABLE")
    env = dict(
        HF_HOME="/mnt/localnvme/huggingface",
        HF_HUB_OFFLINE="1",
        CUDA_HOME="/usr/local/cuda-13.0",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_CACHE_ROOT="/mnt/localnvme/cpu_cache_init_20261007/glm_canary_final/vllm_cache",
        FLASHINFER_WORKSPACE_BASE="/mnt/localnvme/cpu_cache_init_20261007/"
        "glm_canary_final/flashinfer_workspace",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
        TMPDIR=str(root / "tmp"),
    )
    (root / "tmp").mkdir(exist_ok=True)
    os.environ.update(env)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--configs", nargs="+")
    args = parser.parse_args()
    assert args.repeats > 0
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "plan.json").exists(), "Use a fresh output directory"
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    assert not apps, f"GPUs occupied: {apps}"
    plan = make_plan(args)
    save(root / "plan.json", plan)
    env = environment(root)
    save(root / "ownership.json", dict(pid=os.getpid(), pgid=os.getpgrp()))
    hashes = {
        str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).with_name("worker.py"))
    }
    save(
        root / "provenance.json",
        dict(
            source=str(SOURCE),
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
            ).strip(),
            environment=env,
            hashes=hashes,
            model=MODEL,
        ),
    )
    from experiments.h2d_interference_20261009 import worker
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=8,
        distributed_executor_backend="mp",
        kv_cache_dtype="fp8_e4m3",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=65536,
        max_num_batched_tokens=61440,
        max_num_seqs=64,
        kv_cache_memory_bytes=103079215104,
        gpu_memory_utilization=0.80,
        block_size=64,
        enable_flashinfer_autotune=False,
        async_scheduling=True,
        numa_bind=True,
        numa_bind_nodes=[0, 0, 0, 0, 1, 1, 1, 1],
        numa_bind_cpus=[
            f"{i * 16}-{i * 16 + 15},{i * 16 + 128}-{i * 16 + 143}" for i in range(8)
        ],
    )
    save(root / "model_config.json", kwargs)
    llm = None
    started = time.monotonic()
    accepted = 0
    initialized = False
    try:
        llm = LLM(**kwargs)
        params = SamplingParams(max_tokens=1, temperature=0, seed=0, ignore_eos=True)
        ready = llm.collective_rpc(worker.setup, timeout=300, args=(str(root),))
        assert {r["rank"] for r in ready} == set(range(8))
        save(root / "workers.json", ready)
        initialized = True
        prefixes = {}
        previous_shape = None
        for cell in plan["cells"]:
            shape = cell["shape"]
            if shape is not None and shape != previous_shape:
                geom = GEOMETRY[shape]
                prefixes[shape] = [
                    random.Random(100000 + j).choices(
                        range(1000, 30000), k=geom["context"]
                    )
                    for j in range(geom["batch"])
                ]
                # Prime exact block-aligned contexts, then warm actual cached shapes.
                prompts = [dict(prompt_token_ids=p + [31000]) for p in prefixes[shape]]
                llm.generate(prompts, params, use_tqdm=False)
                for warm in range(2):
                    prompts = [
                        dict(
                            prompt_token_ids=p
                            + random.Random(200000 + warm * 100 + j).choices(
                                range(1000, 30000), k=geom["suffix"]
                            )
                        )
                        for j, p in enumerate(prefixes[shape])
                    ]
                    llm.generate(prompts, params, use_tqdm=False)
                previous_shape = shape
            llm.collective_rpc(worker.arm, timeout=120, args=(cell,))
            responses = None
            if cell["mode"] == "copy_only":
                llm.collective_rpc(worker.copy_only, timeout=120)
            else:
                geom = GEOMETRY[shape]
                # Repeat-matched token payloads across all configurations and modes.
                # The first suffix token differs per cell, preventing suffix KV hits.
                prompts = [
                    dict(
                        prompt_token_ids=p
                        + [32000 + cell["id"]]
                        + random.Random(300000 + cell["repeat"] * 100 + j).choices(
                            range(1000, 30000), k=geom["suffix"] - 1
                        )
                    )
                    for j, p in enumerate(prefixes[shape])
                ]
                responses = llm.generate(prompts, params, use_tqdm=False)
                assert len(responses) == geom["batch"]
                assert all(len(r.outputs[0].token_ids) == 1 for r in responses)
            rows = llm.collective_rpc(worker.drain, timeout=180)
            assert {r["rank"] for r in rows} == set(range(8))
            if shape is not None:
                geom = GEOMETRY[shape]
                for row in rows:
                    assert row["suffix"] == [geom["suffix"]] * geom["batch"], row
                    assert row["context"] == [geom["context"]] * geom["batch"], row
                assert len({tuple(r["suffix"]) for r in rows}) == 1
            result = dict(
                cell=cell,
                ranks=rows,
                output_tokens=None
                if responses is None
                else [r.outputs[0].token_ids for r in responses],
            )
            save(root / f"cell-{cell['id']:04d}.json", result)
            accepted += 1
            print(
                json.dumps(
                    dict(
                        event="cell",
                        accepted=accepted,
                        total=len(plan["cells"]),
                        cell=cell,
                    )
                ),
                flush=True,
            )
        closed = llm.collective_rpc(worker.close, timeout=120)
        initialized = False
        save(root / "worker_cleanup.json", closed)
        save(
            root / "summary.json",
            dict(complete=True, accepted=accepted, seconds=time.monotonic() - started),
        )
    finally:
        if llm is not None:
            if initialized:
                # A failed/in-flight cell is reclaimed by executor shutdown.
                print(
                    "Calibration incomplete; shutting down owned executor", flush=True
                )
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
