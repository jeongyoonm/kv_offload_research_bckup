# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned structural trace adapters; context is censored, never truncated."""

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

CHAR = Path("/mnt/localnvme/jeongyoon/trace_characterization_20261012")
OUT = Path("/mnt/localnvme/jeongyoon/real_traces_20261012/datasets")
MODEL = (
    "/mnt/localnvme/huggingface/hub/models--Inferact--GLM-5.3-NVFP4/"
    "snapshots/ce67b36f3669192b5bb233819f0fda6c8a9837f8"
)
LIMIT = 81920


def lcp(a, b):
    return next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = OUT / (name + ".json.gz")
    with gzip.open(path, "wt", compresslevel=3) as f:
        json.dump(value, f)
    return dict(path=str(path), sha256=digest(path))


def mooncake(name, count):
    path = CHAR / "mooncake" / (name + "_trace.jsonl")
    all_rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = []
    excluded = []
    blocks = {}
    parents = {}
    for index, row in enumerate(all_rows[:count]):
        if row["input_length"] + row["output_length"] > LIMIT:
            excluded.append(index)
            continue
        tokens = []
        parent = None
        for h in row["hash_ids"]:
            key = (parent, h)
            if h in parents:
                assert parents[h] == parent, "hash is not a cumulative prefix"
            parents[h] = parent
            if key not in blocks:
                seed = int.from_bytes(
                    hashlib.sha256(f"{name}:{parent}:{h}".encode()).digest()[:8],
                    "little",
                )
                blocks[key] = (
                    np.random.default_rng(seed).integers(128, 31700, 512).tolist()
                )
            tokens.extend(blocks[key])
            parent = h
        rows.append(
            dict(
                index=index,
                prompt=tokens[: row["input_length"]],
                output_tokens=row["output_length"],
                arrival_ms=row["timestamp"],
                hashes=row["hash_ids"],
            )
        )
    # Exact block-level prefix reproduction, with no accidental 64-token sharing.
    sample = random.Random(12).sample(rows, min(150, len(rows)))
    for i, a in enumerate(sample):
        for b in sample[i + 1 :]:
            expected = min(
                lcp(a["hashes"], b["hashes"]) * 512, len(a["prompt"]), len(b["prompt"])
            )
            actual = lcp(a["prompt"], b["prompt"])
            assert actual >= expected and actual // 64 == expected // 64, (
                actual,
                expected,
            )
    original_seconds = (rows[-1]["arrival_ms"] - rows[0]["arrival_ms"]) / 1000
    return save(
        "mooncake_" + name,
        dict(
            kind="mooncake",
            name=name,
            requests=rows,
            source_sha256=digest(path),
            selected_source_range=[0, count],
            source_requests=len(all_rows),
            context_excluded_indices=excluded,
            original_window_seconds=original_seconds,
            original_window_rps=len(rows) / original_seconds,
            block_tokens=512,
            adapter=(
                "Deterministic valid token IDs from cumulative prefix hashes. "
                "Exact input/output lengths. No source text exists. Separate "
                "hash namespace per file. Generated output is not linked to "
                "subsequent hashes."
            ),
        ),
    )


def inferact(count):
    path = CHAR / "inferact/codex_swebenchpro.json"
    raw = json.loads(path.read_text())
    ids = random.Random(20261012).sample(range(len(raw)), count)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    sessions = []
    censorship = []
    summaries = []
    for pos, sid in enumerate(ids):
        messages = raw[sid]["conversations"]
        assert len(messages) % 2 == 0
        history = []
        rows = []
        previous = []
        for turn in range(len(messages) // 2):
            human, assistant = messages[2 * turn : 2 * turn + 2]
            assert human["from"] == "human" and assistant["from"] == "gpt"
            history.append(dict(role="user", content=human["value"]))
            tokens = tokenizer.apply_chat_template(
                history, tokenize=True, add_generation_prompt=True, clear_thinking=False
            )
            output = max(
                1, len(tokenizer.encode(assistant["value"], add_special_tokens=False))
            )
            if len(tokens) + output > LIMIT:
                censorship.append(
                    dict(
                        session=sid,
                        first_excluded_turn=turn,
                        input_tokens=len(tokens),
                        output_tokens=output,
                        omitted_calls=len(messages) // 2 - turn,
                    )
                )
                break
            common = lcp(previous, tokens)
            rows.append(
                dict(
                    index=turn,
                    session=sid,
                    prompt=tokens,
                    output_tokens=output,
                    previous_prompt_lcp=common,
                )
            )
            previous = tokens
            history.append(dict(role="assistant", content=assistant["value"]))
        if rows:
            sessions.append(dict(session=sid, requests=rows))
        summaries.append(
            dict(
                session=sid,
                total_source_calls=len(messages) // 2,
                retained_calls=len(rows),
            )
        )
        print(
            json.dumps(dict(prepared_session=pos + 1, session=sid, retained=len(rows))),
            flush=True,
        )
    return save(
        "inferact",
        dict(
            kind="inferact",
            name="codex_swebenchpro",
            sessions=sessions,
            selected_session_ids=ids,
            session_summary=summaries,
            source_sha256=digest(path),
            context_censorship=censorship,
            adapter=(
                "GLM original chat template, retained recorded conversation "
                "history, real model generation with GLM-tokenized recorded "
                "response length as target. Subsequent history uses recorded "
                "assistant text, not generated text; original GPT cache/output "
                "lengths cannot be reproduced. Causal per-session order, zero "
                "additional tool wait because per-call timestamps are absent. "
                "Censor entire remaining session at first context-limit violation."
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=128)
    parser.add_argument("--moon-requests", type=int, default=1600)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in ["conversation", "toolagent", "synthetic"]:
        result[name] = mooncake(name, args.moon_requests)
        print(json.dumps(dict(prepared=name, **result[name])), flush=True)
    result["inferact"] = inferact(args.sessions)
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
