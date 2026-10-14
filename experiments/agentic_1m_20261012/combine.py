# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combine the exact frozen provider windows without changing request payloads."""

import hashlib
import json
import random

import numpy as np

from experiments.agentic_1m_20261012.common import DATA, ROOT, save
from experiments.agentic_1m_20261012.prepare import Writer, finish


def main():
    writer = Writer("tracelab_mixed_windows")
    sessions = []
    for provider in ("codex", "claude"):
        data = json.loads((DATA / f"tracelab_{provider}_windows.json").read_text())
        blob = np.memmap(data["token_file"], dtype="<u4", mode="r")
        for session in data["sessions"]:
            sid = provider + "-" + session["session"]
            rows = []
            for source in session["requests"]:
                tokens = blob[
                    source["offset"] : source["offset"] + source["input_tokens"]
                ]
                assert (
                    hashlib.sha256(tokens.tobytes()).hexdigest()
                    == source["prompt_sha256"]
                )
                metadata = dict(source, session=sid, provider=provider)
                result = writer.row(tokens, source["output_tokens"], **metadata)
                assert result["prompt_sha256"] == source["prompt_sha256"]
                rows.append(result)
            sessions.append(dict(session=sid, requests=rows))
    random.Random(20261014).shuffle(sessions)
    finish(
        "tracelab_mixed_windows",
        sessions,
        writer,
        adapter=(
            "Exact union of both frozen provider windows; payload bytes, ISL/OSL, "
            "distribution and within-session causality unchanged. Seeded session "
            "interleaving,64 logical agents; HTTP concurrency8/16/32."
        ),
    )
    specs = [
        dict(
            id=f"{dataset}_n{n}_r0",
            dataset=dataset,
            concurrency=n,
            min_interval_seconds=0,
            repeat=0,
        )
        for n in (32, 16, 8)
        for dataset in ("inferact", "tracelab_mixed_windows")
    ]
    specs.append(
        dict(
            id="tracelab_codex_windows_n32_r0",
            dataset="tracelab_codex_windows",
            concurrency=32,
            min_interval_seconds=0,
            repeat=0,
        )
    )
    save(ROOT / "primary_plan_mixed.json", specs)


if __name__ == "__main__":
    main()
