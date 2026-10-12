# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uncensored text replay and explicitly assumed session-local structural replay."""

import argparse
import hashlib
import json
import random
import subprocess

import numpy as np
from transformers import AutoTokenizer

from experiments.agentic_1m_20261012.common import (
    CHAR,
    DATA,
    LIMIT,
    ROOT,
    digest,
    log,
    save,
)
from experiments.real_traces_20261012.prepare import MODEL, prompt_tokens


class Writer:
    """Store token IDs compactly; request metadata never embeds prompt text."""

    def __init__(self, name):
        self.path = DATA / (name + ".u32")
        self.stream = self.path.open("xb")
        self.offset = 0

    def row(self, tokens, output, **metadata):
        array = np.asarray(tokens, dtype="<u4")
        assert len(array) > 0 and output > 0 and len(array) + output <= LIMIT
        result = dict(
            metadata,
            offset=self.offset,
            input_tokens=len(array),
            output_tokens=int(output),
            prompt_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
        )
        array.tofile(self.stream)
        self.offset += len(array)
        return result

    def close(self):
        self.stream.close()
        return dict(token_file=str(self.path), token_sha256=digest(self.path))


def finish(name, sessions, writer, **metadata):
    rows = [r for s in sessions for r in s["requests"]]
    counts = [r["input_tokens"] for r in rows]
    value = dict(
        name=name,
        kind="sessions",
        sessions=sessions,
        max_model_len=LIMIT,
        requests=len(rows),
        input_tokens=sum(counts),
        output_tokens=sum(r["output_tokens"] for r in rows),
        input_distribution={
            str(p): float(np.quantile(counts, p / 100))
            for p in (0, 50, 90, 95, 99, 100)
        },
        **writer.close(),
        **metadata,
    )
    save(DATA / (name + ".json"), value)
    log(
        "dataset_prepared",
        name=name,
        requests=len(rows),
        sessions=len(sessions),
        input_tokens=value["input_tokens"],
        output_tokens=value["output_tokens"],
        distribution=value["input_distribution"],
    )
    return value


def inferact(count):
    name = "inferact"
    raw_path = CHAR / "inferact/codex_swebenchpro.json"
    raw = json.loads(raw_path.read_text())
    selected = random.Random(20261012).sample(range(len(raw)), count)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    writer = Writer(name)
    sessions, excluded = [], []
    for position, sid in enumerate(selected):
        history, rows = [], []
        previous = np.empty(0, dtype="<u4")
        messages = raw[sid]["conversations"]
        for turn in range(len(messages) // 2):
            user, assistant = messages[turn * 2 : turn * 2 + 2]
            assert user["from"] == "human" and assistant["from"] == "gpt"
            history.append(dict(role="user", content=user["value"]))
            tokens = np.asarray(prompt_tokens(tokenizer, history), dtype="<u4")
            output = max(
                1, len(tokenizer.encode(assistant["value"], add_special_tokens=False))
            )
            if len(tokens) + output > LIMIT:
                excluded.append(
                    dict(session=sid, turn=turn, input=len(tokens), output=output)
                )
                break
            n = min(len(previous), len(tokens))
            changes = np.flatnonzero(previous[:n] != tokens[:n])
            common = int(changes[0]) if len(changes) else n
            rows.append(
                writer.row(
                    tokens,
                    output,
                    index=turn,
                    session=sid,
                    previous_prompt_lcp=common,
                    external_wait_seconds=0,
                )
            )
            previous = tokens
            history.append(dict(role="assistant", content=assistant["value"]))
        sessions.append(dict(session=sid, requests=rows))
        log(
            "inferact_session_prepared",
            position=position,
            session=sid,
            requests=len(rows),
        )
    return finish(
        name,
        sessions,
        writer,
        source_sha256=digest(raw_path),
        selected_session_ids=selected,
        excluded=excluded,
        adapter=(
            "Recorded text, GLM template, recorded response length; recorded "
            "history retained. No external waits available."
        ),
    )


def query(sql):
    # Reuse the isolated DuckDB environment; serving dependencies stay fixed.
    code = (
        "import duckdb,json,sys; "
        "c=duckdb.connect(sys.argv[1],read_only=True); "
        "print(json.dumps(c.execute(sys.argv[2]).fetchall(),default=str))"
    )
    return json.loads(
        subprocess.check_output(
            [
                str(CHAR / ".venv/bin/python"),
                "-c",
                code,
                str(CHAR / "tracelab/tracelab_v0.0.2.duckdb"),
                sql,
            ],
            text=True,
        )
    )


def structural_prompt(previous, input_tokens, reported_prefix, rng):
    """Assume continuity only within one session; reset on a context shrink."""
    keep = min(len(previous), reported_prefix, input_tokens)
    if input_tokens < len(previous):
        keep = 0
    fresh = rng.integers(128, 31700, input_tokens - keep, dtype=np.uint32)
    return np.concatenate((previous[:keep], fresh)), keep


def tracelab(provider, count):
    name = "tracelab_" + provider
    candidates = query(f"""select session_id,count(*),max(input_tokens_total)
        from rounds where provider='{provider}' and input_tokens_total>0
        group by session_id order by max(input_tokens_total),session_id""")
    rng = random.Random(20261012 + (1 if provider == "claude" else 0))
    # Equal sampling from source session-context quartiles preserves their weights.
    quartiles = np.array_split(np.arange(len(candidates)), 4)
    selected = []
    for i, q in enumerate(quartiles):
        quota = count // 4 + (i < count % 4)
        selected.extend(candidates[j][0] for j in rng.sample(q.tolist(), quota))
    rng.shuffle(selected)
    escaped = ",".join("'" + s.replace("'", "''") + "'" for s in selected)
    records = query(f"""select session_id,round_pk,round_index,input_tokens_total,
        prefix_tokens,output_tokens from rounds
        where provider='{provider}' and session_id in ({escaped})
        order by session_id,round_index,ingest_seq""")
    waits = dict(
        query(f"""select r.round_pk,epoch(min(t.timestamp))
        from rounds r join timing_events t using(round_pk)
        where r.provider='{provider}' and r.session_id in ({escaped})
        and t.event_type in ('user_message','tool_result') group by r.round_pk""")
    )
    source_by_session = {sid: [] for sid in selected}
    for row in records:
        source_by_session[row[0]].append(row)
    writer = Writer(name)
    sessions, excluded = [], []
    counts = dict(context_shrinks=0, unobserved_prefix_tokens=0, zero_output=0)
    for sid in selected:
        previous = np.empty(0, dtype="<u4")
        seed = int.from_bytes(
            hashlib.sha256((provider + sid).encode()).digest()[:8], "little"
        )
        token_rng = np.random.default_rng(seed)
        rows = []
        for _, pk, index, input_tokens, prefix, output in source_by_session[sid]:
            if not input_tokens or not output:
                excluded.append(dict(round_pk=pk, reason="zero input or zero output"))
                counts["zero_output"] += int(not output)
                continue
            if input_tokens + output > LIMIT:
                raise ValueError("Source request exceeds declared model context")
            tokens, keep = structural_prompt(
                previous, input_tokens, prefix or 0, token_rng
            )
            counts["context_shrinks"] += int(input_tokens < len(previous))
            counts["unobserved_prefix_tokens"] += max(0, (prefix or 0) - keep)
            rows.append(
                writer.row(
                    tokens,
                    output,
                    session=sid,
                    index=int(index),
                    round_pk=pk,
                    reported_prefix_tokens=prefix,
                    previous_prompt_lcp=keep,
                    source_ready_epoch=waits.get(pk),
                    external_wait_seconds=0,
                )
            )
            previous = tokens
        if rows:
            sessions.append(dict(session=sid, requests=rows))
    return finish(
        name,
        sessions,
        writer,
        provider=provider,
        source_sha256=digest(CHAR / "tracelab/tracelab_v0.0.2.duckdb"),
        selected_session_ids=selected,
        excluded=excluded,
        reconstruction_counts=counts,
        adapter=(
            "Structural proxy: synthetic token IDs, same-session prefix continuity "
            "assumed up to reported prefix and previous prompt length; fresh "
            "identities at shrinkage. No cross-session sharing or provider-hit "
            "forcing. Generated output not linked to next synthetic prompt. "
            "Timing intensity adjustable; requests immutable."
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["inferact", "tracelab"], required=True)
    parser.add_argument("--sessions", type=int, required=True)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.kind == "inferact":
        inferact(args.sessions)
    else:
        for provider in ("codex", "claude"):
            tracelab(provider, args.sessions)
    save(
        ROOT / (args.kind + "_preparation_complete.json"), dict(sessions=args.sessions)
    )


if __name__ == "__main__":
    main()
