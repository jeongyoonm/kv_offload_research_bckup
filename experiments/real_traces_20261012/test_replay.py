# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify output accounting and sequential turns before GPU trace replay."""

import asyncio
import io
import json

import aiohttp
from aiohttp import web

from experiments.real_traces_20261012 import replay


def test_full_stream_output_is_required(monkeypatch):
    async def run():
        async def handler(request):
            payload = await request.json()
            assert payload["ignore_eos"]
            assert payload["max_tokens"] == payload["min_tokens"] == 3
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for i in range(3):
                await response.write(
                    (
                        "data: "
                        + json.dumps(
                            {"choices": [{"text": str(i), "finish_reason": None}]}
                        )
                        + "\n\n"
                    ).encode()
                )
            await response.write(
                (
                    "data: "
                    + json.dumps(
                        {
                            "choices": [],
                            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                        }
                    )
                    + "\n\n"
                ).encode()
            )
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/v1/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        monkeypatch.setattr(
            replay,
            "BASE",
            f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}",
        )
        try:
            async with aiohttp.ClientSession() as session:
                result = await replay.request(
                    session,
                    asyncio.Semaphore(1),
                    dict(index=0, prompt=[1, 2], output_tokens=3),
                    "test",
                    0,
                    io.StringIO(),
                )
            assert result["accepted"] and result["stream_chunks"] == 3
            assert result["first_ns"] <= result["end_ns"]
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_inferact_turn_waits_for_previous_completion(monkeypatch, tmp_path):
    events = []

    async def fake(session, sem, row, name, scheduled, sink):
        async with sem:
            events.append(("start", row["session"], row["index"]))
            await asyncio.sleep(0)
            events.append(("end", row["session"], row["index"]))
        return dict(accepted=True)

    monkeypatch.setattr(replay, "request", fake)
    data = dict(
        kind="inferact",
        sessions=[
            dict(session=s, requests=[dict(index=t, session=s) for t in range(3)])
            for s in range(2)
        ],
    )
    responses, _, _, expected, _ = asyncio.run(
        replay.issue(data, dict(concurrency=1, sessions=2), tmp_path)
    )
    assert len(responses) == expected == 6
    for session in range(2):
        own = [(phase, t) for phase, s, t in events if s == session]
        assert own == [(phase, t) for t in range(3) for phase in ("start", "end")]


def test_request_ids_are_not_confused_when_one_is_prefix():
    response = dict(
        accepted=True,
        request="r-1",
        first_ns=1,
        send_ns=0,
        scheduled_ns=0,
        end_ns=2,
        usage=dict(
            prompt_tokens=10,
            completion_tokens=3,
            prompt_tokens_details=dict(cached_tokens=5),
        ),
    )
    events = [
        dict(
            event="job_observed_done",
            job_id=0,
            pid=r,
            request="cmpl-r-10-unique",
            transfer_bytes=64,
        )
        for r in range(8)
    ]
    result = replay.summarize([response], 0, 10, 1, None, events)
    assert result["accepted"]
    assert result["realized_outcomes"] == {"gpu_hit": 1}
