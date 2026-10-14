# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real Rust runner against a barrier-controlled CPU server."""

import csv
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("/mnt/localnvme/jeongyoon/tracelab_request_slots_20261013")
BINARY = ROOT / "build/release/session_runner"
MODEL = Path(
    "/mnt/localnvme/huggingface/hub/models--Inferact--GLM-5.3-NVFP4/"
    "snapshots/ce67b36f3669192b5bb233819f0fda6c8a9837f8"
)
CORPUS = Path("/mnt/localnvme/jeongyoon/tracelab_runner_20261013/corpus/enwik9")


class Mock:
    def __init__(self, pair_barrier=False):
        self.lock = threading.Lock()
        self.calls = []
        self.cached = set()
        self.active = self.peak = 0
        self.barrier = threading.Barrier(2) if pair_barrier else None

    def handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                request = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                assert request["return_token_ids"] and request["ignore_eos"]
                probe = not request["stream"]
                if not probe:
                    assert request.get("request_id"), request
                key = tuple(request["prompt"])
                with owner.lock:
                    cached = len(key) if key in owner.cached else 0
                    owner.cached.add(key)
                    if not probe:
                        owner.active += 1
                        owner.peak = max(owner.peak, owner.active)
                        owner.calls.append(dict(time=time.time(), request=request))
                if not probe:
                    if owner.barrier:
                        owner.barrier.wait(timeout=5)
                    time.sleep(0.02)
                usage = dict(
                    prompt_tokens=len(key),
                    completion_tokens=1,
                    total_tokens=len(key) + 1,
                    prompt_tokens_details=dict(cached_tokens=cached),
                )
                choice = dict(
                    index=0, text="x", token_ids=[198], finish_reason="length"
                )
                result = dict(choices=[choice], usage=usage)
                if probe:
                    body = json.dumps(result).encode()
                    content_type = "application/json"
                else:
                    body = (
                        "data: " + json.dumps(result) + "\n\ndata: [DONE]\n\n"
                    ).encode()
                    content_type = "text/event-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                if not probe:
                    with owner.lock:
                        owner.active -= 1

        return Handler


def run_case(name, rows, limit, run_seconds=None, barrier=False, legacy=False):
    directory = ROOT / "cpu_checks" / name
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "trace.csv"
    with path.open("w") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "id",
                "input_len",
                "output_len",
                "arrival_time",
                "round_idx",
                "tool_wait_after_ms",
                "prefix_len",
            ]
        )
        writer.writerows(rows)
    mock = Mock(barrier)
    server = ThreadingHTTPServer(("127.0.0.1", 0), mock.handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    command = [
        str(BINARY),
        "--trace",
        str(path),
        "--text-file",
        str(CORPUS),
        "--tokenizer",
        str(MODEL),
        "--model",
        "cpu-mock",
        "--token-pool-limit",
        "4096",
        "--base-url",
        f"http://127.0.0.1:{server.server_port}/v1",
        "--log-path",
        str(directory / "rounds.jsonl"),
        "--summary-path",
        str(directory / "summary.json"),
        "--max-active-sessions" if legacy else "--max-in-flight-requests",
        str(limit),
    ]
    if run_seconds is not None:
        command += ["--run-seconds", str(run_seconds)]
    try:
        started = time.monotonic()
        with (directory / "runner.log").open("w") as output:
            subprocess.run(
                command, stdout=output, stderr=output, timeout=15, check=True
            )
        duration = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
    logged = [
        json.loads(line)
        for line in (directory / "rounds.jsonl").read_text().splitlines()
    ]
    assert all(row["status"] == "SUCCESS" for row in logged), logged
    assert all(
        row["server_completion_tokens"] == row["output_len_target"] for row in logged
    )
    assert mock.peak <= limit, mock.peak
    return logged, mock, duration


def main():
    rows = [
        ["a", 64, 1, 0, 0, 1000, 0],
        ["a", 1, 1, 0, 1, 0, 65],
        ["a", 1, 1, 0, 2, 0, 16],
        ["b", 64, 1, 100, 0, 0, 0],
    ]
    logged, mock, _ = run_case("sleep_releases_slot", rows, 1)
    indexed = {(r["session_id"], r["round_idx"]): r for r in logged}
    first, second, other = indexed[("a", 0)], indexed[("a", 1)], indexed[("b", 0)]
    assert (
        first["complete_timestamp"] < other["post_timestamp"] < second["post_timestamp"]
    )
    assert second["post_timestamp"] - first["complete_timestamp"] >= 0.95
    calls = mock.calls
    assert calls[2]["request"]["prompt"][:65] == calls[0]["request"]["prompt"] + [198]
    assert calls[3]["request"]["prompt"][:16] == calls[2]["request"]["prompt"][:16]
    burst = [[str(i), 64, 1, 0, 0, 0, 0] for i in range(8)]
    logged2, mock2, _ = run_case("bounded_burst", burst, 2, barrier=True)
    assert len(logged2) == 8 and mock2.peak == 2
    assert max(r["request_slot_wait_ms"] for r in logged2) > 10
    bounded = [
        ["a", 64, 1, 0, 0, 60000, 0],
        ["a", 1, 1, 0, 1, 0, 65],
        ["future", 64, 1, 60000, 0, 0, 0],
    ]
    logged3, _, duration = run_case("deadline_drains", bounded, 1, run_seconds=0.25)
    assert len(logged3) == 1 and duration < 5
    legacy, _, _ = run_case("legacy_session_cap", rows, 1, legacy=True)
    indexed = {(r["session_id"], r["round_idx"]): r for r in legacy}
    assert indexed[("b", 0)]["post_timestamp"] > indexed[("a", 2)]["complete_timestamp"]
    result = dict(
        passed=True,
        cases=[
            "Sleeping agent releases slot and keeps 1s wait",
            "Exact output carry-forward and shorter prefix retained",
            "Barrier burst capped at two calls; queued requests wait",
            "Bounded run drains submitted work and stops sleeping/future agents",
            "Legacy session-cap behavior unchanged",
        ],
        max_burst_in_flight=mock2.peak,
    )
    (ROOT / "cpu_checks/result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
