# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify expired idle connections are replaced without changing agent tool waits."""

import json

from experiments.tracelab_arrival_intensity_20261013.prepare import ROOT
from experiments.tracelab_request_slots_20261013 import check_runner


class ConnectionMock(check_runner.Mock):
    def handler(self):
        handler = super().handler()
        handler.protocol_version = "HTTP/1.1"
        owner = self

        class Tracked(handler):
            def do_POST(self):
                before = len(owner.calls)
                super().do_POST()
                if len(owner.calls) > before:
                    owner.calls[-1]["client_port"] = self.client_address[1]

        return Tracked


def main():
    check_runner.ROOT = ROOT / "keepalive_checks"
    check_runner.BINARY = ROOT / "build_keepalive/release/session_runner"
    check_runner.Mock = ConnectionMock
    rows = [["a", 64, 1, 0, 0, 2000, 0], ["a", 1, 1, 0, 1, 0, 65]]
    logged, mock, duration = check_runner.run_case("idle_expiry", rows, 1)
    assert len(logged) == 2 and len(mock.calls) == 2
    assert mock.calls[0]["client_port"] != mock.calls[1]["client_port"]
    wait = logged[1]["post_timestamp"] - logged[0]["complete_timestamp"]
    assert wait >= 1.95
    assert mock.calls[1]["request"]["prompt"][:65] == (
        mock.calls[0]["request"]["prompt"] + [198]
    )
    result = dict(
        passed=True,
        cases=[
            "HTTP/1.1 connection renewed after a recorded 2s tool wait",
            "No retries or duplicate calls",
            "Exact real-output token ID carried into next prompt",
            "Tool wait preserved",
        ],
        client_ports=[call["client_port"] for call in mock.calls],
        observed_wait_seconds=wait,
        harness_seconds=duration,
    )
    (ROOT / "keepalive_checks/result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
