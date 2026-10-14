# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve retryable drain rejection across the utility RPC boundary."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from vllm.entrypoints.serve.experimental.kv_load_scheduling import (
    KVLoadSchedulingRequest,
    set_kv_load_scheduling,
)
from vllm.v1.kv_offload.early_readiness import EARLY_COMPLETION_KEY

DRAIN = (
    "KV load scheduling settings require requests, queued steps, "
    "and connector work to be drained"
)


@pytest.mark.parametrize(
    "error,status",
    [
        (RuntimeError(DRAIN), 409),
        (Exception("Call to set_kv_load_scheduling method failed: " + DRAIN), 409),
        (ValueError("invalid option"), 422),
    ],
)
def test_endpoint_maps_known_configuration_rejections(error, status):
    engine = SimpleNamespace(set_kv_load_scheduling=AsyncMock(side_effect=error))
    state = SimpleNamespace(
        engine_client=engine,
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector_extra_config={EARLY_COMPLETION_KEY: False}
            )
        ),
    )
    request = SimpleNamespace(
        client=SimpleNamespace(host="127.0.0.1"),
        app=SimpleNamespace(state=state),
    )
    with pytest.raises(HTTPException) as caught:
        asyncio.run(set_kv_load_scheduling(KVLoadSchedulingRequest(), request))
    assert caught.value.status_code == status
    assert str(error) in caught.value.detail


def test_endpoint_does_not_retry_unrelated_rpc_failure():
    error = Exception("worker configuration rank coverage mismatch")
    engine = SimpleNamespace(set_kv_load_scheduling=AsyncMock(side_effect=error))
    request = SimpleNamespace(
        client=SimpleNamespace(host="127.0.0.1"),
        app=SimpleNamespace(
            state=SimpleNamespace(
                engine_client=engine,
                vllm_config=SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(
                        kv_connector_extra_config={EARLY_COMPLETION_KEY: False}
                    )
                ),
            )
        ),
    )
    with pytest.raises(Exception, match="rank coverage mismatch") as caught:
        asyncio.run(set_kv_load_scheduling(KVLoadSchedulingRequest(), request))
    assert caught.value is error
