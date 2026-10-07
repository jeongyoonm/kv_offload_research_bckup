# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ipaddress
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from vllm.engine.protocol import EngineClient
from vllm.v1.kv_offload.early_readiness import (
    EARLY_COMPLETION_KEY,
    SCHEDULE_DELAY_KEY,
)

router = APIRouter()


class KVLoadSchedulingRequest(BaseModel):
    early_load_completion: bool = False
    schedule_delay_ms: float = Field(default=0, ge=0, le=120)
    experiment_options: dict[str, Any] | None = None
    seal_estimator: dict[str, Any] | None = None


@router.post("/experimental/kv-load-scheduling")
async def set_kv_load_scheduling(
    settings: KVLoadSchedulingRequest, raw_request: Request
):
    client = raw_request.client
    try:
        is_loopback = (
            client is not None and ipaddress.ip_address(client.host).is_loopback
        )
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise HTTPException(status_code=403, detail="loopback clients only")

    config = raw_request.app.state.vllm_config.kv_transfer_config
    extra = config.kv_connector_extra_config if config is not None else {}
    if EARLY_COMPLETION_KEY not in extra and SCHEDULE_DELAY_KEY not in extra:
        raise HTTPException(status_code=404, detail="experiment is not enabled")

    engine: EngineClient = raw_request.app.state.engine_client
    try:
        if settings.seal_estimator is not None:
            return await engine.set_kv_load_scheduling(
                settings.early_load_completion,
                settings.schedule_delay_ms,
                settings.experiment_options,
                settings.seal_estimator,
            )
        if settings.experiment_options is not None:
            return await engine.set_kv_load_scheduling(
                settings.early_load_completion,
                settings.schedule_delay_ms,
                settings.experiment_options,
            )
        return await engine.set_kv_load_scheduling(
            settings.early_load_completion, settings.schedule_delay_ms
        )
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        # Utility RPC failures lose their original exception type in transit.
        # Retry only the known drain rejection; keep unrelated failures visible.
        if (
            "KV load scheduling settings require requests, queued steps, "
            "and connector work to be drained"
        ) in str(error):
            raise HTTPException(status_code=409, detail=str(error)) from error
        if any(
            message in str(error)
            for message in (
                "invalid seal estimator",
                "incompatible seal estimator",
                "seal estimator requires execution timing",
                "seal estimator has no trained coefficients",
            )
        ):
            raise HTTPException(status_code=422, detail=str(error)) from error
        raise


def attach_router(app: FastAPI) -> None:
    app.include_router(router)


@router.get("/experimental/deferral-cohort")
async def get_deferral_cohort(raw_request: Request):
    client = raw_request.client
    try:
        loopback = client is not None and ipaddress.ip_address(client.host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise HTTPException(status_code=403, detail="loopback clients only")
    config = raw_request.app.state.vllm_config.kv_transfer_config
    extra = config.kv_connector_extra_config if config is not None else {}
    if EARLY_COMPLETION_KEY not in extra and SCHEDULE_DELAY_KEY not in extra:
        raise HTTPException(status_code=404, detail="experiment is not enabled")
    engine: EngineClient = raw_request.app.state.engine_client
    return await engine.get_deferral_cohort_status()
