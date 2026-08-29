from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from flashml.deps import get_simpleclick
from flashml.schemas import (
    ErrorResponse,
    InteractiveSegmentRequest,
    InteractiveSegmentResponse,
)
from flashml.services.simpleclick import RemoteSimpleClickService

router = APIRouter(tags=["interactive-segment"])


@router.post(
    "/interactive-segment",
    response_model=InteractiveSegmentResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Interactive object mask from clicks with SimpleClick",
)
async def interactive_segment(
    payload: InteractiveSegmentRequest,
    service=Depends(get_simpleclick),
) -> InteractiveSegmentResponse:
    if isinstance(service, RemoteSimpleClickService):
        return await service.segment(payload)
    return await asyncio.to_thread(service.segment, payload)
