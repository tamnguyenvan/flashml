from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from flashml.deps import get_oneformer
from flashml.schemas import ErrorResponse, SegmentRequest, SegmentResponse
from flashml.services.oneformer import RemoteOneFormerService

router = APIRouter(tags=["segment"])


@router.post(
    "/segment",
    response_model=SegmentResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Semantic wall/floor/rug masks with OneFormer",
)
async def segment(
    payload: SegmentRequest,
    service=Depends(get_oneformer),
) -> SegmentResponse:
    if isinstance(service, RemoteOneFormerService):
        return await service.segment(payload)
    return await asyncio.to_thread(service.segment, payload)
