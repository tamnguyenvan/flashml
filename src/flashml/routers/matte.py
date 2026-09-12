from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from flashml.deps import get_multimatte
from flashml.schemas import ErrorResponse, MatteRequest, MatteResponse
from flashml.services.multimatte import RemoteMultiMatteService

router = APIRouter(tags=["matte"])


@router.post(
    "/matte",
    response_model=MatteResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Promptable object mask with MultiMatte",
)
async def matte(
    payload: MatteRequest,
    service=Depends(get_multimatte),
) -> MatteResponse:
    if isinstance(service, RemoteMultiMatteService):
        return await service.segment(payload)
    return await asyncio.to_thread(service.segment, payload)
