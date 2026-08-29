from __future__ import annotations

import asyncio
import io
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from flashml.deps import get_moge
from flashml.errors import PayloadTooLargeError
from flashml.schemas import ErrorResponse
from flashml.services.moge import RemoteMogeService
from flashml.state import AppState

router = APIRouter(tags=["reconstruct"])


@router.post(
    "/reconstruct",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Reconstruct depth, point map, and optional mesh with MoGe-3",
)
@router.post("/predict", include_in_schema=False)
async def reconstruct(
    file: Annotated[UploadFile, File(description="RGB image (PNG or JPEG)")],
    include_mesh: Annotated[bool, Form()] = True,
    include_debug: Annotated[bool, Form()] = True,
    max_size: Annotated[int, Form(ge=64, le=2048)] = 800,
    resolution_level: Annotated[int, Form(ge=0, le=9)] = 9,
    num_tokens: Annotated[int | None, Form(ge=1200, le=3600)] = None,
    refine_steps: Annotated[int, Form(ge=0, le=8)] = 3,
    fov_x: Annotated[float | None, Form(gt=0.0, lt=180.0)] = None,
    edge_threshold: Annotated[float, Form(ge=0.0)] = 0.04,
    service=Depends(get_moge),
) -> StreamingResponse:
    settings = AppState.settings
    raw = await file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise PayloadTooLargeError(settings.max_upload_bytes)

    filename = file.filename or "image"
    kwargs = dict(
        filename=filename,
        include_mesh=include_mesh,
        include_debug=include_debug,
        max_size=max_size,
        resolution_level=resolution_level,
        num_tokens=num_tokens,
        refine_steps=refine_steps,
        fov_x=fov_x,
        edge_threshold=edge_threshold,
    )
    if isinstance(service, RemoteMogeService):
        archive = await service.reconstruct_remote(
            raw,
            content_type=file.content_type,
            **kwargs,
        )
    else:
        archive = await asyncio.to_thread(service.reconstruct, raw, **kwargs)

    return StreamingResponse(
        io.BytesIO(archive),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=output.zip"},
    )
