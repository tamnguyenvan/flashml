from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response

from flashml.deps import get_flux
from flashml.errors import PayloadTooLargeError
from flashml.schemas import ErrorResponse
from flashml.services.flux import RemoteFluxService
from flashml.state import AppState

router = APIRouter(tags=["remove"])


@router.post(
    "/remove",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Remove objects from an image with FLUX.2 klein (object-removal LoRA)",
)
async def remove(
    file: Annotated[UploadFile, File(description="RGB image (PNG or JPEG)")],
    mask: Annotated[UploadFile, File(description="Binary mask PNG (white = region to remove)")],
    max_size: Annotated[int, Form(ge=64, le=4096)] = 1024,
    service=Depends(get_flux),
) -> Response:
    settings = AppState.settings
    image_raw = await file.read(settings.max_upload_bytes + 1)
    if len(image_raw) > settings.max_upload_bytes:
        raise PayloadTooLargeError(settings.max_upload_bytes)
    mask_raw = await mask.read(settings.max_upload_bytes + 1)
    if len(mask_raw) > settings.max_upload_bytes:
        raise PayloadTooLargeError(settings.max_upload_bytes)

    if isinstance(service, RemoteFluxService):
        content = await service.remove_remote(
            image_raw,
            mask_raw,
            image_content_type=file.content_type,
            mask_content_type=mask.content_type,
            max_size=max_size,
        )
    else:
        content = await asyncio.to_thread(
            service.remove,
            image_raw,
            mask_raw,
            max_size=max_size,
        )

    return Response(content, media_type="image/png")

