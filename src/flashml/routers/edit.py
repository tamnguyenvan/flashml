from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response

from flashml.deps import get_edit
from flashml.errors import PayloadTooLargeError
from flashml.schemas import ErrorResponse
from flashml.services.flux import RemoteFluxService
from flashml.state import AppState

router = APIRouter(tags=["edit"])


@router.post(
    "/edit",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Free-prompt image edit with FLUX.2 klein",
)
async def edit(
    file: Annotated[UploadFile, File(description="RGB source image (PNG or JPEG) to edit")],
    prompt: Annotated[
        str,
        Form(description="Free-text edit instruction, e.g. 'replace the sofa with a wooden bench'"),
    ],
    max_size: Annotated[int, Form(ge=64, le=4096)] = 1024,
    seed: Annotated[int | None, Form(ge=0, le=4294967295)] = None,
    service=Depends(get_edit),
) -> Response:
    settings = AppState.settings
    image_raw = await file.read(settings.max_upload_bytes + 1)
    if len(image_raw) > settings.max_upload_bytes:
        raise PayloadTooLargeError(settings.max_upload_bytes)

    # Validate the prompt up front so the gateway rejects bad input even when
    # proxying to a worker (the worker validates again before inference).
    from flashml.services.flux import validate_edit_prompt

    cleaned_prompt = validate_edit_prompt(prompt, max_chars=settings.flux_max_prompt_chars)

    if isinstance(service, RemoteFluxService):
        content = await service.edit_remote(
            image_raw,
            image_content_type=file.content_type,
            prompt=cleaned_prompt,
            max_size=max_size,
            seed=seed,
        )
    else:
        content = await asyncio.to_thread(
            service.edit,
            image_raw,
            prompt=cleaned_prompt,
            max_size=max_size,
            seed=seed,
        )

    return Response(content, media_type="image/png")
