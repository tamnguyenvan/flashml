from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from flashml.context import request_id_ctx

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Domain error with a stable machine-readable code."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str = "bad_request",
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details


class PayloadTooLargeError(AppError):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(
            f"Payload exceeds the {max_bytes // (1024 * 1024)} MiB limit",
            status_code=413,
            code="payload_too_large",
        )


class InvalidImageError(AppError):
    def __init__(self, message: str = "Invalid image") -> None:
        super().__init__(message, status_code=400, code="invalid_image")


class InputValidationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422, code="validation_error")


class InferenceError(AppError):
    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(
            message,
            status_code=500,
            code="inference_failed",
            details=details,
        )


class DependencyUnavailableError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=503, code="dependency_unavailable")


def _body(
    *,
    error: str,
    code: str,
    details: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": error,
        "code": code,
        "request_id": request_id_ctx.get("-"),
    }
    if details is not None:
        payload["details"] = details
    return payload


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("application error: %s", exc.message, exc_info=exc)
        else:
            logger.warning("%s (%s)", exc.message, exc.code)
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(error=exc.message, code=exc.code, details=exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_body(
                error="Request validation failed",
                code="validation_error",
                details=exc.errors(),
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail
        message = detail if isinstance(detail, str) else "HTTP error"
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(error=message, code="http_error", details=detail),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled exception")
        return JSONResponse(
            status_code=500,
            content=_body(
                error="Internal server error",
                code="internal_error",
            ),
        )
