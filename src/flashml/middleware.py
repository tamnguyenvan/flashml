from __future__ import annotations

import logging
import time
import uuid

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from flashml.context import request_id_ctx

logger = logging.getLogger(__name__)

# Paths that stay public even when API-key auth is enabled.
PUBLIC_PATHS = frozenset(
    {
        "/",
        "/health",
        "/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/docs/oauth2-redirect",
    }
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid ``X-API-Key`` header (unless disabled).

    When ``allowed_keys`` is empty, all requests pass through (auth off).
    Public paths (health/readiness/docs) remain accessible without a key.
    """

    def __init__(
        self,
        app,
        allowed_keys: frozenset[str],
        header_name: str = "X-API-Key",
    ) -> None:
        super().__init__(app)
        self.allowed_keys = frozenset(allowed_keys)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.allowed_keys:
            return await call_next(request)
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        provided = request.headers.get(self.header_name)
        if provided not in self.allowed_keys:
            request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
            logger.warning(
                "rejected request %s %s (missing/invalid API key)",
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Missing or invalid API key",
                    "code": "unauthorized",
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )
        return await call_next(request)


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header_name: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(self.header_name) or str(uuid.uuid4())
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.exception(
                "unhandled error %s %s (%.1f ms)",
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise
        else:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            response.headers[self.header_name] = request_id
            response.headers["Server-Timing"] = f"app;dur={elapsed_ms:.1f}"
            logger.info(
                "%s %s -> %s (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
            return response
        finally:
            request_id_ctx.reset(token)
