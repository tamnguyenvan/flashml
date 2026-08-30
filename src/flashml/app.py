from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from flashml import __version__
from flashml.config import Settings, get_settings
from flashml.errors import register_exception_handlers
from flashml.logging import setup_logging
from flashml.middleware import APIKeyMiddleware, RequestContextMiddleware
from flashml.routers import health, interactive_segment, reconstruct, remove, segment
from flashml.services.flux import build_flux_service
from flashml.services.moge import build_moge_service
from flashml.services.oneformer import build_oneformer_service
from flashml.services.simpleclick import build_simpleclick_service
from flashml.state import AppState

logger = logging.getLogger(__name__)


def _load_enabled_services(settings: Settings) -> None:
    if settings.is_enabled("reconstruct"):
        logger.info("Initializing reconstruct backend")
        AppState.moge = build_moge_service(settings)
        if settings.preload and settings.reconstruct_url is None:
            AppState.moge.preload()
    if settings.is_enabled("interactive-segment"):
        logger.info("Initializing interactive-segment backend")
        AppState.simpleclick = build_simpleclick_service(settings)
        if settings.preload and settings.interactive_segment_url is None:
            AppState.simpleclick.preload()
    if settings.is_enabled("segment"):
        logger.info("Initializing segment backend")
        AppState.oneformer = build_oneformer_service(settings)
        if settings.preload and settings.segment_url is None:
            AppState.oneformer.preload()
    if settings.is_enabled("remove"):
        logger.info("Initializing remove backend")
        AppState.flux = build_flux_service(settings)
        if settings.preload and settings.remove_url is None:
            AppState.flux.preload()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    setup_logging(level=settings.log_level, json_logs=settings.json_logs)
    logger.info("FlashML %s starting (routes=%s)", __version__, sorted(settings.routes))
    await asyncio.to_thread(_load_enabled_services, settings)
    logger.info("FlashML ready")
    try:
        yield
    finally:
        logger.info("FlashML shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="FlashML Inference API",
        version=__version__,
        description=(
            "Unified REST API for MoGe reconstruction, SimpleClick interactive "
            "segmentation, and OneFormer semantic segmentation."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    AppState.settings = settings

    register_exception_handlers(app)
    app.add_middleware(RequestContextMiddleware, header_name=settings.request_id_header)
    app.add_middleware(APIKeyMiddleware, allowed_keys=settings.enabled_api_keys)

    app.include_router(health.router)
    if settings.is_enabled("reconstruct"):
        app.include_router(reconstruct.router)
    if settings.is_enabled("interactive-segment"):
        app.include_router(interactive_segment.router)
    if settings.is_enabled("segment"):
        app.include_router(segment.router)
    if settings.is_enabled("remove"):
        app.include_router(remove.router)

    return app


app = create_app()
