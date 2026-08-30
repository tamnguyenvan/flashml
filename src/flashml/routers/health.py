from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from flashml import __version__
from flashml.schemas import ErrorResponse, HealthResponse, ServiceStatus
from flashml.state import AppState

router = APIRouter(tags=["ops"])


def _services() -> dict[str, ServiceStatus]:
    settings = AppState.settings
    payload: dict[str, ServiceStatus] = {}
    mapping = {
        "reconstruct": AppState.moge,
        "interactive-segment": AppState.simpleclick,
        "segment": AppState.oneformer,
        "remove": AppState.flux,
    }
    for name, service in mapping.items():
        if not settings.is_enabled(name):  # type: ignore[arg-type]
            payload[name] = ServiceStatus(enabled=False, backend="off", ready=False)
        elif service is None:
            payload[name] = ServiceStatus(
                enabled=True,
                backend="unknown",
                ready=False,
                detail="not initialized",
            )
        else:
            payload[name] = service.status()
    return payload


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={500: {"model": ErrorResponse}},
)
async def health() -> HealthResponse:
    """Process liveness. Does not wait for GPU models to finish loading."""
    settings = AppState.settings
    return HealthResponse(
        status="ok",
        version=__version__,
        device=settings.device,
        services=_services(),
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def ready() -> HealthResponse | JSONResponse:
    """Return 200 only when every enabled local/proxy backend reports ready."""
    settings = AppState.settings
    services = _services()
    enabled = [status for name, status in services.items() if settings.is_enabled(name)]  # type: ignore[arg-type]
    all_ready = bool(enabled) and all(item.ready for item in enabled)
    body = HealthResponse(
        status="ready" if all_ready else "not_ready",
        version=__version__,
        device=settings.device,
        services=services,
    )
    if all_ready:
        return body
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump())
