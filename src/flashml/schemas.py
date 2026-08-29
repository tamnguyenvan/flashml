from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    error: str
    code: str
    request_id: str
    details: object | None = None


class Point(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [[320, 240]]})

    x: float
    y: float


class InteractiveSegmentRequest(BaseModel):
    image: str = Field(
        ...,
        min_length=1,
        description="PNG/JPEG as raw base64 or a data URL (data:image/png;base64,...).",
    )
    positive_points: list[list[float]] = Field(
        ...,
        min_length=1,
        description="Foreground clicks as [x, y] in image pixel coordinates.",
    )
    negative_points: list[list[float]] = Field(
        default_factory=list,
        description="Optional background clicks as [x, y].",
    )
    threshold: float = Field(
        0.49,
        gt=0.0,
        lt=1.0,
        description="Probability cutoff used to binarize the SimpleClick mask.",
    )


class InteractiveSegmentResponse(BaseModel):
    mask: str = Field(..., description="PNG mask encoded as base64 (no data URL prefix).")
    mask_format: str = "png"
    mask_shape: list[int] = Field(..., min_length=2, max_length=2)
    positive_points_used: list[list[int]]
    negative_points_used: list[list[int]]
    threshold: float


class SegmentRequest(BaseModel):
    image: str = Field(
        ...,
        min_length=1,
        description="PNG/JPEG as raw base64 or a data URL.",
    )


class SurfaceMask(BaseModel):
    mask: str = Field(..., description="PNG mask as a data URL.")
    score: float | None = None
    box: list[float] | None = None


class SegmentResponse(BaseModel):
    model: str
    provider: str = "oneformer"
    image_size_hw: list[int] = Field(..., min_length=2, max_length=2)
    label_ids: dict[str, list[int]]
    masks: dict[str, list[SurfaceMask]]


class ServiceStatus(BaseModel):
    enabled: bool
    backend: str
    ready: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    services: dict[str, ServiceStatus]
