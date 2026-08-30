from __future__ import annotations

import base64
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from flashml.app import create_app
from flashml.config import Settings
from flashml.schemas import (
    InteractiveSegmentResponse,
    SegmentResponse,
    ServiceStatus,
    SurfaceMask,
)
from flashml.state import AppState

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _settings(**kwargs) -> Settings:
    defaults = dict(
        preload=False,
        require_cuda=False,
        json_logs=False,
        enabled_routes="all",
        reconstruct_url=None,
        interactive_segment_url=None,
        segment_url=None,
    )
    defaults.update(kwargs)
    return Settings(_env_file=None, **defaults)


class FakeMoge:
    backend = "local"

    def preload(self) -> None:
        return None

    def status(self) -> ServiceStatus:
        return ServiceStatus(enabled=True, backend="local", ready=True, detail="fake")

    def reconstruct(self, raw: bytes, **kwargs) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as archive:
            archive.writestr("metadata.json", json.dumps({"filename": kwargs["filename"]}))
            archive.writestr("point_map.npy", b"n")
        return buffer.getvalue()


class FakeSimpleClick:
    backend = "local"

    def preload(self) -> None:
        return None

    def status(self) -> ServiceStatus:
        return ServiceStatus(enabled=True, backend="local", ready=True, detail="fake")

    def segment(self, request):
        return InteractiveSegmentResponse(
            mask=base64.b64encode(PNG_1X1).decode("ascii"),
            mask_format="png",
            mask_shape=[1, 1],
            positive_points_used=[[0, 0]],
            negative_points_used=[],
            threshold=request.threshold,
        )


class FakeOneFormer:
    backend = "local"

    def preload(self) -> None:
        return None

    def status(self) -> ServiceStatus:
        return ServiceStatus(enabled=True, backend="local", ready=True, detail="fake")

    def segment(self, request):
        return SegmentResponse(
            model="fake",
            provider="oneformer",
            image_size_hw=[1, 1],
            label_ids={"wall": [0], "floor": [3], "rug": [7]},
            masks={
                "wall": [
                    SurfaceMask(mask="data:image/png;base64,xx", score=None, box=None)
                ],
                "floor": [],
                "rug": [],
            },
        )


class FakeFlux:
    backend = "local"

    def preload(self) -> None:
        return None

    def status(self) -> ServiceStatus:
        return ServiceStatus(enabled=True, backend="local", ready=True, detail="fake")

    def remove(self, image_bytes: bytes, *, max_size: int) -> bytes:
        return PNG_1X1


@pytest.fixture
def client():
    app = create_app(_settings())
    with TestClient(app) as test_client:
        AppState.moge = FakeMoge()
        AppState.simpleclick = FakeSimpleClick()
        AppState.oneformer = FakeOneFormer()
        AppState.flux = FakeFlux()
        yield test_client


@pytest.fixture
def client_auth():
    app = create_app(_settings(api_keys="secret-1,secret-2"))
    with TestClient(app) as test_client:
        AppState.moge = FakeMoge()
        AppState.simpleclick = FakeSimpleClick()
        AppState.oneformer = FakeOneFormer()
        AppState.flux = FakeFlux()
        yield test_client
