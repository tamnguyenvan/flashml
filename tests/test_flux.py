import io

import pytest
from PIL import Image

from flashml.config import Settings
from flashml.errors import InvalidImageError
from flashml.services.flux import (
    FluxService,
    RemoteFluxService,
    _decode_rgb,
    _pil_to_png,
    _round_dims,
)
from tests.conftest import PNG_1X1


def _create_test_png(width: int, height: int, mode: str = "RGB", color="blue") -> bytes:
    img = Image.new(mode, (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_rgb_valid():
    img = _decode_rgb(PNG_1X1)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (1, 1)


def test_decode_rgb_invalid():
    with pytest.raises(InvalidImageError):
        _decode_rgb(b"corrupted bytes")


def test_pil_to_png():
    img = Image.new("RGB", (10, 10), color="red")
    png_bytes = _pil_to_png(img)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0
    loaded = Image.open(io.BytesIO(png_bytes))
    assert loaded.size == (10, 10)
    assert loaded.mode == "RGB"


def test_flux_service_remove_resizing_and_crop():
    settings = Settings(_env_file=None, require_cuda=False)
    service = FluxService(settings)
    service._ready = True

    def mock_infer_locked(conditioning):
        return Image.new("RGB", (conditioning.width, conditioning.height), color="red")

    service._infer_locked = mock_infer_locked

    img_bytes = _create_test_png(50, 50, "RGB")

    result_bytes = service.remove(img_bytes, max_size=100)
    result_img = Image.open(io.BytesIO(result_bytes))
    assert result_img.size == (50, 50)


def test_flux_service_remove_downscales_large_image():
    settings = Settings(_env_file=None, require_cuda=False)
    service = FluxService(settings)
    service._ready = True

    def mock_infer_locked(conditioning):
        assert conditioning.size == (100, 50)
        return Image.new("RGB", (100, 50), color="green")

    service._infer_locked = mock_infer_locked

    img_bytes = _create_test_png(200, 100, "RGB")

    result_bytes = service.remove(img_bytes, max_size=100)
    result_img = Image.open(io.BytesIO(result_bytes))
    assert result_img.size == (100, 50)


def test_round_dims_rounds_up_to_multiple():
    assert _round_dims(500, 300, 32) == (512, 320)
    assert _round_dims(1024, 1024, 32) == (1024, 1024)
    assert _round_dims(1, 1, 32) == (32, 32)


def test_remote_flux_service():
    settings = Settings(_env_file=None, remove_url="http://remote:8000")
    service = RemoteFluxService(settings)
    assert service.backend == "http"
    assert service._ready is True
    status = service.status()
    assert status.backend == "http"
    assert status.ready is True
    assert status.detail == "http://remote:8000"
