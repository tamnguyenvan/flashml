import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

from flashml.config import Settings
from flashml.errors import InvalidImageError
from flashml.services.rorem import (
    RORemService,
    RemoteRORemService,
    _decode_mask,
    _decode_rgb,
    _dilate_mask,
    _pil_to_png,
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


def test_decode_mask_valid():
    mask_bytes = _create_test_png(10, 10, mode="L", color=255)
    mask = _decode_mask(mask_bytes)
    assert isinstance(mask, Image.Image)
    assert mask.mode == "L"
    assert mask.size == (10, 10)


def test_decode_mask_invalid():
    with pytest.raises(InvalidImageError):
        _decode_mask(b"not an image")


def test_dilate_mask_no_dilation():
    mask = Image.new("L", (10, 10), color=0)
    # Draw a white square in the middle
    for x in range(4, 6):
        for y in range(4, 6):
            mask.putpixel((x, y), 255)

    result = _dilate_mask(mask, 0)
    assert result.size == (10, 10)
    assert result.mode == "L"


def test_dilate_mask_with_dilation():
    mask = Image.new("L", (10, 10), color=0)
    for x in range(4, 6):
        for y in range(4, 6):
            mask.putpixel((x, y), 255)

    result = _dilate_mask(mask, 3)
    assert result.size == (10, 10)
    assert result.mode == "L"
    # After dilation, more pixels should be white
    white_pixels_original = sum(1 for x in range(10) for y in range(10) if mask.getpixel((x, y)) > 128)
    white_pixels_dilated = sum(1 for x in range(10) for y in range(10) if result.getpixel((x, y)) > 128)
    assert white_pixels_dilated >= white_pixels_original


def test_pil_to_png():
    img = Image.new("RGB", (10, 10), color="red")
    png_bytes = _pil_to_png(img)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0
    # Verify it's a valid PNG
    loaded = Image.open(io.BytesIO(png_bytes))
    assert loaded.size == (10, 10)
    assert loaded.mode == "RGB"


def test_rorem_service_remove_resizing_and_crop():
    settings = Settings(_env_file=None, require_cuda=False)
    service = RORemService(settings)
    service._ready = True

    def mock_infer(image, mask):
        assert image.size == mask.size
        return Image.new("RGB", (image.width, image.height), color="red")

    service.pipe = MagicMock()
    service.pipe.return_value = MagicMock()
    service.pipe.__call__ = MagicMock(return_value=MagicMock(images=[Image.new("RGB", (512, 512), color="red")]))

    # Mock the actual inference path
    original_infer = service._infer_locked
    def mock_infer_locked(image, mask):
        return Image.new("RGB", (image.width, image.height), color="red")
    service._infer_locked = mock_infer_locked

    img_bytes = _create_test_png(50, 50, "RGB")
    mask_bytes = _create_test_png(30, 30, "L")

    result_bytes = service.remove(img_bytes, mask_bytes, max_size=100)
    result_img = Image.open(io.BytesIO(result_bytes))

    assert result_img.size == (50, 50)


def test_rorem_service_remove_downscales_large_image():
    settings = Settings(_env_file=None, require_cuda=False)
    service = RORemService(settings)
    service._ready = True

    def mock_infer_locked(image, mask):
        assert image.size == (100, 50)
        assert mask.size == (100, 50)
        return Image.new("RGB", (100, 50), color="green")

    service._infer_locked = mock_infer_locked

    img_bytes = _create_test_png(200, 100, "RGB")
    mask_bytes = _create_test_png(200, 100, "L")

    result_bytes = service.remove(img_bytes, mask_bytes, max_size=100)
    result_img = Image.open(io.BytesIO(result_bytes))
    assert result_img.size == (100, 50)


def test_remote_rorem_service():
    settings = Settings(_env_file=None, remove_url="http://remote:8000")
    service = RemoteRORemService(settings)
    assert service.backend == "http"
    assert service._ready is True
    status = service.status()
    assert status.backend == "http"
    assert status.ready is True
    assert status.detail == "http://remote:8000"