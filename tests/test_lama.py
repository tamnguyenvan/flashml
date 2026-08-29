import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

from flashml.config import Settings
from flashml.errors import InvalidImageError
from flashml.services.lama import LamaService, _decode_mask, _decode_rgb, _pil_to_png
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


def test_lama_service_remove_resizing_and_crop():
    settings = Settings(_env_file=None, require_cuda=False)
    service = LamaService(settings)
    service._ready = True

    # Mock the internal simple_lama model
    def mock_infer(image, mask):
        assert image.size == mask.size
        # Simulate simple-lama padding to multiple of 8 (e.g. 50x50 -> 56x56)
        padded_w = (image.width + 7) // 8 * 8
        padded_h = (image.height + 7) // 8 * 8
        return Image.new("RGB", (padded_w, padded_h), color="red")

    service.model = MagicMock(side_effect=mock_infer)

    # Input is 50x50 image, mask is 30x30 (size mismatch)
    img_bytes = _create_test_png(50, 50, "RGB")
    mask_bytes = _create_test_png(30, 30, "L")

    result_bytes = service.remove(img_bytes, mask_bytes, max_size=100)
    result_img = Image.open(io.BytesIO(result_bytes))

    # Output must be cropped back to 50x50 despite simple-lama internal padding
    assert result_img.size == (50, 50)


def test_lama_service_remove_downscales_large_image():
    settings = Settings(_env_file=None, require_cuda=False)
    service = LamaService(settings)
    service._ready = True

    def mock_infer(image, mask):
        assert image.size == (100, 50)
        assert mask.size == (100, 50)
        return Image.new("RGB", (100, 50), color="green")

    service.model = MagicMock(side_effect=mock_infer)

    # Input is 200x100 image, max_size=100 -> downscales to 100x50
    img_bytes = _create_test_png(200, 100, "RGB")
    mask_bytes = _create_test_png(200, 100, "L")

    result_bytes = service.remove(img_bytes, mask_bytes, max_size=100)
    result_img = Image.open(io.BytesIO(result_bytes))
    assert result_img.size == (100, 50)
