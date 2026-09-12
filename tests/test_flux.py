import io

import pytest
from PIL import Image

from flashml.config import Settings
from flashml.errors import InputValidationError, InvalidImageError
from flashml.services.flux import (
    FluxService,
    RemoteFluxService,
    _decode_rgb,
    _lora_disabled,
    _pil_to_png,
    _round_dims,
    build_flux_service,
    validate_edit_prompt,
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


def test_flux_service_edit_resizing_and_crop():
    settings = Settings(_env_file=None, require_cuda=False)
    service = FluxService(settings)
    service._ready = True

    def mock_infer_locked(conditioning, prompt=None, generator=None):
        return Image.new("RGB", (conditioning.width, conditioning.height), color="red")

    service._infer_locked = mock_infer_locked

    img_bytes = _create_test_png(50, 50, "RGB")

    result_bytes = service.edit(img_bytes, prompt="repaint", max_size=100)
    result_img = Image.open(io.BytesIO(result_bytes))
    assert result_img.size == (50, 50)


def test_flux_service_edit_downscales_large_image():
    settings = Settings(_env_file=None, require_cuda=False)
    service = FluxService(settings)
    service._ready = True

    def mock_infer_locked(conditioning, prompt=None, generator=None):
        assert conditioning.size == (100, 50)
        return Image.new("RGB", (100, 50), color="green")

    service._infer_locked = mock_infer_locked

    img_bytes = _create_test_png(200, 100, "RGB")

    result_bytes = service.edit(img_bytes, prompt="repaint", max_size=100)
    result_img = Image.open(io.BytesIO(result_bytes))
    assert result_img.size == (100, 50)


def test_round_dims_rounds_up_to_multiple():
    assert _round_dims(500, 300, 32) == (512, 320)
    assert _round_dims(1024, 1024, 32) == (1024, 1024)
    assert _round_dims(1, 1, 32) == (32, 32)


def test_remote_flux_service():
    settings = Settings(_env_file=None, edit_url="http://remote:8004")
    service = RemoteFluxService(settings)
    assert service.backend == "http"
    assert service._ready is True
    status = service.status()
    assert status.backend == "http"
    assert status.ready is True
    assert status.detail == "http://remote:8004"


def test_build_flux_service_prefers_local_without_url():
    settings = Settings(_env_file=None)
    assert isinstance(build_flux_service(settings), FluxService)


def test_build_flux_service_remote_with_edit_url():
    settings = Settings(_env_file=None, edit_url="http://remote:8004")
    assert isinstance(build_flux_service(settings), RemoteFluxService)


def test_validate_edit_prompt():
    assert validate_edit_prompt("  make it sunset  ", max_chars=2000) == "make it sunset"
    with pytest.raises(InputValidationError):
        validate_edit_prompt("   ", max_chars=2000)
    with pytest.raises(InputValidationError):
        validate_edit_prompt("", max_chars=2000)
    with pytest.raises(InputValidationError):
        validate_edit_prompt(None, max_chars=2000)
    with pytest.raises(InputValidationError):
        validate_edit_prompt("x" * 2001, max_chars=2000)


def test_edit_forwards_prompt_and_seed(monkeypatch):
    import sys
    import types

    settings = Settings(_env_file=None, require_cuda=False)
    service = FluxService(settings)
    service._ready = True
    service.device = "cpu"

    seen: dict = {}

    class FakeGenerator:
        def __init__(self, device=None):
            self.device = device
            self._seed = None

        def manual_seed(self, seed):
            self._seed = seed
            return self

    fake_torch = types.SimpleNamespace(Generator=FakeGenerator)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def mock_infer_locked(conditioning, prompt=None, generator=None):
        seen["prompt"] = prompt
        seen["generator"] = generator
        seen["size"] = conditioning.size
        return Image.new("RGB", (conditioning.width, conditioning.height), color="red")

    service._infer_locked = mock_infer_locked

    result_bytes = service.edit(
        _create_test_png(50, 50, "RGB"),
        prompt="  make it sunset  ",
        max_size=100,
        seed=7,
    )
    assert seen["prompt"] == "make it sunset"
    assert seen["size"] == (50, 50)
    assert isinstance(seen["generator"], FakeGenerator)
    assert seen["generator"]._seed == 7
    assert Image.open(io.BytesIO(result_bytes)).size == (50, 50)


def test_edit_without_seed_passes_no_generator():
    settings = Settings(_env_file=None, require_cuda=False)
    service = FluxService(settings)
    service._ready = True
    service.device = "cpu"

    seen: dict = {}

    def mock_infer_locked(conditioning, prompt=None, generator=None):
        seen["generator"] = generator
        return Image.new("RGB", (conditioning.width, conditioning.height), color="red")

    service._infer_locked = mock_infer_locked
    service.edit(_create_test_png(20, 20, "RGB"), prompt="repaint", max_size=100)
    assert seen["generator"] is None


def test_edit_validates_prompt_before_inference():
    settings = Settings(_env_file=None, require_cuda=False)
    service = FluxService(settings)
    service._ready = True
    with pytest.raises(InputValidationError):
        service.edit(_create_test_png(10, 10, "RGB"), prompt="  ", max_size=100)


def test_lora_disabled_uses_adapter_context():
    entered: list[bool] = []

    class FakeTransformer:
        from contextlib import contextmanager

        @contextmanager
        def disable_adapter(self):
            entered.append(True)
            yield

    class FakePipe:
        transformer = FakeTransformer()

    with _lora_disabled(FakePipe()):
        pass
    assert entered == [True]


def test_lora_disabled_falls_back_without_adapter():
    class FakePipe:
        transformer = object()

    with _lora_disabled(FakePipe()):
        pass
