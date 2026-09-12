import base64
import io

import pytest
from PIL import Image

from flashml.config import Settings
from flashml.errors import InputValidationError
from flashml.schemas import MatteRequest
from flashml.services.multimatte import (
    MultiMatteService,
    RemoteMultiMatteService,
    _resolve_prompt,
    _resolve_threshold,
)
from tests.conftest import PNG_1X1


def _settings(**kwargs) -> Settings:
    defaults = dict(_env_file=None, require_cuda=False)
    defaults.update(kwargs)
    return Settings(**defaults)


def _png_bytes(width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _request(image_b64: str, **kwargs) -> MatteRequest:
    payload = {"image": image_b64}
    payload.update(kwargs)
    return MatteRequest(**payload)


def test_resolve_prompt_defaults_and_strips():
    assert _resolve_prompt(None, "the main foreground subject") == "the main foreground subject"
    assert _resolve_prompt("  ", "fallback") == "fallback"
    assert _resolve_prompt("  the dog ", "fallback") == "the dog"


def test_resolve_prompt_too_long():
    with pytest.raises(InputValidationError):
        _resolve_prompt("x" * 501, "fallback")


def test_resolve_threshold_default_and_override():
    assert _resolve_threshold(None, 0.5) == 0.5
    assert _resolve_threshold(0.7, 0.5) == 0.7
    with pytest.raises(InputValidationError):
        _resolve_threshold(0.0, 0.5)
    with pytest.raises(InputValidationError):
        _resolve_threshold(1.0, 0.5)


def test_segment_thresholds_alpha_and_returns_png():
    import numpy as np

    service = MultiMatteService(_settings())
    service._ready = True

    # Fake (H, W) alpha in [0, 1]: half above 0.5, half below.
    alpha = np.array([[0.9, 0.1], [0.6, 0.4]], dtype=np.float32)

    class FakeTensor:
        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return alpha

    class FakeModel:
        def predict(self, processor, image, prompt, return_type="alpha"):
            assert return_type == "alpha"
            assert prompt == "the cat"
            assert image.size == (2, 2)
            return FakeTensor()

    service.model = FakeModel()
    service.processor = object()

    request = _request(base64.b64encode(_png_bytes(2, 2)).decode("ascii"), prompt="the cat")
    response = service.segment(request)
    assert response.mask_shape == [2, 2]
    assert response.prompt == "the cat"
    assert response.threshold == pytest.approx(0.5)
    mask_bytes = base64.b64decode(response.mask)
    mask = Image.open(io.BytesIO(mask_bytes))
    assert mask.mode == "L"
    assert mask.size == (2, 2)
    assert list(mask.tobytes()) == [255, 0, 255, 0]


def test_segment_downscales_large_image():
    import numpy as np

    service = MultiMatteService(_settings(multimatte_max_longest_size=100))
    service._ready = True

    seen_sizes: list[tuple[int, int]] = []

    class FakeTensor:
        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones((50, 100), dtype=np.float32)

    class FakeModel:
        def predict(self, processor, image, prompt, return_type="alpha"):
            seen_sizes.append(image.size)
            return FakeTensor()

    service.model = FakeModel()
    service.processor = object()

    request = _request(base64.b64encode(_png_bytes(200, 100)).decode("ascii"), prompt="x")
    response = service.segment(request)
    assert seen_sizes == [(100, 50)]
    assert response.mask_shape == [50, 100]


def test_remote_service_builds_from_matte_url():
    service = RemoteMultiMatteService(_settings(matte_url="http://remote:8005"))
    assert service.backend == "http"
    status = service.status()
    assert status.ready is True
    assert status.detail == "http://remote:8005"


def test_segment_1x1_fixture_image():
    import numpy as np

    service = MultiMatteService(_settings())
    service._ready = True

    class FakeTensor:
        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones((1, 1), dtype=np.float32)

    class FakeModel:
        def predict(self, processor, image, prompt, return_type="alpha"):
            return FakeTensor()

    service.model = FakeModel()
    service.processor = object()
    request = _request(base64.b64encode(PNG_1X1).decode("ascii"))
    response = service.segment(request)
    assert response.mask_shape == [1, 1]
