from flashml.config import Settings
from flashml.services.images import decode_base64_image
from flashml.services.simpleclick import _sample_points, _validate_points
from flashml.errors import InputValidationError, InvalidImageError, PayloadTooLargeError


def test_enabled_routes_all():
    settings = Settings(_env_file=None, enabled_routes="all")
    assert settings.routes == {
        "reconstruct",
        "interactive-segment",
        "segment",
        "remove",
    }


def test_sample_points_keeps_endpoints():
    points = [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]]
    sampled = _sample_points(points, 3)
    assert sampled[0] == [0, 0]
    assert sampled[-1] == [40, 0]
    assert len(sampled) == 3


def test_validate_points_rejects_out_of_bounds():
    try:
        _validate_points([[5, 1]], width=4, height=4, field_name="positive_points", required=True, max_points=24)
        raise AssertionError("expected validation error")
    except InputValidationError as exc:
        assert "outside" in exc.message


def test_decode_base64_rejects_empty():
    try:
        decode_base64_image("", max_bytes=10)
        raise AssertionError("expected invalid image")
    except InvalidImageError:
        pass


def test_decode_base64_enforces_limit():
    import base64

    payload = base64.b64encode(b"hello").decode("ascii")
    try:
        decode_base64_image(payload, max_bytes=2)
        raise AssertionError("expected payload too large")
    except PayloadTooLargeError as exc:
        assert exc.status_code == 413
