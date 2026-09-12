import base64

import pytest

from tests.conftest import PNG_1X1


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["services"]["reconstruct"]["ready"] is True
    assert "X-Request-ID" in response.headers


def test_ready_ok(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_reconstruct_returns_zip(client):
    response = client.post(
        "/reconstruct",
        files={"file": ("tiny.png", PNG_1X1, "image/png")},
        data={"include_mesh": "false", "include_debug": "false"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")


def test_predict_alias(client):
    response = client.post(
        "/predict",
        files={"file": ("tiny.png", PNG_1X1, "image/png")},
    )
    assert response.status_code == 200


def test_interactive_segment_validation_error(client):
    response = client.post(
        "/interactive-segment",
        json={"image": "not-base64", "positive_points": []},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert "request_id" in body


def test_interactive_segment_ok(client):
    image = base64.b64encode(PNG_1X1).decode("ascii")
    response = client.post(
        "/interactive-segment",
        json={"image": image, "positive_points": [[0, 0]], "threshold": 0.49},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mask_format"] == "png"
    assert body["mask_shape"] == [1, 1]


def test_segment_requires_image(client):
    response = client.post("/segment", json={})
    assert response.status_code == 422


def test_segment_ok(client):
    image = "data:image/png;base64," + base64.b64encode(PNG_1X1).decode("ascii")
    response = client.post("/segment", json={"image": image})
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "oneformer"
    assert "wall" in body["masks"]


def test_matte_ok(client):
    image = "data:image/png;base64," + base64.b64encode(PNG_1X1).decode("ascii")
    response = client.post("/matte", json={"image": image, "prompt": "the dog"})
    assert response.status_code == 200
    body = response.json()
    assert body["mask_format"] == "png"
    assert body["mask_shape"] == [1, 1]
    assert body["prompt"] == "the dog"
    assert body["threshold"] == pytest.approx(0.5)


def test_matte_default_prompt(client):
    image = base64.b64encode(PNG_1X1).decode("ascii")
    response = client.post("/matte", json={"image": image})
    assert response.status_code == 200
    assert response.json()["prompt"] == "the main foreground subject"


def test_matte_requires_image(client):
    response = client.post("/matte", json={})
    assert response.status_code == 422


def test_edit_ok(client):
    response = client.post(
        "/edit",
        files={"file": ("room.png", PNG_1X1, "image/png")},
        data={"prompt": "replace the sofa with a wooden bench", "max_size": "800"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == PNG_1X1


def test_edit_ok_with_seed(client):
    response = client.post(
        "/edit",
        files={"file": ("room.png", PNG_1X1, "image/png")},
        data={"prompt": "make it sunset", "seed": "42"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_edit_requires_prompt(client):
    response = client.post(
        "/edit",
        files={"file": ("room.png", PNG_1X1, "image/png")},
    )
    assert response.status_code in (422, 400)


def test_edit_rejects_blank_prompt(client):
    response = client.post(
        "/edit",
        files={"file": ("room.png", PNG_1X1, "image/png")},
        data={"prompt": "   "},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_auth_rejects_missing_key(client_auth):
    response = client_auth.post(
        "/reconstruct",
        files={"file": ("tiny.png", PNG_1X1, "image/png")},
    )
    assert response.status_code == 401
    body = response.json()
    assert body["code"] == "unauthorized"
    assert "X-Request-ID" in response.headers


def test_auth_rejects_invalid_key(client_auth):
    response = client_auth.post(
        "/segment",
        headers={"X-API-Key": "wrong-key"},
        json={"image": "data:image/png;base64," + base64.b64encode(PNG_1X1).decode("ascii")},
    )
    assert response.status_code == 401


def test_auth_accepts_valid_key(client_auth):
    response = client_auth.post(
        "/reconstruct",
        headers={"X-API-Key": "secret-2"},
        files={"file": ("tiny.png", PNG_1X1, "image/png")},
    )
    assert response.status_code == 200


def test_auth_health_stays_public(client_auth):
    response = client_auth.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
