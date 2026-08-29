import base64

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


def test_remove_ok(client):
    response = client.post(
        "/remove",
        files={
            "file": ("room.png", PNG_1X1, "image/png"),
            "mask": ("mask.png", PNG_1X1, "image/png"),
        },
        data={"max_size": "800"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == PNG_1X1


def test_remove_requires_mask(client):
    response = client.post(
        "/remove",
        files={"file": ("room.png", PNG_1X1, "image/png")},
    )
    assert response.status_code in (422, 400)
