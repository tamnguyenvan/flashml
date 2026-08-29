from __future__ import annotations

import base64
import binascii
import io
from typing import Any

from flashml.errors import InvalidImageError, PayloadTooLargeError


def decode_base64_image(value: object, *, max_bytes: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise InvalidImageError("image must be a non-empty base64 string or data URL")

    encoded = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("image is not valid base64") from exc

    if not image_bytes:
        raise InvalidImageError("image is empty")
    if len(image_bytes) > max_bytes:
        raise PayloadTooLargeError(max_bytes)
    return image_bytes


def decode_rgb_array(image_bytes: bytes):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("image could not be decoded; use a PNG or JPEG image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_from_payload(value: object, *, max_bytes: int):
    return decode_rgb_array(decode_base64_image(value, max_bytes=max_bytes))


def pil_rgb_from_payload(value: object, *, max_bytes: int):
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(decode_base64_image(value, max_bytes=max_bytes)))
        return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise InvalidImageError("image could not be decoded; use a PNG or JPEG image") from exc


def mask_png_base64(mask: Any) -> str:
    import cv2
    import numpy as np

    mask_image = np.where(mask, 255, 0).astype(np.uint8)
    encoded, buffer = cv2.imencode(".png", mask_image)
    if not encoded:
        raise RuntimeError("failed to encode segmentation mask")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def mask_png_data_url(mask: Any) -> str:
    from PIL import Image

    output = io.BytesIO()
    Image.fromarray((mask.astype("uint8") * 255)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
