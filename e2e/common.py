"""Shared helpers for the FlashML API client test scripts.

These scripts exercise a *running* FlashML server (the FastAPI gateway plus its
GPU workers), unlike the pytest suite under ``tests/`` which uses fake services.

Run from the repo root, for example::

    python e2e/test_health.py
    python e2e/test_health.py --base-url http://127.0.0.1:8000

Each script talks to ``FLASHML_BASE_URL`` (default ``http://localhost:8000``)
unless overridden with ``--base-url``.
"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import struct
import zlib

DEFAULT_BASE_URL = "http://localhost:8000"


def base_url() -> str:
    """Resolve the gateway URL from env or a sensible default."""
    return os.environ.get("FLASHML_BASE_URL", DEFAULT_BASE_URL)


def build_parser(description: str) -> argparse.ArgumentParser:
    """Standard CLI parser so every client script behaves the same way."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--base-url",
        default=base_url(),
        help=f"FlashML gateway URL (default: {base_url()})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("FLASHML_API_KEY", ""),
        help="X-API-Key header value (default: $FLASHML_API_KEY or empty)",
    )
    return parser


def auth_headers(api_key: str) -> dict[str, str]:
    """Headers carrying the API key, if one was provided."""
    if api_key:
        return {"X-API-Key": api_key}
    return {}


def make_png(width: int = 128, height: int = 128) -> bytes:
    """Build a valid RGB PNG of the given size using only the standard library.

    Produces a small gradient so it can be decoded by any model without pulling
    in Pillow or OpenCV.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        payload = struct.pack(">I", len(data)) + chunk_type + data
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return payload + struct.pack(">I", crc)

    def _v(value: int, size: int) -> int:
        # Map a 0-based index to a 0..255 byte value.
        return (value * 255) // max(size - 1, 1) if size > 1 else 0

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))

    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type 0 (None) for each scanline
        for x in range(width):
            rows.extend((_v(y, height), _v(x, width), 128))
    idat = _chunk(b"IDAT", zlib.compress(bytes(rows), 9))

    return signature + ihdr + idat + _chunk(b"IEND", b"")


def make_mask_png(width: int = 128, height: int = 128) -> bytes:
    """Build a grayscale binary mask PNG with a white region to remove.

    The mask is black except for a white box near the center, and is suitable
    for sending as the ``mask`` field of ``POST /remove``.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        payload = struct.pack(">I", len(data)) + chunk_type + data
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return payload + struct.pack(">I", crc)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))

    # White box spanning the middle 40% of the image.
    box_x0, box_x1 = round(width * 0.3), round(width * 0.7)
    box_y0, box_y1 = round(height * 0.3), round(height * 0.7)

    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type 0
        for x in range(width):
            rows.append(255 if box_x0 <= x < box_x1 and box_y0 <= y < box_y1 else 0)
    idat = _chunk(b"IDAT", zlib.compress(bytes(rows), 9))

    return signature + ihdr + idat + _chunk(b"IEND", b"")



def png_base64(png: bytes | None = None) -> str:
    """Return the PNG as raw base64 (no data URL prefix)."""
    return base64.b64encode(png or make_png()).decode("ascii")


def png_data_url(png: bytes | None = None) -> str:
    """Return the PNG as a ``data:image/png;base64,...`` URL."""
    return "data:image/png;base64," + png_base64(png)


def decode_base64_image(raw: str) -> bytes:
    """Decode raw base64 string or a data URL (e.g. data:image/png;base64,...)."""
    if "," in raw and raw.startswith("data:"):
        _, raw = raw.split(",", 1)
    return base64.b64decode(raw)


def resolve_out_path(
    input_path: str | None,
    default_name: str,
    suffix: str,
    out_arg: str | None = None,
) -> Path:
    """Determine output file path based on input path and custom suffix."""
    if out_arg:
        return Path(out_arg)
    if input_path:
        p = Path(input_path)
        return p.parent / f"{p.stem}{suffix}"
    return Path(default_name)


def has_display() -> bool:
    """Check if graphical display is available."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))