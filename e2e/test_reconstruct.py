"""Client test for ``POST /reconstruct`` (and its alias ``/predict``).

Uploads a multipart image and inspects the returned ZIP archive, validating the
expected members (``point_map.npy`` and ``metadata.json``).

Run:

    python e2e/test_reconstruct.py [--base-url URL] [--image PATH] [--timeout SEC]
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

# Make the editable-repo import of `e2e.common` work when run directly as
# `python e2e/test_reconstruct.py` from outside the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from e2e.common import build_parser, make_png

EXPECTED_ARCHIVE_MEMBERS = {"point_map.npy", "metadata.json"}


def _load_image(path: str | None) -> tuple[bytes, str]:
    if path:
        image_path = Path(path)
        return image_path.read_bytes(), image_path.name
    return make_png(256, 256), "sample.png"


def run_reconstruct(client: httpx.Client, base_url: str, image: bytes, filename: str) -> bytes:
    print(f"POST {base_url}/reconstruct")
    response = client.post(
        "/reconstruct",
        files={"file": (filename, image, "image/png")},
        data={
            "include_mesh": "false",
            "include_debug": "false",
            "max_size": "800",
            "resolution_level": "9",
            "refine_steps": "3",
            "edge_threshold": "0.04",
        },
    )
    response.raise_for_status()

    assert "X-Request-ID" in response.headers, "missing X-Request-ID on /reconstruct"
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("application/zip"), (
        f"expected application/zip, got {content_type!r}"
    )
    return response.content


def inspect_archive(archive: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = set(zf.namelist())
        missing = EXPECTED_ARCHIVE_MEMBERS - names
        assert not missing, f"archive missing expected members: {sorted(missing)}"
        print("  archive     :")
        for name in sorted(names):
            info = zf.getinfo(name)
            print(f"    - {name} ({info.file_size} bytes)")
    print("  OK")


def main() -> int:
    parser = build_parser("Exercise /reconstruct on a running FlashML gateway.")
    parser.add_argument("--image", default=None, help="Path to an image file (default: generated PNG)")
    args = parser.parse_args()

    image, filename = _load_image(args.image)

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            archive = run_reconstruct(client, args.base_url, image, filename)
            inspect_archive(archive)
        except AssertionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        except httpx.HTTPError as exc:
            print(f"REQUEST ERROR: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())