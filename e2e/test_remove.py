"""Client test for ``POST /remove`` (LaMa inpainting).

Uploads an image plus a binary mask and validates that the returned payload is
a PNG image with the ``image/png`` content type.

Run:

    python e2e/test_remove.py [--base-url URL] [--image PATH] [--mask PATH] [--timeout SEC]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the editable-repo import of `e2e.common` work when run directly as
# `python e2e/test_remove.py` from outside the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from e2e.common import auth_headers, build_parser, make_mask_png, make_png

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def run(
    client: httpx.Client,
    base_url: str,
    image_bytes: bytes,
    mask_bytes: bytes,
    image_filename: str,
    mask_filename: str,
    max_size: int,
    api_key: str,
) -> bytes:
    print(f"POST {base_url}/remove")
    response = client.post(
        "/remove",
        headers=auth_headers(api_key),
        files={
            "file": (image_filename, image_bytes, "image/png"),
            "mask": (mask_filename, mask_bytes, "image/png"),
        },
        data={"max_size": str(max_size)},
    )
    response.raise_for_status()

    assert "X-Request-ID" in response.headers, "missing X-Request-ID on /remove"
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("image/png"), (
        f"expected image/png, got {content_type!r}"
    )
    assert response.content.startswith(PNG_SIGNATURE), "response is not a PNG"
    return response.content


def main() -> int:
    parser = build_parser("Exercise /remove on a running FlashML gateway.")
    parser.add_argument("--image", default=None, help="Path to the RGB image (default: generated)")
    parser.add_argument("--mask", default=None, help="Path to the binary mask PNG (default: generated)")
    parser.add_argument("--max-size", type=int, default=1024, help="Longest-side limit (default: 1024)")
    parser.add_argument("--out", default=None, help="Write the returned inpainted PNG to this path")
    args = parser.parse_args()

    if args.image:
        image_bytes = Path(args.image).read_bytes()
        image_filename = Path(args.image).name
    else:
        image_bytes = make_png(256, 256)
        image_filename = "sample.png"

    if args.mask:
        mask_bytes = Path(args.mask).read_bytes()
        mask_filename = Path(args.mask).name
    else:
        mask_bytes = make_mask_png(256, 256)
        mask_filename = "mask.png"

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            result = run(
                client,
                args.base_url,
                image_bytes,
                mask_bytes,
                image_filename,
                mask_filename,
                args.max_size,
                args.api_key,
            )
            print(f"  output      : image/png, {len(result)} bytes")
            if args.out:
                Path(args.out).write_bytes(result)
                print(f"  saved       : {args.out}")
            print("  OK")
        except AssertionError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        except httpx.HTTPError as exc:
            print(f"REQUEST ERROR: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
