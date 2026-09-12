"""Client test for ``POST /matte`` (MultiMatte promptable matting).

Sends an image plus a text prompt to the matte endpoint, saves the returned
binary mask PNG, and optionally displays the result.

Usage:

    python e2e/test_matte.py --image /path/to/photo.jpg --prompt "the dog" [--base-url URL] [--api-key KEY]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the editable-repo import of `e2e.common` work when run directly as
# `python e2e/test_matte.py` from outside the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from e2e.common import (
    auth_headers,
    build_parser,
    decode_base64_image,
    has_display,
    make_png,
    png_data_url,
    resolve_out_path,
)


def run_matte(
    client: httpx.Client,
    base_url: str,
    image_data_url: str,
    prompt: str | None,
    api_key: str,
) -> dict:
    print(f"POST {base_url}/matte")
    payload: dict = {"image": image_data_url}
    if prompt:
        payload["prompt"] = prompt
    response = client.post("/matte", json=payload, headers=auth_headers(api_key))
    response.raise_for_status()

    assert "X-Request-ID" in response.headers, "missing X-Request-ID on /matte"
    body = response.json()
    assert body.get("mask_format") == "png", f"unexpected mask_format: {body.get('mask_format')!r}"
    assert isinstance(body.get("mask"), str) and body["mask"], "missing mask in response"
    return body


def main() -> int:
    parser = build_parser("Exercise /matte on a running FlashML gateway and save the mask.")
    parser.add_argument("--image", default=None, help="Path to an image file (default: generated PNG)")
    parser.add_argument(
        "--prompt",
        default="the main foreground subject",
        help="Text concept to matte (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to save the mask image (default: <image_stem>_matte.png)",
    )
    parser.add_argument("--no-gui", action="store_true", help="Disable display window")
    args = parser.parse_args()

    if args.image:
        image_bytes = Path(args.image).read_bytes()
        image_data_url = png_data_url(image_bytes)
    else:
        image_bytes = make_png(512, 512)
        image_data_url = png_data_url(image_bytes)

    out_mask_path = resolve_out_path(
        args.image,
        "output_matte.png",
        "_matte.png",
        args.out,
    )

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            body = run_matte(client, args.base_url, image_data_url, args.prompt, args.api_key)
            mask_bytes = decode_base64_image(body["mask"])
            out_mask_path.write_bytes(mask_bytes)
            print(f"  saved mask  : {out_mask_path} ({len(mask_bytes)} bytes)")
            print(f"  prompt      : {body.get('prompt')}")
            print(f"  mask shape  : {body.get('mask_shape')}")
            print(f"  threshold   : {body.get('threshold')}")

            if not args.no_gui and has_display():
                try:
                    import cv2
                    import numpy as np
                except ImportError:
                    print("Note: Install opencv-python to display the mask.")
                    print("  OK")
                    return 0
                mask_img = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    window_name = "MultiMatte mask - Press any key to close"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name, mask_img)
                    print("  Displaying result window. Press any key in the window to continue...")
                    cv2.waitKey(0)
                    cv2.destroyWindow(window_name)
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
