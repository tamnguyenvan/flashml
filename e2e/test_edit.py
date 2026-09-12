"""Client test for ``POST /edit`` (FLUX.2 klein free-prompt editing).

Sends a source image plus a text prompt to the edit endpoint, saves the
returned PNG, and optionally displays the result.

Usage:

    python e2e/test_edit.py --image /path/to/room.jpg --prompt "replace the sofa with a wooden bench"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the editable-repo import of `e2e.common` work when run directly as
# `python e2e/test_edit.py` from outside the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from e2e.common import (
    auth_headers,
    build_parser,
    has_display,
    make_png,
    resolve_out_path,
)


def run_edit(
    client: httpx.Client,
    base_url: str,
    image_bytes: bytes,
    prompt: str,
    api_key: str,
    seed: int | None,
) -> bytes:
    print(f"POST {base_url}/edit")
    data: dict[str, str] = {"prompt": prompt}
    if seed is not None:
        data["seed"] = str(seed)
    response = client.post(
        "/edit",
        files={"file": ("image.png", image_bytes, "image/png")},
        data=data,
        headers=auth_headers(api_key),
    )
    response.raise_for_status()

    assert "X-Request-ID" in response.headers, "missing X-Request-ID on /edit"
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("image/png"), f"unexpected content-type: {content_type!r}"
    return response.content


def main() -> int:
    parser = build_parser("Exercise /edit on a running FlashML gateway and save the result.")
    parser.add_argument("--image", default=None, help="Path to a source image file (default: generated PNG)")
    parser.add_argument(
        "--prompt",
        default="replace the sofa with a wooden bench",
        help="Edit instruction (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility")
    parser.add_argument(
        "--out",
        default=None,
        help="Path to save the edited image (default: <image_stem>_edit.png)",
    )
    parser.add_argument("--no-gui", action="store_true", help="Disable display window")
    args = parser.parse_args()

    image_bytes = Path(args.image).read_bytes() if args.image else make_png(512, 512)
    out_path = resolve_out_path(args.image, "output_edit.png", "_edit.png", args.out)

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            content = run_edit(client, args.base_url, image_bytes, args.prompt, args.api_key, args.seed)
            out_path.write_bytes(content)
            print(f"  saved edit  : {out_path} ({len(content)} bytes)")
            print(f"  prompt      : {args.prompt}")

            if not args.no_gui and has_display():
                try:
                    import cv2
                    import numpy as np
                except ImportError:
                    print("Note: Install opencv-python to display the result.")
                    print("  OK")
                    return 0
                img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    window_name = "FLUX edit - Press any key to close"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name, img)
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
