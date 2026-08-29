"""Client test for ``POST /segment`` (OneFormer ADE20K).

Sends an image as a data URL and validates the returned semantic-mask payload
(``provider`` is ``oneformer`` and the ``masks`` dict is present).

Run:

    python e2e/test_segment.py [--base-url URL] [--image PATH] [--timeout SEC]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the editable-repo import of `e2e.common` work when run directly as
# `python e2e/test_segment.py` from outside the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from e2e.common import build_parser, png_data_url


def run(client: httpx.Client, base_url: str, image_data_url: str) -> dict:
    print(f"POST {base_url}/segment")
    response = client.post("/segment", json={"image": image_data_url})
    response.raise_for_status()

    assert "X-Request-ID" in response.headers, "missing X-Request-ID on /segment"
    body = response.json()

    assert body.get("provider") == "oneformer", f"unexpected provider: {body.get('provider')!r}"
    assert isinstance(body.get("masks"), dict), "missing masks dict"
    assert isinstance(body.get("label_ids"), dict), "missing label_ids dict"
    assert len(body.get("image_size_hw", [])) == 2, "image_size_hw must be [height, width]"
    return body


def main() -> int:
    parser = build_parser("Exercise /segment on a running FlashML gateway.")
    parser.add_argument("--image", default=None, help="Path to an image file (default: generated PNG)")
    args = parser.parse_args()

    if args.image:
        image_data_url = png_data_url(Path(args.image).read_bytes())
    else:
        image_data_url = png_data_url()

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            body = run(client, args.base_url, image_data_url)
            print(f"  model       : {body.get('model')}")
            print(f"  provider    : {body.get('provider')}")
            print(f"  image size  : {body.get('image_size_hw')}")
            print(f"  label ids   : {body.get('label_ids')}")
            print(f"  masks       : {sorted(body['masks'])}")
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