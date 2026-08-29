"""Client test for ``POST /interactive-segment`` (SimpleClick).

Sends an image plus foreground clicks and validates the returned mask payload
(``mask``, ``mask_format``, ``mask_shape``, echo of points/threshold).

Run:

    python e2e/test_interactive_segment.py [--base-url URL] [--image PATH] [--timeout SEC]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the editable-repo import of `e2e.common` work when run directly as
# `python e2e/test_interactive_segment.py` from outside the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from e2e.common import build_parser, png_base64


def run(
    client: httpx.Client,
    base_url: str,
    image_base64: str,
    positive_points: list[list[float]],
    negative_points: list[list[float]],
    threshold: float,
) -> dict:
    payload = {
        "image": image_base64,
        "positive_points": positive_points,
        "negative_points": negative_points,
        "threshold": threshold,
    }
    print(f"POST {base_url}/interactive-segment")
    response = client.post("/interactive-segment", json=payload)
    response.raise_for_status()

    assert "X-Request-ID" in response.headers, "missing X-Request-ID on /interactive-segment"
    body = response.json()

    assert isinstance(body.get("mask"), str) and body["mask"], "missing base64 mask"
    assert body.get("mask_format") == "png", f"unexpected mask_format: {body.get('mask_format')!r}"
    assert len(body.get("mask_shape", [])) == 2, "mask_shape must be [height, width]"
    assert body.get("positive_points_used") == positive_points, "positive_points not echoed"
    assert body.get("negative_points_used") == negative_points, "negative_points not echoed"
    return body


def parse_points(raw: str) -> list[list[float]]:
    """Parse a CLI points string like '50,50 120,80' (whitespace-separated pairs)."""
    points: list[list[float]] = []
    for token in raw.split():
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"invalid point {token!r}; expected 'x,y' pairs separated by spaces, e.g. '50,50 120,80'"
            )
        points.append([float(parts[0]), float(parts[1])])
    if not points:
        raise ValueError("no points provided")
    return points


def main() -> int:
    parser = build_parser("Exercise /interactive-segment on a running FlashML gateway.")
    parser.add_argument("--image", default=None, help="Path to an image file (default: generated PNG)")
    parser.add_argument(
        "--positive-points",
        default="50,50",
        help="Foreground clicks as 'x,y' pairs separated by spaces (default: 50,50)",
    )
    parser.add_argument(
        "--negative-points",
        default="",
        help="Optional background clicks as 'x,y' pairs separated by spaces (default: none)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.49,
        help="Probability cutoff in (0, 1) (default: 0.49)",
    )
    args = parser.parse_args()

    try:
        positive_points = parse_points(args.positive_points)
        negative_points = parse_points(args.negative_points) if args.negative_points else []
    except ValueError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    if args.image:
        image_base64 = png_base64(Path(args.image).read_bytes())
    else:
        image_base64 = png_base64()

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            body = run(
                client,
                args.base_url,
                image_base64,
                positive_points,
                negative_points,
                args.threshold,
            )
            print(f"  positive    : {body['positive_points_used']}")
            print(f"  negative    : {body['negative_points_used']}")
            print(f"  mask_format : {body['mask_format']}")
            print(f"  mask_shape  : {body['mask_shape']}")
            print(f"  threshold   : {body['threshold']}")
            print(f"  mask        : base64, {len(body['mask'])} chars")
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