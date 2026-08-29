"""Client test for ``POST /reconstruct`` (MoGe).

Uploads an image, calls ``POST /reconstruct``, and saves the returned ZIP archive directly.

Run:

    python e2e/test_reconstruct.py --image /path/to/room.jpg [--base-url URL] [--api-key KEY] [--out output.zip]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the editable-repo import of `e2e.common` work when run directly as
# `python e2e/test_reconstruct.py` from outside the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from e2e.common import auth_headers, build_parser, make_png, resolve_out_path


def _load_image(path: str | None) -> tuple[bytes, str]:
    if path:
        image_path = Path(path)
        return image_path.read_bytes(), image_path.name
    return make_png(256, 256), "sample.png"


def run_reconstruct(
    client: httpx.Client,
    base_url: str,
    image: bytes,
    filename: str,
    api_key: str,
    *,
    include_mesh: bool = True,
    include_debug: bool = True,
    max_size: int = 800,
    resolution_level: int = 9,
    refine_steps: int = 3,
    edge_threshold: float = 0.04,
) -> bytes:
    print(f"POST {base_url}/reconstruct")
    response = client.post(
        "/reconstruct",
        headers=auth_headers(api_key),
        files={"file": (filename, image, "image/png")},
        data={
            "include_mesh": str(include_mesh).lower(),
            "include_debug": str(include_debug).lower(),
            "max_size": str(max_size),
            "resolution_level": str(resolution_level),
            "refine_steps": str(refine_steps),
            "edge_threshold": str(edge_threshold),
        },
    )
    response.raise_for_status()

    assert "X-Request-ID" in response.headers, "missing X-Request-ID on /reconstruct"
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("application/zip"), (
        f"expected application/zip, got {content_type!r}"
    )
    return response.content


def main() -> int:
    parser = build_parser("Exercise /reconstruct on a running FlashML gateway and save output zip.")
    parser.add_argument("--image", default=None, help="Path to an image file (default: generated PNG)")
    parser.add_argument("--out", default=None, help="Path to save the output ZIP (default: <image_stem>_reconstruct.zip)")
    parser.add_argument("--include-mesh", action="store_true", default=True, help="Include output.glb in zip (default: True)")
    parser.add_argument("--no-mesh", dest="include_mesh", action="store_false", help="Do not include mesh")
    parser.add_argument("--include-debug", action="store_true", default=True, help="Include debug PNGs in zip (default: True)")
    parser.add_argument("--no-debug", dest="include_debug", action="store_false", help="Do not include debug PNGs")
    parser.add_argument("--max-size", type=int, default=800, help="Max image dimension (default: 800)")
    parser.add_argument("--resolution-level", type=int, default=9, help="Resolution level (default: 9)")
    parser.add_argument("--refine-steps", type=int, default=3, help="Refinement steps (default: 3)")
    parser.add_argument("--edge-threshold", type=float, default=0.04, help="Edge threshold (default: 0.04)")
    args = parser.parse_args()

    image, filename = _load_image(args.image)
    out_path = resolve_out_path(args.image, "output_reconstruct.zip", "_reconstruct.zip", args.out)

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            archive = run_reconstruct(
                client,
                args.base_url,
                image,
                filename,
                args.api_key,
                include_mesh=args.include_mesh,
                include_debug=args.include_debug,
                max_size=args.max_size,
                resolution_level=args.resolution_level,
                refine_steps=args.refine_steps,
                edge_threshold=args.edge_threshold,
            )
            out_path.write_bytes(archive)
            size_mb = len(archive) / (1024 * 1024)
            print(f"  status      : 200 OK")
            print(f"  archive     : {len(archive)} bytes ({size_mb:.2f} MB)")
            print(f"  saved to    : {out_path}")
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