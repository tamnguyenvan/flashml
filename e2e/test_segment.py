"""Client test for ``POST /segment`` (OneFormer ADE20K).

Sends an image to the OneFormer segmentation endpoint, saves all category mask PNGs
plus a colored composite segmentation overlay, and optionally displays the result.

Usage:

    python e2e/test_segment.py --image /path/to/room.jpg [--base-url URL] [--api-key KEY] [--out overlay.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the editable-repo import of `e2e.common` work when run directly as
# `python e2e/test_segment.py` from outside the repo root.
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

# Distinct color palette for segment classes (BGR format)
PALETTE = [
    (220, 100, 30),   # Blue / Cyan
    (40, 200, 80),    # Green
    (200, 50, 200),   # Magenta
    (30, 200, 240),   # Yellow
    (50, 100, 240),   # Orange
    (180, 180, 50),   # Teal
    (100, 50, 220),   # Purple
    (50, 220, 180),   # Lime
]


def run_segment(client: httpx.Client, base_url: str, image_data_url: str, api_key: str) -> dict:
    print(f"POST {base_url}/segment")
    response = client.post("/segment", json={"image": image_data_url}, headers=auth_headers(api_key))
    response.raise_for_status()

    assert "X-Request-ID" in response.headers, "missing X-Request-ID on /segment"
    body = response.json()
    assert body.get("provider") == "oneformer", f"unexpected provider: {body.get('provider')!r}"
    assert isinstance(body.get("masks"), dict), "missing masks dict in response"
    return body


def save_and_visualize_masks(
    image_bytes: bytes,
    masks_dict: dict[str, str],
    label_ids: dict[str, int],
    out_overlay_path: Path,
    show_gui: bool = True,
) -> None:
    """Save each category mask PNG and generate a colorized composite overlay."""
    out_dir = out_overlay_path.parent
    stem = out_overlay_path.stem.replace("_segment_overlay", "").replace("_segment", "")

    # 1. Save individual category mask files
    saved_masks: list[Path] = []
    decoded_masks: dict[str, bytes] = {}
    for label, mask_items in masks_dict.items():
        if not mask_items:
            print(f"  no mask     : {label}")
            continue

        # The API returns a list of mask objects per category.
        # For wall/floor/rug we expect at most one mask.
        mask_item = mask_items[0]
        mask_b64 = mask_item["mask"]

        mask_bytes = decode_base64_image(mask_b64)
        decoded_masks[label] = mask_bytes

        mask_file = out_dir / f"{stem}_segment_{label}.png"
        mask_file.write_bytes(mask_bytes)
        saved_masks.append(mask_file)

        print(f"  saved mask  : {mask_file.name} ({len(mask_bytes)} bytes)")

    # 2. Build composite color overlay if cv2 is available
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Note: Install opencv-python to generate color overlay image (pip install opencv-python).")
        return

    orig_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if orig_img is None:
        return

    h, w = orig_img.shape[:2]
    composite_color = np.zeros((h, w, 3), dtype=np.uint8)
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    legend_items: list[tuple[str, tuple[int, int, int]]] = []

    for i, (label, mask_bytes) in enumerate(decoded_masks.items()):
        color = PALETTE[i % len(PALETTE)]
        mask_img = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        if mask_img.shape[:2] != (h, w):
            mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

        binary = mask_img > 127
        composite_color[binary] = color
        combined_mask[binary] = 255
        legend_items.append((label, color))

    # Blend colored masks on original image
    has_segments = combined_mask > 0
    overlay = orig_img.copy()
    overlay[has_segments] = (
        orig_img[has_segments] * 0.45 + composite_color[has_segments] * 0.55
    ).astype(np.uint8)

    # Draw legend at top-left
    legend_y = 30
    for label, color in legend_items:
        cv2.rectangle(overlay, (15, legend_y - 15), (35, legend_y + 5), color, -1)
        cv2.rectangle(overlay, (15, legend_y - 15), (35, legend_y + 5), (0, 0, 0), 1)
        text = f"{label} (ID {label_ids.get(label, '?')})"
        cv2.putText(overlay, text, (45, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (45, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 30

    cv2.imwrite(str(out_overlay_path), overlay)
    print(f"  saved overlay: {out_overlay_path}")

    if show_gui and has_display():
        window_name = "OneFormer Segmentation - Press any key to close"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1200, w), min(900, h))
        cv2.imshow(window_name, overlay)
        print("  Displaying result window. Press any key in the window to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


def main() -> int:
    parser = build_parser("Exercise /segment on a running FlashML gateway and save all category masks.")
    parser.add_argument("--image", default=None, help="Path to an image file (default: generated PNG)")
    parser.add_argument(
        "--out",
        default=None,
        help="Path to save the color overlay image (default: <image_stem>_segment_overlay.png)",
    )
    parser.add_argument("--no-gui", action="store_true", help="Disable display window")
    args = parser.parse_args()

    if args.image:
        image_bytes = Path(args.image).read_bytes()
        image_data_url = png_data_url(image_bytes)
    else:
        image_bytes = make_png(512, 512)
        image_data_url = png_data_url(image_bytes)

    out_overlay_path = resolve_out_path(
        args.image,
        "output_segment_overlay.png",
        "_segment_overlay.png",
        args.out,
    )

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            body = run_segment(client, args.base_url, image_data_url, args.api_key)
            masks = body.get("masks", {})
            label_ids = body.get("label_ids", {})

            print(f"  model       : {body.get('model')}")
            print(f"  image size  : {body.get('image_size_hw')}")
            print(f"  categories  : {list(masks.keys())}")

            save_and_visualize_masks(
                image_bytes,
                masks,
                label_ids,
                out_overlay_path,
                show_gui=not args.no_gui,
            )
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