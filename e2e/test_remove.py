"""Client test for ``POST /remove`` (FLUX.2 klein object removal).

Pipeline: Show original -> user draws mask -> Enter -> call API -> show result

Usage:

    # Interactive brush drawing mode (OpenCV GUI):
    python e2e/test_remove.py --image /path/to/room.jpg

    # Non-interactive / CLI mode with existing mask:
    python e2e/test_remove.py --image /path/to/room.jpg --mask /path/to/mask.png --no-gui
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

# Make the editable-repo import of `e2e.common` work when run directly as
# `python e2e/test_remove.py` from outside the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from e2e.common import (
    auth_headers,
    build_parser,
    has_display,
    make_mask_png,
    make_png,
    resolve_out_path,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def run_remove(
    client: httpx.Client,
    base_url: str,
    image_bytes: bytes,
    mask_bytes: bytes,
    image_filename: str,
    mask_filename: str,
    max_size: int,
    api_key: str,
) -> bytes:
    print(f"POST {base_url}/remove (image: {len(image_bytes)} bytes, mask: {len(mask_bytes)} bytes)")
    start = time.time()
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
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.2f}s")

    assert "X-Request-ID" in response.headers, "missing X-Request-ID on /remove"
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("image/png"), f"expected image/png, got {content_type!r}"
    assert response.content.startswith(PNG_SIGNATURE), "response is not a valid PNG"
    return response.content


def draw_mask_gui(image_bytes: bytes, initial_brush_radius: int = 15) -> bytes:
    """Open an interactive OpenCV brush drawing window to create a binary inpainting mask."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Note: opencv-python is not installed. Run: pip install opencv-python", file=sys.stderr)
        return b""

    orig_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if orig_img is None:
        raise ValueError("Could not decode image with OpenCV")

    h, w = orig_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    drawing = False
    last_point = None
    brush_radius = initial_brush_radius

    window_name = "FLUX.2 klein: Draw mask (L-drag) | +/- size | c=clear | Enter=run | q=quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1200, w), min(900, h))

    def make_overlay() -> np.ndarray:
        display = orig_img.copy()
        mask_bool = mask > 0
        display[mask_bool] = (
            display[mask_bool] * 0.35 + np.array([0, 0, 255], dtype=np.float32) * 0.65
        ).astype(np.uint8)

        # Brush preview circle at cursor would need mouse pos - skip for simplicity
        info = f"Brush: {brush_radius}px  Mask: {mask_bool.sum()}px  [+/-] size  [c]lear  [Enter] Inpaint  [q]uit"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return display

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, last_point, brush_radius
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(mask, pt, brush_radius, 255, -1)
            last_point = pt
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            if last_point is not None:
                cv2.line(mask, last_point, pt, 255, thickness=brush_radius * 2)
            cv2.circle(mask, pt, brush_radius, 255, -1)
            last_point = pt
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            last_point = None

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, make_overlay())
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # ENTER or SPACE
            if np.count_nonzero(mask) == 0:
                print("Please draw on the image to select the region to remove.")
                continue
            break
        elif key in (ord("+"), ord("="), ord("]")):
            brush_radius = min(100, brush_radius + 3)
        elif key in (ord("-"), ord("_"), ord("[")):
            brush_radius = max(2, brush_radius - 3)
        elif key in (ord("c"), ord("C")):
            mask.fill(0)
        elif key in (27, ord("q"), ord("Q")):  # ESC or q
            mask.fill(0)
            break

    cv2.destroyWindow(window_name)
    if np.count_nonzero(mask) == 0:
        return b""

    ok, buf = cv2.imencode(".png", mask)
    return buf.tobytes() if ok else b""


def show_result_gui(image_bytes: bytes, mask_bytes: bytes, inpainted_bytes: bytes) -> None:
    """Display side-by-side: Original | Mask Overlay | Inpainted Result"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return

    orig_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    mask_img = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    result_img = cv2.imdecode(np.frombuffer(inpainted_bytes, np.uint8), cv2.IMREAD_COLOR)

    if orig_img is None or result_img is None:
        return

    h, w = orig_img.shape[:2]
    if result_img.shape[:2] != (h, w):
        result_img = cv2.resize(result_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # 1. Original
    panel1 = orig_img.copy()
    cv2.putText(panel1, "Original", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel1, "Original", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. Mask overlay on original
    panel2 = orig_img.copy()
    if mask_img is not None:
        if mask_img.shape[:2] != (h, w):
            mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_img > 127
        panel2[mask_bool] = (
            panel2[mask_bool] * 0.35 + np.array([0, 0, 255], dtype=np.float32) * 0.65
        ).astype(np.uint8)
    cv2.putText(panel2, "Mask", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel2, "Mask", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Inpainted result
    panel3 = result_img.copy()
    cv2.putText(panel3, "Inpainted", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel3, "Inpainted", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    combined = np.hstack([panel1, panel2, panel3])
    window_name = "Result: Original | Mask | Inpainted  (any key to close)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1800, combined.shape[1]), min(700, combined.shape[0]))
    cv2.imshow(window_name, combined)
    print("  Displaying result. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> int:
    parser = build_parser("Exercise /remove on a running FlashML gateway with interactive mask drawing.")
    parser.add_argument("--image", default=None, help="Path to the RGB image (default: generated PNG)")
    parser.add_argument("--mask", default=None, help="Path to an existing binary mask PNG (optional)")
    parser.add_argument("--max-size", type=int, default=1024, help="Longest-side limit (default: 1024)")
    parser.add_argument("--out", default=None, help="Path to save the inpainted result PNG (default: <image_stem>_inpainted.png)")
    parser.add_argument("--no-gui", action="store_true", help="Disable interactive GUI even if display is available")
    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image)
        image_bytes = image_path.read_bytes()
        image_filename = image_path.name
    else:
        image_bytes = make_png(512, 512)
        image_filename = "sample.png"

    mask_bytes = b""
    mask_filename = "mask.png"

    if args.mask:
        mask_path = Path(args.mask)
        mask_bytes = mask_path.read_bytes()
        mask_filename = mask_path.name
    elif not args.no_gui and has_display():
        print("Opening brush window... Left-click/drag to paint the region to remove. Press ENTER to run inpainting.")
        mask_bytes = draw_mask_gui(image_bytes)
        if not mask_bytes:
            print("No mask drawn. Exiting.")
            return 0
        # Save the drawn mask immediately so it can be inspected before inpainting.
        out_path_early = resolve_out_path(args.image, "output.png", "_output.png", args.out)
        drawn_mask_path = out_path_early.parent / f"{out_path_early.stem}_mask.png"
        drawn_mask_path.write_bytes(mask_bytes)
        print(f"  saved drawn mask: {drawn_mask_path}")
    else:
        print("No mask specified and GUI disabled. Using default center box mask.")
        mask_bytes = make_mask_png(512, 512)

    out_path = resolve_out_path(args.image, "output_inpainted.png", "_inpainted.png", args.out)

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            result_bytes = run_remove(
                client,
                args.base_url,
                image_bytes,
                mask_bytes,
                image_filename,
                mask_filename,
                args.max_size,
                args.api_key,
            )
            out_path.write_bytes(result_bytes)
            print(f"  saved result: {out_path} ({len(result_bytes)} bytes)")

            # Save the drawn mask alongside the result for convenience
            drawn_mask_path = out_path.parent / f"{out_path.stem}_mask.png"
            drawn_mask_path.write_bytes(mask_bytes)
            print(f"  saved mask  : {drawn_mask_path}")

            if not args.no_gui and has_display():
                show_result_gui(image_bytes, mask_bytes, result_bytes)

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