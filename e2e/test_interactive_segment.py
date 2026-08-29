"""Client test for ``POST /interactive-segment`` (SimpleClick).

Allows user to interactively click / draw polylines on an image, sends those
points to the SimpleClick endpoint, and visualizes/saves the segmentation mask.

Usage:

    # Interactive drawing mode (OpenCV GUI):
    python e2e/test_interactive_segment.py --image /path/to/room.jpg

    # Non-interactive / CLI mode:
    python e2e/test_interactive_segment.py --image /path/to/room.jpg --positive-points "150,200 160,210" --no-gui
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Make the editable-repo import of `e2e.common` work when run directly as
# `python e2e/test_interactive_segment.py` from outside the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from e2e.common import (
    auth_headers,
    build_parser,
    decode_base64_image,
    has_display,
    png_base64,
    resolve_out_path,
)


def run_interactive_segment(
    client: httpx.Client,
    base_url: str,
    image_base64: str,
    positive_points: list[list[float]],
    negative_points: list[list[float]],
    threshold: float,
    api_key: str,
) -> dict:
    payload = {
        "image": image_base64,
        "positive_points": positive_points,
        "negative_points": negative_points,
        "threshold": threshold,
    }
    print(f"POST {base_url}/interactive-segment ({len(positive_points)} positive, {len(negative_points)} negative points)")
    response = client.post("/interactive-segment", json=payload, headers=auth_headers(api_key))
    response.raise_for_status()

    assert "X-Request-ID" in response.headers, "missing X-Request-ID on /interactive-segment"
    body = response.json()
    assert isinstance(body.get("mask"), str) and body["mask"], "missing base64 mask in response"
    return body


def parse_points(raw: str) -> list[list[float]]:
    """Parse a CLI points string like '50,50 120,80'."""
    points: list[list[float]] = []
    for token in raw.split():
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"invalid point {token!r}; expected 'x,y' pairs, e.g. '50,50 120,80'")
        points.append([float(parts[0]), float(parts[1])])
    return points


def collect_points_gui(image_bytes: bytes) -> tuple[list[list[float]], list[list[float]]]:
    """Open an OpenCV window for the user to click/draw positive and negative points/polylines."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Note: opencv-python is not installed. To use interactive GUI, run: pip install opencv-python", file=sys.stderr)
        return [], []

    np_arr = np.frombuffer(image_bytes, np.uint8)
    orig_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if orig_img is None:
        raise ValueError("Could not decode image with OpenCV")

    pos_points: list[list[float]] = []
    neg_points: list[list[float]] = []
    drawing = False
    is_positive = True

    window_name = "SimpleClick: Left-Drag/Click=Positive, Right-Click=Negative | [c]=Clear | [Enter]=Run | [q]=Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1200, orig_img.shape[1]), min(900, orig_img.shape[0]))

    def redraw() -> np.ndarray:
        display = orig_img.copy()
        # Draw positive points & polylines
        for i, pt in enumerate(pos_points):
            p = (int(pt[0]), int(pt[1]))
            cv2.circle(display, p, 4, (0, 255, 0), -1)
            cv2.circle(display, p, 6, (0, 100, 0), 1)
            if i > 0:
                prev = (int(pos_points[i - 1][0]), int(pos_points[i - 1][1]))
                # Only connect if points are close enough (part of same stroke)
                dist = (p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2
                if dist < 2500:
                    cv2.line(display, prev, p, (0, 255, 0), 2)

        # Draw negative points
        for pt in neg_points:
            p = (int(pt[0]), int(pt[1]))
            cv2.circle(display, p, 4, (0, 0, 255), -1)
            cv2.circle(display, p, 6, (0, 0, 100), 1)

        info_text = f"Positive: {len(pos_points)} pts | Negative: {len(neg_points)} pts | [Enter]=Segment [c]=Clear [q]=Quit"
        cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return display

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, is_positive
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            is_positive = True
            pos_points.append([float(x), float(y)])
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            if is_positive:
                pos_points.append([float(x), float(y)])
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            neg_points.append([float(x), float(y)])

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, redraw())
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # ENTER or SPACE
            if not pos_points:
                print("Please draw/click at least one positive point before submitting.")
                continue
            break
        elif key in (ord("c"), ord("C")):
            pos_points.clear()
            neg_points.clear()
        elif key in (27, ord("q"), ord("Q")):  # ESC or q
            pos_points.clear()
            neg_points.clear()
            break

    cv2.destroyWindow(window_name)
    return pos_points, neg_points


def visualize_and_save(
    image_bytes: bytes,
    mask_bytes: bytes,
    pos_points: list[list[float]],
    neg_points: list[list[float]],
    out_path: Path,
    show_gui: bool = True,
) -> None:
    """Create a colored mask overlay on the original image and save/display it."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        # Fallback without cv2: save raw mask PNG
        mask_out = out_path.parent / f"{out_path.stem}_mask.png"
        mask_out.write_bytes(mask_bytes)
        print(f"  saved mask  : {mask_out}")
        return

    orig_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    mask_img = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

    if orig_img is None or mask_img is None:
        return

    if mask_img.shape[:2] != orig_img.shape[:2]:
        mask_img = cv2.resize(mask_img, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create colored overlay (cyan/green highlight for mask)
    binary_mask = (mask_img > 127).astype(np.uint8)
    colored_mask = np.zeros_like(orig_img)
    colored_mask[binary_mask == 1] = [0, 230, 100]  # BGR

    overlay = orig_img.copy()
    overlay = cv2.addWeighted(colored_mask, 0.45, overlay, 0.55, 0)
    # Blend only where mask is present
    result = np.where(binary_mask[:, :, None] == 1, overlay, orig_img)

    # Draw mask contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 120), 2)

    # Draw points used
    for pt in pos_points:
        cv2.circle(result, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
    for pt in neg_points:
        cv2.circle(result, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

    # Save overlay image & raw mask
    cv2.imwrite(str(out_path), result)
    mask_path = out_path.parent / f"{out_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), mask_img)

    print(f"  saved result: {out_path}")
    print(f"  saved mask  : {mask_path}")

    if show_gui and has_display():
        window_name = "SimpleClick Result - Press any key to close"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1200, result.shape[1]), min(900, result.shape[0]))
        cv2.imshow(window_name, result)
        print("  Displaying result window. Press any key in the window to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


def main() -> int:
    parser = build_parser("Exercise /interactive-segment on a running FlashML gateway with GUI drawing.")
    parser.add_argument("--image", default=None, help="Path to an image file (default: generated PNG)")
    parser.add_argument(
        "--positive-points",
        default=None,
        help="Foreground clicks as 'x,y' pairs separated by spaces (e.g. '100,150 120,180')",
    )
    parser.add_argument(
        "--negative-points",
        default="",
        help="Optional background clicks as 'x,y' pairs separated by spaces",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.49,
        help="Probability cutoff in (0, 1) (default: 0.49)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to save the segmented result image (default: <image_stem>_interactive_segment.png)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable interactive GUI even if display is available",
    )
    args = parser.parse_args()

    if args.image:
        image_bytes = Path(args.image).read_bytes()
    else:
        from e2e.common import make_png
        image_bytes = make_png(512, 512)

    image_b64 = png_base64(image_bytes)

    # Determine points: either from CLI flags or from GUI
    positive_points: list[list[float]] = []
    negative_points: list[list[float]] = []

    if args.positive_points is not None:
        try:
            positive_points = parse_points(args.positive_points)
            negative_points = parse_points(args.negative_points) if args.negative_points else []
        except ValueError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
    elif not args.no_gui and has_display():
        print("Opening image window... Left-click/drag to draw foreground points. Press ENTER to submit.")
        positive_points, negative_points = collect_points_gui(image_bytes)
        if not positive_points:
            print("No points drawn. Exiting.")
            return 0
    else:
        # Fallback default point
        positive_points = [[50.0, 50.0]]
        print("Using default point [[50.0, 50.0]] (use GUI or --positive-points to specify).")

    out_path = resolve_out_path(
        args.image,
        "output_interactive_segment.png",
        "_interactive_segment.png",
        args.out,
    )

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            body = run_interactive_segment(
                client,
                args.base_url,
                image_b64,
                positive_points,
                negative_points,
                args.threshold,
                args.api_key,
            )
            mask_bytes = decode_base64_image(body["mask"])
            print(f"  mask shape  : {body.get('mask_shape')}")
            print(f"  threshold   : {body.get('threshold')}")

            visualize_and_save(
                image_bytes,
                mask_bytes,
                positive_points,
                negative_points,
                out_path,
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