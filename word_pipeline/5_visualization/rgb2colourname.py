#!/usr/bin/env python3
"""
Create Prediction_full_with_colourname.json from Prediction_full.json.

The output is identical to the input except:
    "color_name": "colorname"
is replaced by:
    "color_name": "<coarse name>"

Colour naming reuses the functions of

    2_Colour/4_Test_colour_with_annotation/1_colour_patch.py

directly, so every step matches that file:

    1. image loaded with PIL and converted to RGB
    2. sauvola_binarization(img)
    3. extract_fg_bg_pixels_local(img_np, binary_img, polygon, LOCAL_BG_PADDING)
    4. "unknown" when fg_pixels is empty, or fewer than 10 fg/bg pixels
       (same guards as process_image() / get_text_color_via_gmm())
    5. bg_median = per-channel median of bg_pixels
    6. human_colour_name_contextual(text_rgb, bg_median)

The names come from the rule-based pre-grouping in 1_colour_patch.py. Those
rules were refined during rule-assisted human annotation of the test pages,
and are applied here unchanged. On pages outside the test set nobody checked
the names, so they are provisional coarse labels for browsing and
visualisation, not evaluated results. The predicted RGB (color_code) is
never modified.

It also writes output_visualized_predict_Colour/: each predicted polygon is
drawn in its predicted RGB and labelled with the generated colour name and
the predicted colour code.

"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# =============================================================================
# PATHS / CONFIG
# =============================================================================

HERE = Path(__file__).resolve().parent

COLOUR_PATCH_PATH = (
    HERE.parent
    / "2_Colour"
    / "4_Test_colour_with_annotation"
    / "1_colour_patch.py"
)

DEFAULT_INPUT = HERE / "Prediction_full.json"
DEFAULT_OUTPUT = HERE / "Prediction_full_with_colourname.json"
DEFAULT_IMG_DIR = HERE / "img"
DEFAULT_VIS_DIR = HERE / "output_visualized_predict_Colour"

PLACEHOLDER = "colorname"

# Same style as 1_visualize_adaptcolour_predict_v1.py.
TEXT_BG_ALPHA = 0.70
POLYGON_THICKNESS_BASE = 4

# Same minimum pixel counts as get_text_color_via_gmm() in 1_colour_patch.py.
MIN_FG_PIXELS = 10
MIN_BG_PIXELS = 10


# =============================================================================
# LOAD 1_colour_patch.py
# =============================================================================

@contextlib.contextmanager
def _working_dir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_colour_patch(path: Path):
    """
    Import 1_colour_patch.py as a module.

    Its top level calls os.makedirs() on relative output folders, so it is
    executed inside a throwaway directory to avoid creating empty folders
    in the caller's working directory.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Colour mapping source not found:\n  {path}")

    spec = importlib.util.spec_from_file_location("colour_patch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from:\n  {path}")

    module = importlib.util.module_from_spec(spec)
    with tempfile.TemporaryDirectory() as tmp, _working_dir(tmp):
        spec.loader.exec_module(module)

    required = (
        "sauvola_binarization",
        "extract_fg_bg_pixels_local",
        "human_colour_name_contextual",
        "LOCAL_BG_PADDING",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"{path} is missing: {', '.join(missing)}"
        )
    return module


# =============================================================================
# HELPERS
# =============================================================================

def parse_color_code(color_code):
    """Parse stored predicted RGB 'R-G-B' into (R, G, B)."""
    if isinstance(color_code, (list, tuple)) and len(color_code) >= 3:
        try:
            rgb = tuple(int(round(float(x))) for x in color_code[:3])
        except (TypeError, ValueError):
            return None
    elif isinstance(color_code, str) and color_code.strip():
        parts = color_code.strip().split("-")
        if len(parts) < 3:
            return None
        try:
            rgb = tuple(int(round(float(x))) for x in parts[:3])
        except ValueError:
            return None
    else:
        return None

    if any(x < 0 or x > 255 for x in rgb):
        return None
    return rgb


def resolve_image_path(img_dir: Path, file_name: str):
    """Find the source image corresponding to COCO file_name."""
    if not img_dir.is_dir() or not file_name:
        return None

    direct = img_dir / file_name
    if direct.is_file():
        return direct

    stem = Path(file_name).stem
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def load_page(cp, path: Path):
    """
    Load a page exactly as process_image() in 1_colour_patch.py does.

    Returns (img_np, binary_img).
    """
    img = Image.open(path).convert("RGB")
    binary_img = cp.sauvola_binarization(img)
    img_np = np.array(img)
    return img_np, binary_img


def name_for_annotation(cp, ann, page):
    """
    Coarse colour name for one annotation, following process_image() and
    get_text_color_via_gmm() in 1_colour_patch.py, but using the stored
    predicted color_code as the text RGB instead of recomputing it. The local
    background median is passed only because contextual_dark_ink_is_black()
    uses it.

    Returns (colour_name, reason) where reason is used for statistics.
    """
    rgb = parse_color_code(ann.get("color_code"))
    if rgb is None:
        return "unknown", "missing_rgb"

    if page is None:
        return cp.human_colour_name_contextual(rgb, None), "no_image"

    polygon = ann.get("segmentation", [[]])[0]
    if len(polygon) < 6:
        # process_image() skips these annotations entirely.
        return cp.human_colour_name_contextual(rgb, None), "invalid_polygon"

    img_np, binary_img = page
    fg_pixels, bg_pixels, _ = cp.extract_fg_bg_pixels_local(
        img_np=img_np,
        binary_img=binary_img,
        polygon=polygon,
        padding=cp.LOCAL_BG_PADDING,
    )

    if len(fg_pixels) == 0:
        return "unknown", "no_fg"

    if len(fg_pixels) < MIN_FG_PIXELS or len(bg_pixels) < MIN_BG_PIXELS:
        return "unknown", "too_few_pixels"

    bg_median = np.median(bg_pixels, axis=0).astype(int)
    return cp.human_colour_name_contextual(rgb, bg_median), "ok"


# =============================================================================
# VISUALIZATION
# =============================================================================

def safe_poly(seg):
    """Convert a flat segmentation list into an Nx2 polygon array."""
    if seg is None or len(seg) < 6:
        return None
    try:
        return np.array(seg, dtype=np.int32).reshape((-1, 2))
    except Exception:
        return None


def draw_text_with_transparent_black_bg(img, lines, org, font, scales,
                                        thickness, alpha=TEXT_BG_ALPHA):
    """Draw white text lines on one semi-transparent black background."""
    if isinstance(lines, str):
        lines = [lines]
    if isinstance(scales, (int, float)):
        scales = [scales] * len(lines)
    lines = [str(line) for line in lines if str(line)]
    if not lines:
        return

    sizes = []
    for line, scale in zip(lines, scales):
        (tw, th), baseline = cv2.getTextSize(line, font, scale, thickness)
        sizes.append((tw, th, baseline))

    gap = max(2, int(round(max(scales) * 4)))
    block_w = max(tw for tw, _, _ in sizes)
    block_h = sum(th + baseline for _, th, baseline in sizes) + gap * (len(sizes) - 1)

    x, y = org
    H, W = img.shape[:2]
    pad = max(2, int(max(scales) * 4))

    x = max(pad, min(x, W - block_w - pad - 1))
    y = max(sizes[0][1] + pad, min(y, H - block_h - pad - 1))

    x1 = max(0, x - pad)
    y1 = max(0, y - sizes[0][1] - pad)
    x2 = min(W - 1, x + block_w + pad)
    y2 = min(H - 1, y1 + block_h + 2 * pad)
    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2].copy()
    img[y1:y2, x1:x2] = cv2.addWeighted(
        np.zeros_like(roi), alpha, roi, 1 - alpha, 0
    )

    cursor_y = y
    for line, scale, (_, th, baseline) in zip(lines, scales, sizes):
        cv2.putText(img, line, (x, cursor_y), font, scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
        cursor_y += th + baseline + gap


def visualize_colour_names(data, img_dir: Path, vis_dir: Path):
    """
    Draw every predicted polygon in its predicted RGB, labelled with
    color_name and color_code (no orientation / writing tool).

    Returns (saved, missing_images).
    """
    vis_dir.mkdir(parents=True, exist_ok=True)

    anns_by_image = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann.get("image_id"), []).append(ann)

    font = cv2.FONT_HERSHEY_SIMPLEX
    saved = 0
    missing = 0

    for image in tqdm(data.get("images", []), desc="Visualize colour"):
        file_name = str(image.get("file_name", ""))
        path = resolve_image_path(img_dir, file_name)
        canvas = cv2.imread(str(path)) if path is not None else None
        if canvas is None:
            missing += 1
            continue

        H, W = canvas.shape[:2]
        font_scale = max(0.28, min(0.62, max(H, W) / 4200.0))
        code_scale = max(0.24, font_scale * 0.85)
        text_thickness = 1
        polygon_thickness = max(POLYGON_THICKNESS_BASE,
                                int(round(max(H, W) / 900.0)))
        text_offset = max(8, int(font_scale * 28))

        labels_to_draw = []
        for ann in anns_by_image.get(image.get("id"), []):
            raw_code = ann.get("color_code")
            rgb = parse_color_code(raw_code)
            bgr = (rgb[2], rgb[1], rgb[0]) if rgb is not None else (0, 255, 0)
            colour_name = str(ann.get("color_name", "unknown"))
            if isinstance(raw_code, str) and raw_code.strip():
                colour_code = raw_code.strip()
            elif rgb is not None:
                colour_code = "-".join(str(channel) for channel in rgb)
            else:
                colour_code = ""
            lines = [colour_name, colour_code] if colour_code else [colour_name]

            for seg in ann.get("segmentation", []):
                pts = safe_poly(seg)
                if pts is None:
                    continue
                cv2.polylines(canvas, [pts], isClosed=True, color=bgr,
                              thickness=polygon_thickness)

                top_point = pts[np.lexsort((pts[:, 0], pts[:, 1]))[0]]
                tx, ty = int(top_point[0]), int(top_point[1])
                labels_to_draw.append(
                    (lines, tx, max(text_offset, ty - text_offset))
                )

        # Text after all polygons so labels stay on top.
        for lines, tx, ty in labels_to_draw:
            draw_text_with_transparent_black_bg(
                canvas, lines, (tx, ty), font,
                [font_scale, code_scale], text_thickness,
            )

        cv2.imwrite(str(vis_dir / Path(file_name).name), canvas)
        saved += 1

    return saved, missing


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replace placeholder color_name values in Prediction_full.json "
            "using the functions of 1_colour_patch.py."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Input JSON (default: Prediction_full.json)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSON (default: "
                             "Prediction_full_with_colourname.json)")
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR,
                        help="Directory containing original page images")
    parser.add_argument("--vis-dir", type=Path, default=DEFAULT_VIS_DIR,
                        help="Output folder for colour-name visualizations "
                             "(default: output_visualized_predict_Colour)")
    parser.add_argument("--colour-patch", type=Path,
                        default=COLOUR_PATCH_PATH,
                        help="Path to 1_colour_patch.py")
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    print()
    print("=" * 80)
    print("RGB -> COLOUR NAME")
    print("=" * 80)
    print(f"Input JSON:\n  {args.input}")
    print(f"Output JSON:\n  {args.output}")
    print(f"Image directory:\n  {args.img_dir}")
    print(f"Visualization:\n  {args.vis_dir}")
    print(f"Mapping source:\n  {args.colour_patch}")

    if not args.input.is_file():
        raise SystemExit(f"Input JSON not found:\n  {args.input}")

    cp = load_colour_patch(args.colour_patch)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {
        image["id"]: image
        for image in data.get("images", [])
        if "id" in image
    }
    annotations = data.get("annotations", [])

    print()
    print(f"Images      : {len(images)}")
    print(f"Annotations : {len(annotations)}")

    # Group placeholder annotations by page so each page is binarized once.
    by_file = {}
    n_skipped = 0
    for ann in annotations:
        if str(ann.get("color_name", "")).strip().lower() != PLACEHOLDER:
            n_skipped += 1
            continue
        meta = images.get(ann.get("image_id"))
        file_name = str(meta.get("file_name", "")) if meta else ""
        by_file.setdefault(file_name, []).append(ann)

    counts = Counter()
    reasons = Counter()

    for file_name, anns in tqdm(by_file.items(), desc="Pages"):
        path = resolve_image_path(args.img_dir, file_name)
        page = load_page(cp, path) if path is not None else None

        for ann in anns:
            colour_name, reason = name_for_annotation(cp, ann, page)
            ann["color_name"] = colour_name
            counts[colour_name] += 1
            reasons[reason] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_vis, n_vis_missing = visualize_colour_names(data, args.img_dir,
                                                  args.vis_dir)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output: {args.output}")
    print(f"Visualization: {args.vis_dir} "
          f"(saved={n_vis}, missing images={n_vis_missing})")
    print()
    print(f"Replaced placeholders  : {sum(counts.values())}")
    print(f"Skipped non-placeholder: {n_skipped}")
    for reason in ("ok", "no_fg", "too_few_pixels", "missing_rgb",
                   "no_image", "invalid_polygon"):
        if reasons[reason]:
            print(f"  {reason:<16}: {reasons[reason]}")
    print()
    print("Colour-name distribution:")
    for name, count in counts.most_common():
        print(f"  {name:<10}: {count}")


if __name__ == "__main__":
    main()
