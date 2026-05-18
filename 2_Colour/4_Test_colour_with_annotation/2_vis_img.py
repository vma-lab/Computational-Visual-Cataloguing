import os
import json
import cv2
import numpy as np

# ========= Paths =========
IMG_FOLDER = "img"
JSON_FOLDER = "ColoredJson_Output"
OUTPUT_FOLDER = "visualized_colour_labels"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ========= Options =========
DRAW_POLYGON = True
DRAW_LABEL = True

# ============================================================
# ID FILTER OPTION
# ============================================================
# None -> visualize all annotations
# Example:
#SHOW_ONLY_IDS = [1026]
SHOW_ONLY_IDS = None

# If True:
#   When SHOW_ONLY_IDS is not None, only selected IDs are drawn/labeled.
# If False:
#   Polygons are drawn for all, but labels only for selected IDs.
ONLY_DRAW_SELECTED_IDS = True

# ========= Text Style =========
FONT_SCALE = 0.26
TEXT_THICKNESS = 1
TEXT_BG_ALPHA = 0.60

# ========= Polygon Style =========
POLYGON_THICKNESS_BASE = 2
DEFAULT_POLYGON_COLOR = (255, 0, 0)  # fallback blue in BGR


def safe_poly(seg):
    if seg is None or len(seg) < 6:
        return None

    try:
        return np.array(seg, dtype=np.int32).reshape((-1, 2))
    except Exception:
        return None


def parse_color_code_to_bgr(color_code):
    """
    Convert color_code from RGB string to OpenCV BGR.

    Example:
        "75-71-60" -> (60, 71, 75)
    """
    try:
        parts = str(color_code).split("-")

        if len(parts) != 3:
            return DEFAULT_POLYGON_COLOR

        r, g, b = map(int, parts)

        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        return (b, g, r)

    except Exception:
        return DEFAULT_POLYGON_COLOR


def should_process_id(instance_id):
    """
    Return True if this instance should be visualized.

    SHOW_ONLY_IDS = None:
        process all

    SHOW_ONLY_IDS = [id1, id2, ...]:
        process only selected IDs
    """
    if SHOW_ONLY_IDS is None:
        return True

    return instance_id in SHOW_ONLY_IDS


def should_draw_label_id(instance_id):
    """
    Return True if this instance should have label text.
    """
    if SHOW_ONLY_IDS is None:
        return True

    return instance_id in SHOW_ONLY_IDS


def draw_text_with_transparent_black_bg(
    img,
    text,
    org,
    font,
    scale,
    thickness,
    alpha=0.60,
):
    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    x, y = org
    H, W = img.shape[:2]

    pad = 3

    x = max(pad, min(x, W - tw - pad - 1))
    y = max(th + pad, min(y, H - baseline - pad - 1))

    x1 = max(0, x - pad)
    y1 = max(0, y - th - pad)
    x2 = min(W - 1, x + tw + pad)
    y2 = min(H - 1, y + baseline + pad)

    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2].copy()
    black_bg = np.zeros_like(roi)

    blended = cv2.addWeighted(
        black_bg,
        alpha,
        roi,
        1 - alpha,
        0
    )

    img[y1:y2, x1:x2] = blended

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def find_image_path(file_name):
    direct_path = os.path.join(IMG_FOLDER, file_name)

    if os.path.exists(direct_path):
        return direct_path

    base = os.path.splitext(file_name)[0]

    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        candidate = os.path.join(IMG_FOLDER, base + ext)

        if os.path.exists(candidate):
            return candidate

    return None


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def visualize_one_json(json_file):
    json_path = os.path.join(JSON_FOLDER, json_file)

    data = load_json(json_path)

    images = data.get("images", [])

    if not images:
        print(f"Warning: no image info in {json_file}")
        return

    image_info = images[0]
    file_name = image_info.get("file_name")

    if file_name is None:
        print(f"Warning: missing file_name in {json_file}")
        return

    img_path = find_image_path(file_name)

    if img_path is None:
        print(f"Warning: image not found for {file_name}")
        return

    image = cv2.imread(img_path)

    if image is None:
        print(f"Warning: cannot open image: {img_path}")
        return

    H, W = image.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = FONT_SCALE
    text_thickness = TEXT_THICKNESS

    polygon_thickness = max(
        POLYGON_THICKNESS_BASE,
        int(round(max(H, W) / 1600.0))
    )

    labels_to_draw = []

    annotations = data.get("annotations", [])

    total_annotations = len(annotations)
    drawn_annotations = 0
    labeled_annotations = 0

    for ann in annotations:
        instance_id = ann.get("id", "N/A")

        selected = should_process_id(instance_id)

        if ONLY_DRAW_SELECTED_IDS and not selected:
            continue

        color_name = ann.get("color_name", "unknown")
        color_code = ann.get("color_code", "N/A")

        polygon_color = parse_color_code_to_bgr(color_code)

        color_code_display = str(color_code).replace("-", ".")

        label = (
            f"{instance_id},"
            f"{color_name},"
            f"{color_code_display}"
        )

        segs = ann.get("segmentation", [])

        if not segs:
            continue

        for seg in segs:
            pts = safe_poly(seg)

            if pts is None:
                continue

            if DRAW_POLYGON:
                cv2.polylines(
                    image,
                    [pts],
                    isClosed=True,
                    color=polygon_color,
                    thickness=polygon_thickness,
                )

            drawn_annotations += 1

            if DRAW_LABEL and should_draw_label_id(instance_id):
                top_idx_sorted = np.lexsort(
                    (pts[:, 0], pts[:, 1])
                )

                top_point = pts[top_idx_sorted[0]]

                tx = int(top_point[0])
                ty = int(top_point[1])

                text_offset = max(
                    8,
                    int(font_scale * 20)
                )

                ty_text = max(
                    text_offset,
                    ty - text_offset
                )

                labels_to_draw.append(
                    (label, tx, ty_text)
                )

                labeled_annotations += 1

    if DRAW_LABEL:
        for label, tx, ty_text in labels_to_draw:
            draw_text_with_transparent_black_bg(
                image,
                label,
                (tx, ty_text),
                font,
                font_scale,
                text_thickness,
                alpha=TEXT_BG_ALPHA,
            )

    out_path = os.path.join(
        OUTPUT_FOLDER,
        os.path.basename(file_name)
    )

    cv2.imwrite(out_path, image)

    print(
        f"Saved: {out_path} | "
        f"total anns={total_annotations} | "
        f"drawn={drawn_annotations} | "
        f"labels={labeled_annotations}"
    )


if __name__ == "__main__":
    if not os.path.exists(IMG_FOLDER):
        print(f"Error: image folder not found: {IMG_FOLDER}")
        raise SystemExit

    if not os.path.exists(JSON_FOLDER):
        print(f"Error: JSON folder not found: {JSON_FOLDER}")
        raise SystemExit

    json_files = [
        f for f in os.listdir(JSON_FOLDER)
        if f.lower().endswith(".json")
    ]

    print("=" * 80)
    print(f"Images folder       : {IMG_FOLDER}")
    print(f"JSON folder         : {JSON_FOLDER}")
    print(f"Output folder       : {OUTPUT_FOLDER}")
    print(f"JSON files found    : {len(json_files)}")
    print(f"SHOW_ONLY_IDS       : {SHOW_ONLY_IDS}")
    print(f"ONLY_DRAW_SELECTED  : {ONLY_DRAW_SELECTED_IDS}")
    print("=" * 80)

    for json_file in json_files:
        visualize_one_json(json_file)

    print("=" * 80)
    print("Visualization complete.")
    print("=" * 80)
