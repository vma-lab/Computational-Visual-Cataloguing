import os
import json
import cv2
import numpy as np

# ========= Paths =========
GT_JSON_PATH = "GT_updated.json"
PRED_JSON_PATH = "Prediction_full.json"

IMG_FOLDER = "img"
OUTPUT_FOLDER = "output_visualized_GT_updated"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ========= Options =========
# "all_gt"       : visualize all images listed in GT_updated.json
# "intersection": visualize only images that appear in both GT_updated.json and Prediction_full.json
VIS_MODE = "intersection"

# ========= Style =========
BLUE = (255, 0, 0)          # OpenCV BGR: blue
TEXT_BG_ALPHA = 0.70        # higher = darker black background
DRAW_LABEL = True


# ========= Helpers =========
def safe_poly(seg):
    """Convert a flat segmentation list into an Nx2 polygon array."""
    if seg is None or len(seg) < 6:
        return None

    try:
        return np.array(seg, dtype=np.int32).reshape((-1, 2))
    except Exception:
        return None


def normalize_label(value):
    """Normalize GT labels to short labels for compact visualization."""
    if value is None:
        return "N/A"

    value = str(value).lower()

    mapping = {
        "horizontal": "hor",
        "vertical": "vrt",
        "tilt": "tlt",
        "tilted": "tlt",
        "pencil": "pcl",
        "ink": "ink",
    }

    return mapping.get(value, value)


def draw_text_with_transparent_black_bg(
    img,
    text,
    org,
    font,
    scale,
    thickness,
    alpha=0.70,
):
    """Draw white text on a semi-transparent black background."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x, y = org
    H, W = img.shape[:2]

    pad = max(4, int(scale * 6))

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
    blended = cv2.addWeighted(black_bg, alpha, roi, 1 - alpha, 0)

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


def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_annotations_by_image_id(annotations):
    """Group COCO annotations by image_id."""
    anns_by_image = {}

    for ann in annotations:
        image_id = ann.get("image_id")
        anns_by_image.setdefault(image_id, []).append(ann)

    return anns_by_image


def get_file_names_from_json(path):
    """Return image file names from a COCO-style JSON file."""
    if not os.path.exists(path):
        return set()

    data = load_json(path)
    return {img.get("file_name") for img in data.get("images", [])}


# ========= Load GT JSON =========
gt_data = load_json(GT_JSON_PATH)

gt_images = {img["id"]: img for img in gt_data.get("images", [])}
gt_annotations = gt_data.get("annotations", [])
gt_anns_by_image = group_annotations_by_image_id(gt_annotations)

# ========= Optional intersection with Prediction_full.json =========
if VIS_MODE == "intersection":
    pred_file_names = get_file_names_from_json(PRED_JSON_PATH)
    selected_images = {
        img_id: img_info
        for img_id, img_info in gt_images.items()
        if img_info.get("file_name") in pred_file_names
    }
elif VIS_MODE == "all_gt":
    selected_images = gt_images
else:
    raise ValueError("VIS_MODE must be either 'all_gt' or 'intersection'.")

print(f"Visualization mode: {VIS_MODE}")
print(f"Images selected: {len(selected_images)}")


# ========= Visualization =========
for img_id, img_info in selected_images.items():
    file_name = img_info["file_name"]
    img_path = os.path.join(IMG_FOLDER, file_name)

    if not os.path.exists(img_path):
        print(f"Warning: image not found: {file_name}")
        continue

    image = cv2.imread(img_path)

    if image is None:
        print(f"Warning: cannot open image: {file_name}")
        continue

    H, W = image.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(1.2, max(H, W) / 1800.0))
    text_thickness = max(1, int(round(font_scale * 2.0)))
    polygon_thickness = max(2, int(round(max(H, W) / 1600.0)))

    img_anns = gt_anns_by_image.get(img_id, [])

    labels_to_draw = []

    # ===== First pass: draw all polygons =====
    for ann in img_anns:
        segs = ann.get("segmentation", [])
        if not segs:
            continue

        orientation = normalize_label(ann.get("orientation", "N/A"))
        writing_tool = normalize_label(ann.get("writing_tool", "N/A"))
        label = f"{orientation}, {writing_tool}"

        for seg in segs:
            pts = safe_poly(seg)
            if pts is None:
                continue

            cv2.polylines(
                image,
                [pts],
                isClosed=True,
                color=BLUE,
                thickness=polygon_thickness,
            )

            if DRAW_LABEL:
                top_idx_sorted = np.lexsort((pts[:, 0], pts[:, 1]))
                top_point = pts[top_idx_sorted[0]]

                tx, ty = int(top_point[0]), int(top_point[1])

                text_offset = max(10, int(font_scale * 22))
                ty_text = max(text_offset, ty - text_offset)

                labels_to_draw.append((label, tx, ty_text))

    # ===== Second pass: draw labels on top of all polygons =====
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

    out_path = os.path.join(OUTPUT_FOLDER, os.path.basename(file_name))
    cv2.imwrite(out_path, image)
    print(f"Saved: {out_path}")

print("Visualization complete.")
