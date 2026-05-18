import os
import json
import cv2
import numpy as np

# ========= Paths =========
json_path = "Prediction_full.json"
img_folder = "img"
output_folder = "output_visualized_predict_full"
os.makedirs(output_folder, exist_ok=True)


# ========= Style =========
TEXT_BG_ALPHA = 0.70          # Higher = darker background, easier to read
POLYGON_THICKNESS_BASE = 4    # Increase this: 5 = thick, 8 = very thick, 10 = extra thick


# ========= Helpers =========
def parse_color_code(color_code_str):
    """Convert 'R-G-B' string to BGR tuple for OpenCV."""
    if isinstance(color_code_str, str):
        parts = color_code_str.split("-")
        if len(parts) == 3:
            try:
                r, g, b = map(int, parts)
                return (b, g, r)
            except Exception:
                pass
    return (0, 255, 0)


def safe_poly(seg):
    """Convert a flat segmentation list into an Nx2 polygon array."""
    if seg is None or len(seg) < 6:
        return None

    try:
        return np.array(seg, dtype=np.int32).reshape((-1, 2))
    except Exception:
        return None


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

    # Keep text inside image
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


# ========= Load JSON =========
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = {img["id"]: img for img in data.get("images", [])}
annotations = data.get("annotations", [])


# ========= Group annotations by image_id =========
anns_by_image = {}

for ann in annotations:
    image_id = ann.get("image_id")
    anns_by_image.setdefault(image_id, []).append(ann)


# ========= Visualization =========
for img_id, img_info in images.items():
    file_name = img_info["file_name"]
    img_path = os.path.join(img_folder, file_name)

    if not os.path.exists(img_path):
        print(f"Warning: image not found: {file_name}")
        continue

    image = cv2.imread(img_path)

    if image is None:
        print(f"Warning: cannot open image: {file_name}")
        continue

    H, W = image.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Adaptive text size
    font_scale = max(0.45, min(1.2, max(H, W) / 1800.0))
    text_thickness = max(1, int(round(font_scale * 2.0)))

    # Thicker adaptive polygon line
    # Increase POLYGON_THICKNESS_BASE above if you want even thicker polygons.
    polygon_thickness = max(
        POLYGON_THICKNESS_BASE,
        int(round(max(H, W) / 900.0))
    )

    img_anns = anns_by_image.get(img_id, [])

    # Store labels and draw them later, so text is always on top
    labels_to_draw = []

    # ===== First pass: draw all polygons =====
    for ann in img_anns:
        segs = ann.get("segmentation", [])
        if not segs:
            continue

        color = parse_color_code(ann.get("color_code", ""))

        orientation = ann.get("pred_orientation", ann.get("orientation", "N/A"))
        writing_tool = ann.get("pred_writing_tool", ann.get("writing_tool", "N/A"))
        label = f"{orientation}, {writing_tool}"

        for seg in segs:
            pts = safe_poly(seg)
            if pts is None:
                continue

            cv2.polylines(
                image,
                [pts],
                isClosed=True,
                color=color,
                thickness=polygon_thickness,
            )

            # Top-left-most point of polygon
            top_idx_sorted = np.lexsort((pts[:, 0], pts[:, 1]))
            top_point = pts[top_idx_sorted[0]]

            tx, ty = int(top_point[0]), int(top_point[1])

            # Put label slightly above polygon, but still visible
            text_offset = max(10, int(font_scale * 22))
            ty_text = max(text_offset, ty - text_offset)

            labels_to_draw.append((label, tx, ty_text))

    # ===== Second pass: draw all text labels on top =====
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

    out_path = os.path.join(output_folder, os.path.basename(file_name))
    cv2.imwrite(out_path, image)
    print(f"Saved: {out_path}")

print("Visualization complete.")
