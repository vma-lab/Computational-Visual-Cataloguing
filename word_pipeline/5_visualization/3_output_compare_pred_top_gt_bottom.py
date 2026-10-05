import os
import cv2
import numpy as np

# ========= Paths =========
pred_folder = "output_visualized_predict_full"
gt_folder = "output_visualized_GT_updated"
output_folder = "output_compare_pred_top_gt_bottom"

os.makedirs(output_folder, exist_ok=True)


def resize_to_width(img, target_width):
    """Resize image to target width while preserving aspect ratio."""
    h, w = img.shape[:2]

    if w == target_width:
        return img

    scale = target_width / float(w)
    new_h = int(round(h * scale))

    return cv2.resize(
        img,
        (target_width, new_h),
        interpolation=cv2.INTER_AREA,
    )


def pad_to_width(img, target_width):
    """Pad image to target width using white background."""
    h, w = img.shape[:2]

    if w >= target_width:
        return img

    pad_right = target_width - w

    return cv2.copyMakeBorder(
        img,
        0,
        0,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


# ========= Find common image names =========
pred_files = {
    f for f in os.listdir(pred_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
}

gt_files = {
    f for f in os.listdir(gt_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
}

common_files = sorted(pred_files & gt_files)

print(f"Prediction images: {len(pred_files)}")
print(f"Ground-truth images: {len(gt_files)}")
print(f"Common images: {len(common_files)}")


# ========= Combine images =========
for fname in common_files:
    pred_path = os.path.join(pred_folder, fname)
    gt_path = os.path.join(gt_folder, fname)

    pred_img = cv2.imread(pred_path)
    gt_img = cv2.imread(gt_path)

    if pred_img is None:
        print(f"Warning: cannot read prediction image: {fname}")
        continue

    if gt_img is None:
        print(f"Warning: cannot read GT image: {fname}")
        continue

    # Resize both images to same width
    target_width = max(pred_img.shape[1], gt_img.shape[1])

    pred_img = resize_to_width(pred_img, target_width)
    gt_img = resize_to_width(gt_img, target_width)

    pred_img = pad_to_width(pred_img, target_width)
    gt_img = pad_to_width(gt_img, target_width)

    # Stack vertically:
    # prediction on top
    # ground truth on bottom
    combined = np.vstack([pred_img, gt_img])

    out_path = os.path.join(output_folder, fname)

    cv2.imwrite(out_path, combined)

    print(f"Saved: {out_path}")

print("Done.")
