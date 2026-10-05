"""
Qualitative inference only (no ground-truth coarse colour names required).

This script extracts ink RGB via binarization + GMM and writes patches / JSON.
It does NOT classify coarse colour names. 

The predicted RGB is visualized as the matching colour fill on the predicted
polygon in the final pipeline visualisation output.

Mapping from this predicted RGB to coarse colour names (for quantitative
evaluation vs GT) is presented in:
  2_Colour/4_Test_colour_with_annotation/1_colour_patch.py
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

# === CONFIG ===
image_folder = "imgs"
json_folder = "jsons"
output_folder = "ColoredName_" + "CropedPatch_Output"
output_json_folder = "ColoredJson_" + "Output"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_json_folder, exist_ok=True)

COLOR_NAME_PLACEHOLDER = "colorname"

LOCAL_BG_PADDING = 40


def sauvola_binarization(img_pil,
                         win_list=(15, 31, 61),
                         k_strict=0.08, k_loose=0.28,
                         bernsen_contrast_min=12,
                         post_open=3, post_close=3):
    rgb = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = gray / 255.0

    bg = cv2.GaussianBlur(gray, (0, 0), 51)
    bg = np.clip(bg, 1e-6, 1.0)
    norm = np.clip((gray / bg) * 0.5 + 0.25, 0.0, 1.0)

    blur_r = cv2.GaussianBlur(norm, (0, 0), 5)
    retinex = np.log(norm + 1e-6) - np.log(blur_r + 1e-6)

    rmin, rmax = np.percentile(retinex, (1, 99))
    if rmax > rmin:
        retinex = (retinex - rmin) / (rmax - rmin)
    retinex = np.clip(retinex, 0, 1).astype(np.float32)


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply((retinex * 255).astype(np.uint8)).astype(np.float32) / 255.0


    def local_stats(img, w):
        k = (w, w)
        m = cv2.boxFilter(img, ddepth=-1, ksize=k, normalize=True)
        m2 = cv2.boxFilter(img * img, ddepth=-1, ksize=k, normalize=True)
        s = cv2.sqrt(np.maximum(m2 - m * m, 0))
        st = cv2.getStructuringElement(cv2.MORPH_RECT, (w, w))
        vmin = cv2.erode(img, st)
        vmax = cv2.dilate(img, st)
        return m, s, vmin, vmax

    def sauvola(img, w, k, R=0.5):
        m, s, _, _ = local_stats(img, w)
        T = m * (1.0 + k * ((s / R) - 1.0))
        return (img < T).astype(np.uint8)

    def wolf_jolion(img, w, k):
        m, s, vmin, _ = local_stats(img, w)
        R = np.max(s) + 1e-6
        T = m + k * ((s / R) - 1.0) * (m - vmin)
        return (img < T).astype(np.uint8)

    def bernsen(img, w, cmin):
        _, _, vmin, vmax = local_stats(img, w)
        C = vmax - vmin
        T = (vmax + vmin) * 0.5
        return ((img < T) & (C >= (cmin / 255.0))).astype(np.uint8)

    H, W = enh.shape
    votes_strict = np.zeros((H, W), np.int16)
    votes_loose = np.zeros((H, W), np.int16)

    for w in win_list:
        b_s_sau = sauvola(enh, w, k_strict)
        b_s_wjf = wolf_jolion(enh, w, k=-0.2)
        b_s_ber = bernsen(enh, w, bernsen_contrast_min)
        votes_strict += b_s_sau + b_s_wjf + b_s_ber

        b_l_sau = sauvola(enh, w, k_loose)
        b_l_wjf = wolf_jolion(enh, w, k=+0.2)
        b_l_ber = bernsen(enh, w, max(8, bernsen_contrast_min - 4))
        votes_loose += b_l_sau + b_l_wjf + b_l_ber

    strong = (votes_strict >= 4).astype(np.uint8)
    weak = (votes_loose >= 3).astype(np.uint8)

    k3 = np.ones((3, 3), np.uint8)
    strong_dil = cv2.dilate(strong, k3, 1)
    fused = ((strong == 1) | ((weak == 1) & (strong_dil == 1))).astype(np.uint8)

    if post_open > 0:
        fused = cv2.morphologyEx(
            fused,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (post_open, post_open))
        )
    if post_close > 0:
        fused = cv2.morphologyEx(
            fused,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (post_close, post_close))
        )

    dt = cv2.distanceTransform(fused, cv2.DIST_L2, 3)
    sw = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    min_area = max(16, int(2.0 * sw * sw))
    max_ar = 25

    num, lab, stats, _ = cv2.connectedComponentsWithStats(fused, connectivity=8)
    keep = np.zeros(num, np.uint8)
    keep[0] = 0
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        ar = max(w / h, h / w) if h > 0 and w > 0 else 1
        if area >= min_area and ar <= max_ar:
            keep[i] = 1
    cleaned = keep[lab]

    bin_img = (1 - cleaned) * 255
    return bin_img.astype(np.uint8)


def get_local_crop_bounds_from_polygon(polygon, width, height, padding=40):
    xs = polygon[0::2]
    ys = polygon[1::2]

    x1 = int(max(0, np.floor(min(xs) - padding)))
    y1 = int(max(0, np.floor(min(ys) - padding)))
    x2 = int(min(width, np.ceil(max(xs) + padding)))
    y2 = int(min(height, np.ceil(max(ys) + padding)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def extract_fg_bg_pixels_local(img_np, binary_img, polygon, padding=40):

    H, W = img_np.shape[:2]

    bounds = get_local_crop_bounds_from_polygon(
        polygon, W, H, padding=padding
    )

    if bounds is None:
        empty_mask = np.zeros((H, W), dtype=np.uint8)
        return (
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint8),
            empty_mask,
        )

    x1, y1, x2, y2 = bounds
    local_img = img_np[y1:y2, x1:x2]
    local_binary = binary_img[y1:y2, x1:x2]

    local_polygon_xy = [
        (polygon[i] - x1, polygon[i + 1] - y1)
        for i in range(0, len(polygon), 2)
    ]

    local_mask_pil = Image.new("L", (x2 - x1, y2 - y1), 0)
    ImageDraw.Draw(local_mask_pil).polygon(local_polygon_xy, fill=1)
    local_mask = np.array(local_mask_pil).astype(bool)

    local_foreground = (local_binary == 0)
    fg_mask = local_mask & local_foreground
    fg_pixels = local_img[fg_mask]

    bg_mask = (~local_mask) & (cv2.cvtColor(local_img, cv2.COLOR_RGB2GRAY) > 80)
    bg_pixels = local_img[bg_mask]

    if len(bg_pixels) < 10:
        bg_pixels = local_img[~local_mask]

    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = local_mask.astype(np.uint8)

    return fg_pixels, bg_pixels, full_mask


def get_text_color_via_gmm(fg_pixels, bg_pixels):
    """
    Extract median RGB of the text cluster (color_code only).
    """
    if len(fg_pixels) < 10 or len(bg_pixels) < 10:
        return np.array([0, 0, 0])

    bg_median = np.median(bg_pixels, axis=0).astype(int)

    try:
        gmm = GaussianMixture(n_components=2, covariance_type="tied", random_state=42)
        gmm.fit(fg_pixels)
        labels = gmm.predict(fg_pixels)
        centers = gmm.means_
        dists = np.linalg.norm(centers - bg_median, axis=1)
        text_cluster = np.argmax(dists)
        text_pixels = fg_pixels[labels == text_cluster]
    except Exception:
        text_pixels = fg_pixels

    if len(text_pixels) == 0:
        text_pixels = fg_pixels

    text_median = np.median(text_pixels, axis=0)
    return text_median.astype(int)



def process_image(image_name):
    image_base = os.path.splitext(image_name)[0]
    image_path = os.path.join(image_folder, image_name)
    json_path = os.path.join(json_folder, image_base + ".json")

    if not os.path.exists(image_path) or not os.path.exists(json_path):
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img = Image.open(image_path).convert("RGB")
    binary_img = sauvola_binarization(img)
    img_np = np.array(img)

    for ann in data.get("annotations", []):
        polygon = ann.get("segmentation", [[]])[0]
        if len(polygon) < 6:
            continue

        fg_pixels, bg_pixels, mask_np = extract_fg_bg_pixels_local(
            img_np=img_np,
            binary_img=binary_img,
            polygon=polygon,
            padding=LOCAL_BG_PADDING,
        )

        if len(fg_pixels) > 0:
            text_median = get_text_color_via_gmm(fg_pixels, bg_pixels)
        else:
            text_median = np.array([0, 0, 0])

        color_code = "-".join(map(str, text_median))
        color_name = COLOR_NAME_PLACEHOLDER

        ann["color_name"] = color_name
        ann["color_code"] = color_code

        masked_img = np.ones_like(img_np) * 255
        masked_img[mask_np == 1] = img_np[mask_np == 1]
        patch = Image.fromarray(masked_img)

        bbox = Image.fromarray((mask_np * 255).astype(np.uint8)).getbbox()
        if bbox:
            cropped = patch.crop(bbox)
          
            patch_name = f"{image_base}___{ann['id']}_{color_name}_{color_code}.png"
            cropped.save(os.path.join(output_folder, patch_name))

    with open(os.path.join(output_json_folder, image_base + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    if not os.path.exists(image_folder) or not os.path.exists(json_folder):
        print("Input folders not found.")
    else:
        image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        print("Processing images...")
        Parallel(n_jobs=max(2, os.cpu_count() - 2), backend="loky")(
            delayed(process_image)(img_name)
            for img_name in tqdm(image_files, desc="Processing", unit="image")
        )
        print(f"Saved patches to: {output_folder}")
        print(f"Saved JSON to: {output_json_folder}")
        print("Processing complete.")
