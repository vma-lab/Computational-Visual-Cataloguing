import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import albumentations as A
from multiprocessing import Pool, cpu_count
from pathlib import Path

# === CONFIG ===
IMG_DIR = "img"  # <- set your input images root; leave "" to auto-resolve
ANNOT_PATH = "Converted_DLA_train.json"
OUTPUT_DIR = "augmented_result/img"
AUG_TIMES = 4
NUM_WORKERS = max(1, cpu_count() - 1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(Path(OUTPUT_DIR).parent, exist_ok=True)

transform = A.ReplayCompose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.15),
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.6),
    A.Affine(scale=(0.96, 1.04), shear=(-3, 3), p=0.6),
    A.Perspective(scale=(0.01, 0.02), p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.6),
    A.RandomGamma(gamma_limit=(95, 105), p=0.2),
    A.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.01, p=0.2),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.08, 0.15), p=0.2),
    A.MultiplicativeNoise(multiplier=(0.96, 1.04), per_channel=True, p=0.15),
    A.ImageCompression(p=0.15)
])

# === LOAD COCO JSON ===
with open(ANNOT_PATH, "r") as f:
    coco = json.load(f)

ann_map = {}
for ann in coco["annotations"]:
    ann_map.setdefault(ann["image_id"], []).append(ann)

categories = coco["categories"]

# Build a quick lookup for original image size by id
orig_img_by_id = {im["id"]: im for im in coco["images"]}

# --- Path resolver ---
def resolve_image_path(file_name: str) -> str:
    """
    Try to resolve an image path from several candidate roots:
    1) IMG_DIR (if set)
    2) current directory
    3) ./img (sibling of annot)
    4) folder next to ANNOT_PATH named 'img'
    5) treat file_name as already containing a relative path
    """
    candidates = []
    fn = file_name

    if IMG_DIR:
        candidates.append(Path(IMG_DIR) / fn)
    candidates.append(Path(fn))
    candidates.append(Path("img") / fn)
    candidates.append(Path(ANNOT_PATH).parent / "img" / fn)
    # If file_name already includes subfolders like "img/xxx.jpg"
    candidates.append(Path(file_name))

    for c in candidates:
        if c.exists():
            return str(c)
    return str(Path(IMG_DIR) / fn) if IMG_DIR else file_name

def clean_polygon(contour, img_w, img_h, max_points=50):
    if contour is None or len(contour) < 4:
        return None
    if contour.ndim != 2 or contour.shape[1] != 2:
        return None

    if len(contour) > max_points:
        peri = cv2.arcLength(contour, True)
        epsilon = 0.001 * peri
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        while len(simplified) > max_points and epsilon < 0.05 * peri:
            epsilon *= 1.2
            simplified = cv2.approxPolyDP(simplified, epsilon, True)
        contour = simplified.squeeze(1)

    if len(contour) < 4 or contour.ndim != 2 or contour.shape[1] != 2:
        return None

    flat = contour.flatten()
    if len(flat) < 8 or len(flat) % 2 != 0:
        return None

    area = cv2.contourArea(contour)
    if area < 50:
        return None

    return contour, area

def process_image(args):
    img_info, anns, img_id_start, ann_id_start = args
    img_path = resolve_image_path(img_info["file_name"])
    img_orig = cv2.imread(img_path)

    if img_orig is None:
        print(f"⚠️ Cannot read image: {img_info['file_name']} (tried: {img_path})")
        return [], []

    img_results = []
    ann_results = []

    for aug_idx in range(AUG_TIMES + 1):
        if aug_idx == 0:
            img_aug = img_orig.copy()
            replay = None
            new_filename = Path(img_info["file_name"]).name
        else:
            replayed = transform(image=img_orig)
            img_aug = replayed["image"]
            replay = replayed["replay"]
            name_prefix = os.path.splitext(Path(img_info['file_name']).name)[0]
            new_filename = f"{name_prefix}_aug_{aug_idx}.jpg"

        out_path = os.path.join(OUTPUT_DIR, new_filename)
        ok = cv2.imwrite(out_path, img_aug)
        if not ok:
            print(f"⚠️ Failed to write: {out_path}")

        curr_img_id = img_id_start + aug_idx
        img_results.append({
            "id": curr_img_id,
            "file_name": new_filename,
            "width": int(img_aug.shape[1]),
            "height": int(img_aug.shape[0]),
        })

        for ann in anns:
            mask = np.zeros(img_orig.shape[:2], dtype=np.uint8)
            poly = np.array(ann["segmentation"][0]).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], color=1)

            if aug_idx == 0:
                mask_aug = mask
            else:
                mask_aug = A.ReplayCompose.replay(replay, image=img_orig, mask=mask)["mask"]

            contours, _ = cv2.findContours(mask_aug, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cleaned = clean_polygon(contour.squeeze(1), img_aug.shape[1], img_aug.shape[0])
                if cleaned is None:
                    continue
                contour, area = cleaned
                x, y, w_box, h_box = cv2.boundingRect(contour)
                segmentation = contour.flatten().astype(float).tolist()

                ann_results.append({
                    "id": ann_id_start,
                    "image_id": curr_img_id,
                    "category_id": ann["category_id"],
                    "segmentation": [segmentation],
                    "bbox": [float(x), float(y), float(w_box), float(h_box)],
                    "area": float(area),
                    "iscrowd": ann.get("iscrowd", 0)
                })
                ann_id_start += 1

    return img_results, ann_results

# === MULTIPROCESSING ===
args_list = []
start_img_id = 10000
start_ann_id = 20000
for idx, img_info in enumerate(coco["images"]):
    anns = ann_map.get(img_info["id"], [])
    args_list.append((
        img_info,
        anns,
        start_img_id + idx * (AUG_TIMES + 1),
        start_ann_id + idx * 1000
    ))

with Pool(NUM_WORKERS) as pool:
    results = list(tqdm(pool.imap_unordered(process_image, args_list), total=len(args_list)))

# === COMBINE RESULTS ===
new_images = []
new_annotations = []
for imgs, anns in results:
    new_images.extend(imgs)
    new_annotations.extend(anns)

# === SAVE TRAINING JSON ===
train_json = {
    "info": coco.get("info", {}),
    "licenses": coco.get("licenses", []),
    "images": new_images,
    "annotations": new_annotations,
    "categories": categories
}
with open("augmented_result/instances_training.json", "w") as f:
    json.dump(train_json, f, indent=2)

# === SAVE VALIDATION + TEST (CLEANED ORIGINAL) ===
cleaned_val = {
    "info": coco.get("info", {}),
    "licenses": coco.get("licenses", []),
    "images": [],
    "annotations": [],
    "categories": categories
}

# map old -> new ids and store sizes correctly
val_img_id_map = {}
new_img_id = 30000
for img in coco["images"]:
    val_img_id_map[img["id"]] = new_img_id
    cleaned_val["images"].append({
        "id": new_img_id,
        "file_name": Path(img["file_name"]).name,
        "width": int(img["width"]),
        "height": int(img["height"]),
    })
    new_img_id += 1

# proper lookup for width/height per original image_id
size_by_old_id = {im["id"]: (int(im["width"]), int(im["height"])) for im in coco["images"]}

new_ann_id = 40000
for ann in coco["annotations"]:
    w0, h0 = size_by_old_id[ann["image_id"]]
    poly = np.array(ann["segmentation"][0]).reshape((-1, 2)).astype(np.int32)
    cleaned = clean_polygon(poly, w0, h0)
    if cleaned is None:
        continue
    poly2, area2 = cleaned
    x, y, w_box, h_box = cv2.boundingRect(poly2)
    segmentation = poly2.flatten().astype(float).tolist()

    cleaned_val["annotations"].append({
        "id": new_ann_id,
        "image_id": val_img_id_map[ann["image_id"]],
        "category_id": ann["category_id"],
        "segmentation": [segmentation],
        "bbox": [float(x), float(y), float(w_box), float(h_box)],
        "area": float(area2),
        "iscrowd": ann.get("iscrowd", 0)
    })
    new_ann_id += 1

with open("augmented_result/instances_val.json", "w") as f:
    json.dump(cleaned_val, f, indent=2)

with open("augmented_result/instances_test.json", "w") as f:
    json.dump(cleaned_val, f, indent=2)

print("✅ Done: Multiprocessing augmentation complete.")
print(f"Saved images to: {OUTPUT_DIR}")

