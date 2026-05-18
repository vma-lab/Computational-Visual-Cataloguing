import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from tqdm import tqdm

# === Config ===
PATCH_FOLDER = "text_patches_train_80"
OUTPUT_FOLDER = "text_patches_augmented_betterv3_80"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Augment count per type
AUG_SQUARE = 12
AUG_OTHERS = 6
AUG_TLT_EXTRA = 5 # Additional augmentations for tlt

SQUARE_RATIO_THR = 0.8
CLASS_NAMES = ["hor", "vrt", "tlt"]

# === Augmentation Pipelines ===
common_aug = A.Compose([
    A.Affine(rotate=(-10, 10), scale=(0.95, 1.05), shear={"x": 0, "y": 0}, p=1.0),
    A.Perspective(scale=(0.02, 0.06), keep_size=True, p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 3), p=0.5),
        A.GaussNoise(var_limit=(5, 15), p=0.5),
    ], p=0.3),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    A.CropAndPad(percent=(-0.1, 0), pad_mode=cv2.BORDER_CONSTANT, p=0.5),
])

tlt_aug = A.Compose([
    A.Affine(rotate=(0, 0), scale=(0.95, 1.05), shear={"x": 0, "y": 0}, p=1.0),
    A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.2),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 3), p=0.5),
        A.GaussNoise(var_limit=(5, 15), p=0.5),
    ], p=0.3),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    A.CoarseDropout(max_holes=2, max_height=0.2, max_width=0.2, p=0.2),
    A.CropAndPad(percent=(-0.1, 0), pad_mode=cv2.BORDER_CONSTANT, p=0.5),
])

# === Count class frequencies ===
class_counts = {cls: 0 for cls in CLASS_NAMES}
image_records = []

for fname in sorted(os.listdir(PATCH_FOLDER)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    parts = fname.split("___")
    if len(parts) < 2:
        continue
    label_parts = parts[1].split("_")
    if len(label_parts) < 2:
        continue

    label = label_parts[1].strip().lower()
    if label not in CLASS_NAMES:
        continue

    class_counts[label] += 1
    image_records.append((fname, label))

if not any(class_counts.values()):
    raise ValueError("❌ No class labels found. Check filename format.")

max_class_count = max(class_counts.values())

# === Augmentation loop ===
for fname, label in tqdm(image_records, desc="Augmenting"):
    img_path = os.path.join(PATCH_FOLDER, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not load {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    w, h = rect[1]

    if w == 0 or h == 0:
        continue

    ratio = min(w, h) / max(w, h)
    is_square_like = ratio >= SQUARE_RATIO_THR

    # Save original
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, fname), img)

    # --- Main Augmentation ---
    if label == "tlt":
        for i in range(1, AUG_TLT_EXTRA + 1):
            aug_img = tlt_aug(image=img)["image"]
            aug_name = fname.rsplit(".", 1)[0] + f"_tlt{i}.png"
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, aug_name), aug_img)
    else:
        aug_count = AUG_SQUARE if is_square_like else AUG_OTHERS
        for i in range(1, aug_count + 1):
            aug_img = common_aug(image=img)["image"]
            aug_name = fname.rsplit(".", 1)[0] + f"_aug{i}.png"
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, aug_name), aug_img)

    # --- Class Balancing ---
    need_extra = max_class_count - class_counts[label]
    if need_extra > 0:
        per_image_extra = min(10, int(np.ceil(need_extra / class_counts[label])))
        for i in range(1, per_image_extra + 1):
            if label == "tlt":
                aug_img = tlt_aug(image=img)["image"]
            else:
                aug_img = common_aug(image=img)["image"]
            aug_name = fname.rsplit(".", 1)[0] + f"_bal{i}.png"
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, aug_name), aug_img)

print("\n✅ Augmentation completed and saved to:", OUTPUT_FOLDER)

