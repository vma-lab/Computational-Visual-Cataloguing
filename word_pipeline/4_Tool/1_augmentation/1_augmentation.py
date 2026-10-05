import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from collections import defaultdict
import random

# === Config ===
PATCH_FOLDER = "text_patches_train_80"
OUTPUT_FOLDER = "augmented_patches_tool_v3"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Writing tool labels ===
tool_labels = {"pcl": 0, "ink": 1}
AUG_TIMES = 5  # mỗi ảnh augment 5 lần

# Fixed seed so the augmented training set is reproducible
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# === Augmentation pipeline (giảm nhẹ) ===
augmentation = A.Compose([
    # --- Blur + Noise (giảm nhẹ) ---
    A.OneOf([
        A.GaussianBlur(blur_limit=(1, 3), p=0.6),
        A.MotionBlur(blur_limit=3, p=0.4),
    ], p=0.6),

    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 30.0), mean=0, p=0.7),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.25), p=0.7),
    ], p=0.5),

    # --- Color Jittering (giảm nhẹ) ---
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
], p=1.0)

# === Store samples by class ===
tool_to_images = defaultdict(list)

# === Step 1: Augment each original image 5 times ===
for fname in tqdm(sorted(os.listdir(PATCH_FOLDER)), desc="Initial augmentation"):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    parts = fname.split("___")
    if len(parts) < 2:
        continue
    label_parts = parts[1].split("_")
    if len(label_parts) < 3:
        continue

    tool = label_parts[2].lower()
    if tool not in tool_labels:
        continue

    img_path = os.path.join(PATCH_FOLDER, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Save original
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, fname), img)

    base_name, ext = os.path.splitext(fname)
    for i in range(AUG_TIMES):
        aug_img = augmentation(image=img)["image"]
        aug_name = f"{base_name}_aug{i+1}{ext}"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, aug_name), aug_img)

    # Save for balancing
    tool_to_images[tool].append((base_name, img, ext))

# === Step 2: Balance classes ===
tool_counts = {tool: len(tool_to_images[tool]) * (AUG_TIMES + 1) for tool in tool_to_images}
max_count = max(tool_counts.values())

print(f"\n🔢 Image count after initial augmentation:")
for tool, count in tool_counts.items():
    print(f" - {tool}: {count} images")

for tool in tool_labels:
    if tool not in tool_to_images:
        print(f"⚠️ No samples found for tool '{tool}'")
        continue

    current_total = tool_counts[tool]
    deficit = max_count - current_total

    if deficit <= 0:
        continue

    print(f"🔁 Balancing class '{tool}' with {deficit} extra augmentations...")

    samples = tool_to_images[tool]
    for i in range(deficit):
        base_name, base_img, ext = random.choice(samples)
        aug_img = augmentation(image=base_img)["image"]
        aug_name = f"{base_name}_balance{i+1}{ext}"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, aug_name), aug_img)

print("\n✅ Augmentation and class balancing completed.")

