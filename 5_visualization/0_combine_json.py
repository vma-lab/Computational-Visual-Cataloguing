import os
import json
from tqdm import tqdm

# === CONFIG ===
INPUT_DIR = "ColoredOrnedTooledJson_Output"
OUTPUT_PATH = "Prediction_full.json"

# === INIT ===
merged = {
    "info": {
        "description": "Merged COCO Text Detection Dataset",
        "version": "1.0",
        "year": 2025,
        "contributor": "Auto-merger",
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "text", "supercategory": "text"}]
}

# === Counters ===
img_id_counter = 0
ann_id_counter = 0

# === Process ===
json_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json")])
print(f"🔍 Found {len(json_files)} JSON files to merge.")

for jf in tqdm(json_files):
    path = os.path.join(INPUT_DIR, jf)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️  Error reading {jf}: {e}")
        continue

    # Extract lists safely
    images = data.get("images", [])
    anns = data.get("annotations", [])

    # Remap image and annotation IDs
    id_map = {}
    for img in images:
        old_id = img.get("id", None)
        img_id_counter += 1
        img["id"] = img_id_counter
        id_map[old_id] = img_id_counter
        merged["images"].append(img)

    for ann in anns:
        ann_id_counter += 1
        old_img_id = ann.get("image_id")
        new_img_id = id_map.get(old_img_id, old_img_id)
        ann["id"] = ann_id_counter
        ann["image_id"] = new_img_id
        merged["annotations"].append(ann)

print(f"✅ Total images: {len(merged['images'])}")
print(f"✅ Total annotations: {len(merged['annotations'])}")

# === SAVE ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"💾 Saved merged dataset to: {OUTPUT_PATH}")

