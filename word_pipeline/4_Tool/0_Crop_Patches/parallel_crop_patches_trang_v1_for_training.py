import os
import json
import numpy as np
from PIL import Image, ImageDraw
from joblib import Parallel, delayed
from tqdm import tqdm

# === CONFIG ===
annotation_file = "instances_training.json"
image_folder = "imgs"
output_folder = "text_patches_train_80"
os.makedirs(output_folder, exist_ok=True)

# === ABBREVIATION MAPS ===
abbr = {
    "orientation": {"horizontal": "hor", "vertical": "vrt", "tilt": "tlt"},
    "writing_tool": {"pen": "pn", "pencil": "pcl", "ink": "ink"},
    "language": {"english": "en", "german": "de", "french": "fr"}
}

# === LOAD ANNOTATIONS & IMAGE MAP ===
with open(annotation_file, "r", encoding="utf-8") as f:
    data = json.load(f)

image_id_to_filename = {
    img["id"]: os.path.basename(img["file_name"])
    for img in data.get("images", [])
}

annotations = data.get("annotations", [])


# === WORKER FUNCTION ===
def process_annotation(ann):
    try:
        image_id = ann["image_id"]
        if image_id not in image_id_to_filename:
            return

        image_name = image_id_to_filename[image_id]
        image_path = os.path.join(image_folder, image_name)
        if not os.path.exists(image_path):
            return

        # Load image only once
        img = Image.open(image_path).convert("RGB")

        # Extract polygon and validate
        polygon = ann.get("segmentation", [[]])[0]
        if len(polygon) < 6:
            return
        polygon_xy = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]

        # Create binary mask and masked image
        mask = Image.new('L', img.size, 0)
        ImageDraw.Draw(mask).polygon(polygon_xy, fill=1)
        mask_array = np.array(mask)
        img_array = np.array(img)

        masked_array = np.ones_like(img_array) * 255  # white background
        masked_array[mask_array == 1] = img_array[mask_array == 1]
        masked_img = Image.fromarray(masked_array)

        # Crop to bounding box
        bbox = mask.getbbox()
        if bbox is None:
            return
        cropped_patch = masked_img.crop(bbox)

        # Generate filename
        base_name = os.path.splitext(image_name)[0]
        aid = ann["id"]
        ori = abbr["orientation"].get(ann.get("orientation", ""), "unk")
        tool = abbr["writing_tool"].get(ann.get("writing_tool", ""), "unk")
        lang = abbr["language"].get(ann.get("language", ""), "unk")
        patch_name = f"{base_name}___{aid}_{ori}_{tool}_{lang}.png"
        patch_path = os.path.join(output_folder, patch_name)

        # Save cropped patch
        cropped_patch.save(patch_path)

    except Exception as e:
        print(f"[ERROR] Annotation ID {ann.get('id')} — {e}")


# === PARALLEL EXECUTION ===
num_jobs = max(2, os.cpu_count() - 2)
print(f"🔄 Processing {len(annotations)} annotations using {num_jobs} workers...")

Parallel(n_jobs=num_jobs, backend="loky")(
    delayed(process_annotation)(ann) for ann in tqdm(annotations)
)

print("✅ Done.")

