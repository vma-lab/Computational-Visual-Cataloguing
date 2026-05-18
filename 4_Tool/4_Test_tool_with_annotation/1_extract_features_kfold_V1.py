import os
import torch
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

# === Configuration ===
patch_folder = "text_patches_test_20"  # your input patches
model_name = "clip-ViT-L-14"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load CLIP model ===
model = SentenceTransformer(model_name, device=device)

# === Label Mapping for writing tool classification
tool_to_label = {"pcl": 0, "ink": 1}

# === Read files and extract embeddings ===
data = []

for fname in tqdm(sorted(os.listdir(patch_folder))):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    parts = fname.split("___")
    if len(parts) < 2:
        continue  # invalid name, skip

    label_parts = parts[1].split("_")
    if len(label_parts) < 3:
        continue  # not enough fields

    tool_code = label_parts[2].lower()  # 3rd field (index 2) is the tool: pcl or ink

    label = tool_to_label.get(tool_code)
    if label is None:
        continue  # skip if not pcl or ink

    img_path = os.path.join(patch_folder, fname)
    try:
        image = Image.open(img_path).convert("RGB")
        embedding = model.encode([image], convert_to_tensor=True)[0].cpu().numpy()
        data.append((embedding, label, fname))
    except Exception as e:
        print(f"⚠️ Failed to process {fname}: {e}")

# === Save features to disk ===
if data:
    features = np.array([x[0] for x in data])
    labels = np.array([x[1] for x in data])
    filenames = [x[2] for x in data]

    np.savez("writing_tool_features_clipL14_all.npz", features=features, labels=labels, filenames=filenames)
    print("✅ Saved: writing_tool_features_clipL14_all.npz")
else:
    print("❌ No valid patches found.")

