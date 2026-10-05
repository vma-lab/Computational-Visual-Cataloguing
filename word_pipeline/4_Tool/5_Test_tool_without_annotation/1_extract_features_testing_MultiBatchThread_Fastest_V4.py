import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

# === Configuration ===
patch_folder = "ColoredOrnedName_Output2Input"
model_name = "clip-ViT-L-14"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128  # Safe and efficient for RTX 4090
num_workers = 32  # Maximize CPU loading speed

# === Load CLIP model ===
model = SentenceTransformer(model_name, device=device)

# === Read files ===
all_files = []
for fname in sorted(os.listdir(patch_folder)):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    parts = fname.split("___")
    if len(parts) != 2:
        continue
    after_triple_underscore = os.path.splitext(parts[1])[0]
    label_fields = after_triple_underscore.split("_")
    aid = label_fields[0]
    color_info = "_".join(label_fields[1:]) if len(label_fields) >= 2 else ""
    img_path = os.path.join(patch_folder, fname)
    all_files.append((img_path, aid, fname, color_info))

# === Helper to load a single image ===
def load_image(file_info):
    img_path, aid, fname, color_info = file_info
    try:
        with Image.open(img_path) as image:
            return (image.convert("RGB"), aid, fname, color_info)
    except Exception as e:
        print(f"⚠️ Failed loading {fname}: {e}")
        return None

# === Multi-threaded image loading ===
print("⚡ Loading images with CPU...")
images_data = []
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    for result in tqdm(executor.map(load_image, all_files), total=len(all_files)):
        if result is not None:
            images_data.append(result)

# === Batch encode images ===
print("⚡ Encoding images with GPU...")
data = []
for i in tqdm(range(0, len(images_data), batch_size)):
    batch = images_data[i:i+batch_size]
    batch_images = [item[0] for item in batch]
    batch_embeddings = model.encode(
        batch_images,
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=False,
        device=device
    )
    batch_embeddings = batch_embeddings.cpu().numpy()
    for emb, (_, aid, fname, color_info) in zip(batch_embeddings, batch):
        data.append((emb, aid, fname, color_info))

# === Save features to disk ===
if data:
    print("💾 Saving features...")
    features = np.stack([x[0] for x in data])
    aids = np.array([x[1] for x in data])
    filenames = [x[2] for x in data]
    color_infos = np.array([x[3] for x in data])

    np.savez_compressed(  # Use compression for faster IO
        "features_clipL14.npz",
        features=features, aids=aids, filenames=filenames, color_infos=color_infos
    )
    print("✅ Saved: features_clipL14.npz")
else:
    print("❌ No valid patches found!")

