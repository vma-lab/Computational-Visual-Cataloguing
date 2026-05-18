import os
import torch
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import numpy as np

# === Configuration ===
patch_folder = "ColoredName_CropedPatch_Output2Input"
model_name = "clip-ViT-L-14"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128  # Increased batch size (4090 VRAM can handle it easily)
num_workers = 32  # Explicitly control workers for thread pool

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
    image_base = parts[0]
    after_triple_underscore = parts[1]
    after_triple_underscore = os.path.splitext(after_triple_underscore)[0]
    label_fields = after_triple_underscore.split("_")
    aid = label_fields[0]
    color_info = "_".join(label_fields[1:]) if len(label_fields) >= 3 else ""
    img_path = os.path.join(patch_folder, fname)
    all_files.append((img_path, aid, fname, color_info))

# === Helper to load images ===
def load_image(file_info):
    img_path, aid, fname, color_info = file_info
    try:
        with Image.open(img_path) as image:  # Use 'with' to auto-close file faster
            return (image.convert("RGB"), aid, fname, color_info)
    except Exception as e:
        print(f"⚠️ Failed processing {fname}: {e}")
        return None

# === Multi-threaded image loading ===
print("⚡ Loading images...")
images_data = []
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    for result in tqdm(executor.map(load_image, all_files), total=len(all_files)):
        if result is not None:
            images_data.append(result)

# === Batch encode ===
print("⚡ Encoding images...")
data = []
for i in tqdm(range(0, len(images_data), batch_size)):
    batch = images_data[i:i+batch_size]
    batch_images = [item[0] for item in batch]
    batch_embeddings = model.encode(
        batch_images, 
        convert_to_tensor=True, 
        batch_size=batch_size, 
        show_progress_bar=False, 
        device=device  # explicitly confirm device usage
    )
    batch_embeddings = batch_embeddings.cpu().numpy()
    for emb, (_, aid, fname, color_info) in zip(batch_embeddings, batch):
        data.append((emb, aid, fname, color_info))

# === Save features ===
if data:
    print("💾 Saving features...")
    features = np.stack([x[0] for x in data])
    aids = np.array([x[1] for x in data])
    filenames = [x[2] for x in data]
    color_infos = np.array([x[3] for x in data])

    np.savez_compressed(  # Use compressed npz (smaller, faster disk IO)
        "features_clipL14.npz",
        features=features, aids=aids, filenames=filenames, color_infos=color_infos
    )
    print("✅ Saved: features_clipL14.npz")
else:
    print("❌ No valid patches found!")

