import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import numpy as np

# === Configuration ===
patch_folder = "ColoredName_CropedPatch_Output2Input"
output_npz = "features_clipL14.npz"
model_name = "clip-ViT-L-14"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
num_workers = 32

# The letterbox setting is read from the metadata of the trained model
MODEL_PATH = "Orn_classifier_v16_trained_on_all_augmented_training_data.pt"
META_PATH = MODEL_PATH.replace(".pt", ".meta.json")

if not os.path.isfile(META_PATH):
    raise SystemExit(
        f"META_PATH does not exist: {META_PATH}\n"
        "Copy it from ../3_Train_augmented_data/ together with the .pt file."
    )
with open(META_PATH, "r", encoding="utf-8") as f:
    use_letterbox = bool(json.load(f)["use_letterbox_square"])


def letterbox_square_pil(image, fill=255):
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (fill, fill, fill))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def load_image(path):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    return letterbox_square_pil(rgb) if use_letterbox else rgb


# === Read files ===
filenames, aids, color_infos = [], [], []
for fname in sorted(os.listdir(patch_folder)):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    parts = fname.split("___")
    if len(parts) != 2:
        continue
    label_fields = os.path.splitext(parts[1])[0].split("_")
    filenames.append(fname)
    aids.append(label_fields[0])
    color_infos.append("_".join(label_fields[1:]) if len(label_fields) >= 3 else "")

print(f"Device: {device}")
print(f"Model: {model_name}")
print(f"Patch folder: {patch_folder}  patches: {len(filenames)}  letterbox={use_letterbox}")
if not filenames:
    raise SystemExit("No valid patches found!")

# === Load CLIP model ===
model = SentenceTransformer(model_name, device=device)

# === Batch encode (multi-threaded image loading) ===
all_features = []
with ThreadPoolExecutor(max_workers=num_workers) as pool:
    for start in tqdm(range(0, len(filenames), batch_size), desc="Extracting CLIP features"):
        batch_files = filenames[start:start + batch_size]
        images = list(pool.map(lambda f: load_image(os.path.join(patch_folder, f)), batch_files))
        with torch.no_grad():
            embeddings = model.encode(
                images,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
        all_features.append(embeddings.astype(np.float32))

# === Save features ===
features = np.concatenate(all_features, axis=0)
np.savez_compressed(
    output_npz,
    features=features,
    aids=np.asarray(aids),
    filenames=filenames,
    color_infos=np.asarray(color_infos),
    letterbox=np.asarray(use_letterbox),
)
print(f"Saved: {output_npz}  features={features.shape}")
