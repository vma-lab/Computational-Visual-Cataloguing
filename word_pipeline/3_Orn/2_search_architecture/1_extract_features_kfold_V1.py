import os
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np


# === Configuration ===
patch_folder = "text_patches_augmented_betterv3_80"
output_npz = "orientation_features_clipL14_all.npz"

model_name = "clip-ViT-L-14"
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
NUM_WORKERS = 16

# Pad each patch to a white square before the CLIP preprocessing
# (resize + centre crop), so the whole word stays in the image.
USE_LETTERBOX_SQUARE = True

short_to_full = {"hor": 0, "vrt": 1, "tlt": 2}


def parse_label(fname):
    parts = fname.split("___")
    if len(parts) < 2:
        return None

    label_parts = parts[-1].split("_")
    if len(label_parts) < 4:
        return None

    label_code = label_parts[1].lower()
    return short_to_full.get(label_code)


def letterbox_square_pil(image, fill=255):
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (fill, fill, fill))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def load_image(path):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    return letterbox_square_pil(rgb) if USE_LETTERBOX_SQUARE else rgb


def main():
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Letterbox square: {USE_LETTERBOX_SQUARE}")

    model = SentenceTransformer(model_name, device=device)

    files = []
    labels = []

    for fname in sorted(os.listdir(patch_folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        label = parse_label(fname)
        if label is None:
            continue

        files.append(fname)
        labels.append(label)

    print(f"Valid images: {len(files)}")
    if not files:
        raise RuntimeError("No valid images found. Check filename format and folder path.")

    all_features = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for start in tqdm(range(0, len(files), BATCH_SIZE), desc="Extracting CLIP features"):
            batch_files = files[start:start + BATCH_SIZE]
            images = list(pool.map(lambda f: load_image(os.path.join(patch_folder, f)), batch_files))

            with torch.no_grad():
                embeddings = model.encode(
                    images,
                    batch_size=BATCH_SIZE,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                )

            all_features.append(embeddings.astype(np.float32))

    features = np.concatenate(all_features, axis=0)
    labels_np = np.array(labels, dtype=np.int64)
    filenames_np = np.array(files)

    np.savez(
        output_npz,
        features=features,
        labels=labels_np,
        filenames=filenames_np,
        letterbox=np.asarray(USE_LETTERBOX_SQUARE),
    )

    print(f"\nSaved: {output_npz}")
    print(f"features shape: {features.shape}")
    print(f"labels shape: {labels_np.shape}")
    print(f"filenames: {len(filenames_np)}")

    counts = np.bincount(labels_np, minlength=3)
    print(f"class counts: hor={counts[0]}, vrt={counts[1]}, tlt={counts[2]}")


if __name__ == "__main__":
    main()
