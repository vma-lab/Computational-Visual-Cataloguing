import os
import torch
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np


# === Configuration ===
PATCH_FOLDER = "text_patches_test_20"
OUTPUT_NPZ = "orientation_features_clipL14_all.npz"

MODEL_NAME = "clip-ViT-L-14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256

CLASS_MAP = {
    "hor": 0,
    "vrt": 1,
    "tlt": 2,
}


def parse_label(fname: str):
    parts = fname.split("___")
    if len(parts) < 2:
        return None

    label_parts = parts[-1].split("_")
    if len(label_parts) < 4:
        return None

    label_code = label_parts[1].lower()
    return CLASS_MAP.get(label_code)


def collect_files(patch_folder: str):
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

    return files, labels


def load_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Skip {path}: {e}")
        return None


def main():
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Patch folder: {PATCH_FOLDER}")

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    files, labels = collect_files(PATCH_FOLDER)

    print(f"Valid images: {len(files)}")

    if len(files) == 0:
        raise RuntimeError("No valid images found. Check filename format and folder path.")

    all_features = []
    all_labels = []
    all_filenames = []

    for start in tqdm(range(0, len(files), BATCH_SIZE), desc="Extracting CLIP features"):
        batch_files = files[start:start + BATCH_SIZE]
        batch_labels = labels[start:start + BATCH_SIZE]

        images = []
        valid_files = []
        valid_labels = []

        for fname, label in zip(batch_files, batch_labels):
            img_path = os.path.join(PATCH_FOLDER, fname)
            img = load_image(img_path)

            if img is None:
                continue

            images.append(img)
            valid_files.append(fname)
            valid_labels.append(label)

        if not images:
            continue

        with torch.no_grad():
            embeddings = model.encode(
                images,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )

        all_features.append(embeddings.astype(np.float32))
        all_labels.extend(valid_labels)
        all_filenames.extend(valid_files)

    features = np.concatenate(all_features, axis=0)
    labels_np = np.array(all_labels, dtype=np.int64)
    filenames_np = np.array(all_filenames)

    np.savez(
        OUTPUT_NPZ,
        features=features,
        labels=labels_np,
        filenames=filenames_np,
    )

    print(f"\nSaved: {OUTPUT_NPZ}")
    print(f"features shape: {features.shape}")
    print(f"labels shape: {labels_np.shape}")
    print(f"filenames: {len(filenames_np)}")

    counts = np.bincount(labels_np, minlength=len(CLASS_MAP))
    print(f"class counts: hor={counts[0]}, vrt={counts[1]}, tlt={counts[2]}")


if __name__ == "__main__":
    main()
