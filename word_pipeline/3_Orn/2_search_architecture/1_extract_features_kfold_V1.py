import os
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


def main():
    print(f"Device: {device}")
    print(f"Model: {model_name}")

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
            img_path = os.path.join(patch_folder, fname)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Skip {fname}: {e}")
                continue

            images.append(image)
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
        output_npz,
        features=features,
        labels=labels_np,
        filenames=filenames_np,
    )

    print(f"\nSaved: {output_npz}")
    print(f"features shape: {features.shape}")
    print(f"labels shape: {labels_np.shape}")
    print(f"filenames: {len(filenames_np)}")

    counts = np.bincount(labels_np, minlength=3)
    print(f"class counts: hor={counts[0]}, vrt={counts[1]}, tlt={counts[2]}")


if __name__ == "__main__":
    main()
