import os
import torch
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np


# === Configuration ===
patch_folder = "augmented_patches_tool_v3"
output_npz = "writing_tool_features_clipL14_all.npz"

model_name = "clip-ViT-L-14"
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256

tool_to_label = {
    "pcl": 0,
    "ink": 1,
}


def parse_label(fname):
    parts = fname.split("___")
    if len(parts) < 2:
        return None

    label_parts = parts[1].split("_")
    if len(label_parts) < 3:
        return None

    tool_code = label_parts[2].lower()
    return tool_to_label.get(tool_code)


def main():
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Patch folder: {patch_folder}")

    model = SentenceTransformer(model_name, device=device)
    model.eval()

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

    if len(files) == 0:
        raise RuntimeError("No valid patches found. Check filename format.")

    all_features = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(files), BATCH_SIZE),
            desc="Extracting CLIP features",
        ):
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

            if len(images) == 0:
                continue

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

    counts = np.bincount(labels_np, minlength=2)

    print(f"\nSaved: {output_npz}")
    print(f"features shape: {features.shape}")
    print(f"labels shape: {labels_np.shape}")
    print(f"filenames: {len(filenames_np)}")
    print(f"class counts: pcl={counts[0]}, ink={counts[1]}")


if __name__ == "__main__":
    main()
