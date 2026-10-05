import os
import re
import json
import shutil
from dataclasses import dataclass
from typing import List, Optional

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


# === Config ===
MODEL_PATH = "writing_tool_classifier_V09_trained_on_all_training_data.pt"
NPZ_PATH = "features_clipL14.npz"

PATCH_FOLDER = "ColoredOrnedName_Output2Input"
JSON_INPUT_FOLDER = "ColoredOrnedJson_Output2Input"
JSON_OUTPUT_FOLDER = "ColoredOrnedTooledJson_Output"
PATCH_OUTPUT_FOLDER = "ColoredOrnedTooledName_Output"

IN_DIM = 768
OUT_DIM = 2
BATCH_SIZE = 128
SAVE_PATCHES = False

# None = auto-detect from MODEL_PATH, e.g. V9
VERSION: Optional[int] = None

L2_NORMALIZE_FEATURES = True

CLASS_NAMES = ["pcl", "ink"]
id_to_label = {0: "pcl", 1: "ink"}


# === Prepare output folders ===
os.makedirs(JSON_OUTPUT_FOLDER, exist_ok=True)
if SAVE_PATCHES:
    os.makedirs(PATCH_OUTPUT_FOLDER, exist_ok=True)


# === Architecture catalogue ===
@dataclass
class ArchSpec:
    id: str
    depth: int
    widths: List[int]
    use_bn: bool
    dropout: float
    activation: str = "relu"


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def get_arch_catalogue() -> List[ArchSpec]:
    return [
        ArchSpec("V01", 0, [], False, 0.0, "-"),
        ArchSpec("V02", 1, [128], False, 0.0, "relu"),
        ArchSpec("V03", 1, [256], False, 0.0, "relu"),
        ArchSpec("V04", 1, [256], False, 0.2, "relu"),
        ArchSpec("V05", 2, [512, 256], False, 0.3, "relu"),
        ArchSpec("V06", 2, [512, 256], False, 0.4, "relu"),
        ArchSpec("V07", 2, [512, 256], True, 0.3, "relu"),
        ArchSpec("V08", 2, [1024, 512], False, 0.4, "relu"),
        ArchSpec("V09", 3, [1024, 512, 256], True, 0.4, "relu"),
        ArchSpec("V10", 4, [1024, 512, 256, 256], False, 0.5, "relu"),
        ArchSpec("V11", 2, [128, 128], False, 0.2, "relu"),
        ArchSpec("V12", 2, [160, 160], False, 0.2, "relu"),
        ArchSpec("V13", 2, [192, 192], False, 0.2, "relu"),
        ArchSpec("V14", 2, [224, 224], False, 0.2, "relu"),
        ArchSpec("V15", 2, [256, 256], False, 0.2, "relu"),
        ArchSpec("V16", 2, [256, 256], True, 0.2, "relu"),
        ArchSpec("V17", 3, [192, 192, 192], True, 0.2, "relu"),
        ArchSpec("V18", 3, [224, 224, 224], False, 0.3, "relu"),
        ArchSpec("V19", 4, [160, 160, 160, 160], True, 0.3, "relu"),
        ArchSpec("V20", 3, [256, 256, 256], True, 0.4, "relu"),
    ]


def parse_version_from_path(path: str) -> Optional[int]:
    m = re.search(r"v(\d+)", os.path.basename(path).lower())
    return int(m.group(1)) if m else None


def find_spec_by_version(version: int) -> ArchSpec:
    target = f"V{version:02d}"
    for spec in get_arch_catalogue():
        if spec.id == target:
            return spec
    raise ValueError(f"Unknown architecture: {target}")


def build_model_from_spec(spec: ArchSpec) -> nn.Module:
    if spec.depth == 0:
        return nn.Sequential(nn.Linear(IN_DIM, OUT_DIM))

    if len(spec.widths) != spec.depth:
        raise ValueError(
            f"{spec.id}: depth={spec.depth} != len(widths)={len(spec.widths)}"
        )

    layers: List[nn.Module] = []
    prev = IN_DIM
    for w in spec.widths:
        layers.append(nn.Linear(prev, w))
        if spec.use_bn:
            layers.append(nn.BatchNorm1d(w))
        layers.append(_act(spec.activation))
        if spec.dropout > 0:
            layers.append(nn.Dropout(spec.dropout))
        prev = w

    layers.append(nn.Linear(prev, OUT_DIM))
    return nn.Sequential(*layers)


# === Resolve version ===
parsed_version = parse_version_from_path(MODEL_PATH)
chosen_version = VERSION if VERSION is not None else parsed_version

if chosen_version is None:
    raise RuntimeError(
        "Cannot infer version from MODEL_PATH. "
        "Please set VERSION manually, e.g. VERSION = 09."
    )

if parsed_version is not None and VERSION is not None and parsed_version != VERSION:
    print(
        f"⚠️ MODEL_PATH suggests V{parsed_version:02d}, "
        f"but VERSION={VERSION}. Using V{VERSION:02d}."
    )

spec = find_spec_by_version(chosen_version)

print(
    f"🔒 Using architecture: {spec.id} | "
    f"depth={spec.depth} | widths={spec.widths} | "
    f"BN={spec.use_bn} | dropout={spec.dropout}"
)


# === Load model ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# One state dict per seed ({"state_dicts": [...]}) or a single state dict;
# softmax probabilities are averaged over the seeds.
ckpt = torch.load(MODEL_PATH, map_location=device)
states = list(ckpt["state_dicts"]) if isinstance(ckpt, dict) and "state_dicts" in ckpt else [ckpt]

models = []
for state in states:
    model = build_model_from_spec(spec).to(device)
    model.load_state_dict(state)
    model.eval()
    models.append(model)

print(f"✅ Loaded checkpoint: {MODEL_PATH} ({len(models)} seeds)")


# === Load features and metadata ===
data = np.load(NPZ_PATH, allow_pickle=True)

X = torch.tensor(data["features"], dtype=torch.float32)

if L2_NORMALIZE_FEATURES:
    X = nn.functional.normalize(X, p=2, dim=1)

X = X.to(device)

filenames = data["filenames"]
aids = data["aids"]

color_infos = data["color_infos"] if "color_infos" in data.files else None
labels = data["labels"] if "labels" in data.files else None

if labels is not None:
    y_true = np.asarray(labels, dtype=int)

print(f"Samples: {len(X)}")
print(f"Feature dim: {X.shape[1]}")


# === Predict ===
preds = []

with torch.no_grad():
    for i in tqdm(range(0, len(X), BATCH_SIZE), desc="Predicting writing_tool"):
        xb = X[i:i + BATCH_SIZE]
        probs = torch.stack([m(xb).softmax(dim=1) for m in models]).mean(dim=0)
        preds.extend(probs.argmax(dim=1).cpu().numpy())

preds = np.asarray(preds, dtype=int)

pred_counts = np.bincount(preds, minlength=OUT_DIM)
print("Predicted class counts:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {pred_counts[i]}")


# === Optional metrics ===
ckpt_stem = os.path.splitext(os.path.basename(MODEL_PATH))[0]

if labels is not None:
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro", zero_division=0)

    print(f"\n🧪 Accuracy: {acc:.4f}")
    print(f"🧪 Macro-F1: {f1:.4f}")

    cm = confusion_matrix(y_true, preds, labels=list(range(OUT_DIM)))

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{spec.id} | Acc={acc:.4f} | F1={f1:.4f}")
    plt.tight_layout()

    cm_path = f"confusion_matrix_tool_{spec.id}_{ckpt_stem}.png"
    plt.savefig(cm_path)
    plt.close()

    print(f"🖼️ Figure saved to: {cm_path}")

    df = pd.DataFrame({
        "filename": filenames,
        "true_id": y_true,
        "true_label": [id_to_label[int(x)] for x in y_true],
        "predicted_id": preds,
        "predicted_label": [id_to_label[int(x)] for x in preds],
    })

    csv_path = f"test_predictions_tool_{spec.id}_{ckpt_stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Predictions saved to: {csv_path}")


# === Build aid -> predicted label mapping ===
aid_to_prediction = {}
filename_to_label = {}

for fname, aid, pred in zip(filenames, aids, preds):
    label = id_to_label[int(pred)]
    aid_to_prediction[int(aid)] = label
    filename_to_label[str(fname)] = label


# === Update JSON files ===
json_files = sorted([
    f for f in os.listdir(JSON_INPUT_FOLDER)
    if f.endswith(".json")
])

updated_files = 0

for json_file in tqdm(json_files, desc="Updating JSON files"):
    input_path = os.path.join(JSON_INPUT_FOLDER, json_file)
    output_path = os.path.join(JSON_OUTPUT_FOLDER, json_file)

    with open(input_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    updated = False

    for ann in json_data.get("annotations", []):
        ann_id = ann.get("id")
        if ann_id in aid_to_prediction:
            ann["writing_tool"] = aid_to_prediction[ann_id]
            updated = True

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    if updated:
        updated_files += 1

print(f"\n✅ Updated JSON files: {updated_files}/{len(json_files)}")


# === Rename and save patches ===
if SAVE_PATCHES:
    missing = 0

    for fname in tqdm(filenames, desc="Saving updated patches"):
        fname = str(fname)
        label = filename_to_label.get(fname)

        src_path = os.path.join(PATCH_FOLDER, fname)

        if not os.path.exists(src_path):
            print(f"⚠️ Missing patch file: {fname}")
            missing += 1
            continue

        base, ext = os.path.splitext(fname)
        new_name = f"{base}_{label}{ext}"
        dst_path = os.path.join(PATCH_OUTPUT_FOLDER, new_name)

        shutil.copy(src_path, dst_path)

    print(f"🎯 All JSONs updated and patches saved. Missing patches: {missing}")
else:
    print("🎯 All JSONs updated. Patch saving skipped because SAVE_PATCHES=False.")
