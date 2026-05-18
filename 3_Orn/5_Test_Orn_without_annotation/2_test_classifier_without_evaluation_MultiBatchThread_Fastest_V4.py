import os
import re
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# ===================== Config =====================
MODEL_PATH = "Orn_classifier_v09_trained_on_all_augmented_training_data.pt"

L2_NORMALIZE_FEATURES = True

NPZ_PATH = "features_clipL14.npz"
PATCH_FOLDER = "ColoredName_CropedPatch_Output2Input"
JSON_INPUT_FOLDER = "ColoredJson_Output2Input"
JSON_OUTPUT_FOLDER = "ColoredOrnedJson_Output"
PATCH_OUTPUT_FOLDER = "ColoredOrnedName_Output"
CM_SAVE_DIR = "confusion_matrices"

IN_DIM = 768
OUT_DIM = 3
BATCH_SIZE = 128
SAVE_PATCHES = True

PREFER_NPZ_PREDICTIONS = False

CLASS_NAMES = ["hor", "vrt", "tlt"]
id_to_label = {i: label for i, label in enumerate(CLASS_NAMES)}

os.makedirs(JSON_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CM_SAVE_DIR, exist_ok=True)

if SAVE_PATCHES:
    os.makedirs(PATCH_OUTPUT_FOLDER, exist_ok=True)


# ===================== Architecture catalogue =====================
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
        ArchSpec("V06", 2, [512, 256], False, 0.3, "relu"),
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


def find_spec_by_version(version: int) -> ArchSpec:
    target_id = f"V{version:02d}"

    for spec in get_arch_catalogue():
        if spec.id == target_id:
            return spec

    raise ValueError(f"Unknown architecture {target_id}")


def build_model_from_spec(
    spec: ArchSpec,
    in_dim: int = IN_DIM,
    out_dim: int = OUT_DIM,
) -> nn.Module:
    if spec.depth == 0:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    layers: List[nn.Module] = []
    hidden = spec.widths[0]

    layers.append(nn.Linear(in_dim, hidden))

    if spec.use_bn:
        layers.append(nn.BatchNorm1d(hidden))

    layers.append(_act(spec.activation))

    if spec.dropout > 0:
        layers.append(nn.Dropout(spec.dropout))

    for _ in range(1, spec.depth):
        layers.append(nn.Linear(hidden, hidden))

        if spec.use_bn:
            layers.append(nn.BatchNorm1d(hidden))

        layers.append(_act(spec.activation))

        if spec.dropout > 0:
            layers.append(nn.Dropout(spec.dropout))

    layers.append(nn.Linear(hidden, out_dim))

    return nn.Sequential(*layers)


def state_dict_has_bn(state: dict) -> bool:
    return any(".running_mean" in k or ".running_var" in k for k in state.keys())


# ===================== Resolve fixed checkpoint =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.isfile(MODEL_PATH):
    raise SystemExit(f"MODEL_PATH does not exist: {MODEL_PATH}")

chosen_path = MODEL_PATH

m = re.search(r"v(\d+)", os.path.basename(chosen_path).lower())

if m is None:
    raise SystemExit(f"Cannot parse version from checkpoint name: {chosen_path}")

chosen_version = int(m.group(1))
spec = find_spec_by_version(chosen_version)

print(f"Using checkpoint: {chosen_path}")
print(
    f"Architecture: {spec.id} | "
    f"depth={spec.depth} | "
    f"widths={spec.widths} | "
    f"BN={spec.use_bn} | "
    f"dropout={spec.dropout} | "
    f"act={spec.activation}"
)


# ===================== Load NPZ =====================
print(f"\nLoading: {NPZ_PATH}")

data = np.load(NPZ_PATH, allow_pickle=True)

features_np = data["features"]
filenames = data["filenames"] if "filenames" in data.files else None
aids = data["aids"] if "aids" in data.files else None
color_infos = data["color_infos"] if "color_infos" in data.files else None
y_true = np.asarray(data["labels"], dtype=np.int64) if "labels" in data.files else None

npz_preds = data["predictions"] if "predictions" in data.files else None
npz_clf_version = str(data["classifier_version"]) if "classifier_version" in data.files else None

print(f"samples={len(features_np)} | feat_dim={features_np.shape[1]}")


# ===================== Predict =====================
reuse_npz = (
    PREFER_NPZ_PREDICTIONS
    and npz_preds is not None
    and npz_clf_version == spec.id
)

if reuse_npz:
    print(f"Reusing predictions from NPZ ({spec.id}).")
    preds = np.asarray(npz_preds, dtype=np.int64)

else:
    model = build_model_from_spec(spec, IN_DIM, OUT_DIM).to(device)

    state = torch.load(chosen_path, map_location=device)

    ckpt_has_bn = state_dict_has_bn(state)

    if ckpt_has_bn != spec.use_bn:
        raise SystemExit(
            f"Architecture mismatch: checkpoint "
            f"{'has' if ckpt_has_bn else 'has no'} BatchNorm, "
            f"but {spec.id} "
            f"{'expects' if spec.use_bn else 'does not use'} BatchNorm."
        )

    model.load_state_dict(state)
    model.eval()

    X_test = torch.tensor(features_np, dtype=torch.float32)

    if L2_NORMALIZE_FEATURES:
        X_test = nn.functional.normalize(X_test, p=2, dim=1)

    X_test = X_test.to(device)

    print(f"Predicting on {len(X_test)} samples...")

    preds_list: List[int] = []

    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            xb = X_test[i:i + BATCH_SIZE]
            out = model(xb).argmax(dim=1).cpu().numpy()
            preds_list.extend(out)

    preds = np.asarray(preds_list, dtype=np.int64)


# ===================== Prediction distribution =====================
class_counts = np.bincount(preds, minlength=len(CLASS_NAMES))

print("\nPredicted class distribution:")
for name, count in zip(CLASS_NAMES, class_counts):
    print(f"{name}: {int(count)}")


# ===================== Metrics =====================
ckpt_stem = os.path.splitext(os.path.basename(chosen_path))[0].lower()

if y_true is not None:
    acc = accuracy_score(y_true, preds)
    f1m = f1_score(y_true, preds, average="macro", zero_division=0)
    f1_pc = f1_score(
        y_true,
        preds,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro-F1: {f1m:.4f}")
    print(
        "F1 per class: "
        + " ".join(
            f"{name}={score:.4f}"
            for name, score in zip(CLASS_NAMES, f1_pc)
        )
    )

    cm = confusion_matrix(y_true, preds, labels=list(range(len(CLASS_NAMES))))

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
    plt.title(f"{spec.id} | Acc={acc:.4f} | Macro-F1={f1m:.4f}")
    plt.tight_layout()

    cm_path = os.path.join(CM_SAVE_DIR, f"confmat_{ckpt_stem}.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"Confusion matrix saved to: {cm_path}")


# ===================== Save predictions CSV =====================
if filenames is not None:
    out_csv = f"test_predictions_{ckpt_stem}.csv"

    df_rows = {
        "filename": filenames,
        "predicted_id": preds,
        "predicted_label": [id_to_label[i] for i in preds],
    }

    if aids is not None:
        df_rows["aid"] = aids

    if color_infos is not None:
        df_rows["color_info"] = color_infos

    if y_true is not None:
        df_rows["true_id"] = y_true
        df_rows["true_label"] = [id_to_label[i] for i in y_true]

    pd.DataFrame(df_rows).to_csv(out_csv, index=False)

    print(f"Predictions saved to: {out_csv}")


# ===================== Build lookups =====================
aid_to_prediction = {}

if aids is not None:
    try:
        aids_int = [int(a) for a in aids]
    except Exception:
        aids_int = list(aids)

    aid_to_prediction = {
        aid: id_to_label[pred]
        for aid, pred in zip(aids_int, preds)
    }

filename_to_label = {}

if filenames is not None:
    filename_to_label = {
        fname: id_to_label[pred]
        for fname, pred in zip(filenames, preds)
    }


# ===================== Update JSON files =====================
json_files = (
    sorted([f for f in os.listdir(JSON_INPUT_FOLDER) if f.endswith(".json")])
    if os.path.isdir(JSON_INPUT_FOLDER)
    else []
)

print(f"\nUpdating JSON files to: {JSON_OUTPUT_FOLDER}")


def process_json(json_file):
    try:
        path_in = os.path.join(JSON_INPUT_FOLDER, json_file)
        path_out = os.path.join(JSON_OUTPUT_FOLDER, json_file)

        with open(path_in, "r", encoding="utf-8") as f:
            data_json = json.load(f)

        updated = False

        if "annotations" in data_json and aid_to_prediction:
            for ann in data_json["annotations"]:
                aid = ann.get("id")

                if aid in aid_to_prediction:
                    ann["orientation"] = aid_to_prediction[aid]
                    updated = True

        with open(path_out, "w", encoding="utf-8") as f:
            json.dump(data_json, f, indent=2, ensure_ascii=False)

        return f"Updated {json_file}" if updated else f"No updates in {json_file}"

    except Exception as e:
        return f"Error in {json_file}: {e}"


if json_files:
    with ThreadPoolExecutor() as executor:
        for result in tqdm(
            executor.map(process_json, json_files),
            total=len(json_files),
        ):
            print(result)
else:
    print("No JSON files found.")


# ===================== Rename & save patches =====================
if SAVE_PATCHES and filenames is not None:
    print("\nSaving renamed patches...")

    def save_patch(fname):
        try:
            if fname not in filename_to_label:
                return f"Missing prediction for: {fname}"

            label = filename_to_label[fname]

            src = os.path.join(PATCH_FOLDER, fname)

            if not os.path.exists(src):
                return f"Missing file: {fname}"

            new_name = f"{os.path.splitext(fname)[0]}_{label}.png"
            dst = os.path.join(PATCH_OUTPUT_FOLDER, new_name)

            shutil.copy(src, dst)

            return f"Saved {new_name}"

        except Exception as e:
            return f"Error: {fname} – {e}"

    with ThreadPoolExecutor() as executor:
        for _ in tqdm(
            executor.map(save_patch, filenames),
            total=len(filenames),
        ):
            pass

    print(f"\nAll JSONs and patches updated using {spec.id}")

else:
    print(f"\nAll JSONs updated. Patch saving skipped. Using {spec.id}")
