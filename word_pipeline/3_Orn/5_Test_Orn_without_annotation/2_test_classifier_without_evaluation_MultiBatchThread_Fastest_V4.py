import os
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from joblib import Parallel, delayed
from torch import nn
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# ===================== Config =====================
MODEL_PATH = "Orn_classifier_v16_trained_on_all_augmented_training_data.pt"
META_PATH = MODEL_PATH.replace(".pt", ".meta.json")

L2_NORMALIZE_FEATURES = True

NPZ_PATH = "features_clipL14.npz"
PATCH_FOLDER = "ColoredName_CropedPatch_Output2Input"
JSON_INPUT_FOLDER = "ColoredJson_Output2Input"
JSON_OUTPUT_FOLDER = "ColoredOrnedJson_Output"
PATCH_OUTPUT_FOLDER = "ColoredOrnedName_Output"
CM_SAVE_DIR = "confusion_matrices"

CLIP_DIM = 768
GEO_DIM = 20
OUT_DIM = 3
BATCH_SIZE = 512
SAVE_PATCHES = True

CLASS_NAMES = ["hor", "vrt", "tlt"]
id_to_label = {i: label for i, label in enumerate(CLASS_NAMES)}

os.makedirs(JSON_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CM_SAVE_DIR, exist_ok=True)

if SAVE_PATCHES:
    os.makedirs(PATCH_OUTPUT_FOLDER, exist_ok=True)


# ===================== Architecture =====================
@dataclass
class ArchSpec:
    id: str
    depth: int
    widths: List[int]
    use_bn: bool
    dropout: float
    activation: str = "relu"


def _act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name in ("relu", "-"):
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


def find_spec_by_id(arch_id: str) -> ArchSpec:
    for spec in get_arch_catalogue():
        if spec.id == arch_id:
            return spec
    raise ValueError(f"Unknown architecture {arch_id}")


def _hidden_layers(spec: ArchSpec, in_dim: int) -> Tuple[List[nn.Module], int]:
    if len(spec.widths) != spec.depth:
        raise ValueError(f"{spec.id}: depth={spec.depth} != len(widths)={len(spec.widths)}")
    layers: List[nn.Module] = []
    prev = in_dim
    for w in spec.widths:
        layers.append(nn.Linear(prev, w))
        if spec.use_bn:
            layers.append(nn.BatchNorm1d(w))
        layers.append(_act(spec.activation))
        if spec.dropout > 0:
            layers.append(nn.Dropout(spec.dropout))
        prev = w
    return layers, prev


class DualBranchOrn(nn.Module):
    """Searched MLP on the CLIP embedding (without its output layer) and a small
    MLP on the geometric features, concatenated and mapped to the classes."""

    def __init__(self, spec: ArchSpec, out_dim: int = OUT_DIM, clip_dim: int = CLIP_DIM,
                 geo_dim: int = GEO_DIM, geo_hidden: int = 64):
        super().__init__()
        self.clip_dim = clip_dim
        self.geo_dim = geo_dim
        layers, clip_out = _hidden_layers(spec, clip_dim)
        self.clip_backbone = nn.Sequential(*layers) if layers else nn.Identity()
        drop = spec.dropout if spec.dropout > 0 else 0.2
        self.geo_net = nn.Sequential(
            nn.Linear(geo_dim, geo_hidden),
            nn.BatchNorm1d(geo_hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(geo_hidden, geo_hidden),
            nn.BatchNorm1d(geo_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(clip_out + geo_hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clip = x[:, : self.clip_dim]
        geo = x[:, self.clip_dim : self.clip_dim + self.geo_dim]
        return self.head(torch.cat([self.clip_backbone(clip), self.geo_net(geo)], dim=1))


def build_model_from_spec(spec: ArchSpec, use_geo: bool, geo_hidden: int = 64) -> nn.Module:
    if use_geo:
        return DualBranchOrn(spec, geo_hidden=geo_hidden)
    layers, prev = _hidden_layers(spec, CLIP_DIM)
    layers.append(nn.Linear(prev, OUT_DIM))
    return nn.Sequential(*layers)


# ===================== Geometric features =====================
def _axis_feats(ys, xs):
    if len(xs) < 3:
        return [0.0, 0.0, 0.0, 0.0]
    pts = np.stack([xs, ys], 1).astype(np.float64)
    pts -= pts.mean(0)
    cov = pts.T @ pts / len(pts)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, 1]
    ang = np.arctan2(major[1], major[0])
    elong = 1.0 - np.sqrt(max(evals[0], 1e-9) / max(evals[1], 1e-9))
    return [np.cos(2 * ang), np.sin(2 * ang), abs(np.sin(2 * ang)), elong]


def geo_features(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        return np.zeros(GEO_DIM, dtype=np.float32)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # patches are cropped with pure white outside the polygon
    mask = (img.min(axis=2) < 250).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(mask)

    feats = [np.log(w / h), np.log(w * h + 1), mask.mean()]

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), ra = cv2.minAreaRect(c)
        if rw < rh:
            rw, rh, ra = rh, rw, ra + 90
        t = np.deg2rad(ra)
        rect_fill = cv2.contourArea(c) / max(rw * rh, 1)
        feats += [
            np.log((rw + 1) / (rh + 1)),
            np.cos(2 * t),
            np.sin(2 * t),
            abs(np.sin(2 * t)),
            rect_fill,
        ]
    else:
        feats += [0.0] * 5

    feats += _axis_feats(ys, xs)

    inside = gray[mask > 0]
    if inside.size > 10:
        thr, _ = cv2.threshold(
            inside.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        ink = (gray < thr) & (mask > 0)
        iy, ix = np.nonzero(ink)
        feats += _axis_feats(iy, ix)
        feats += [ink.sum() / max(mask.sum(), 1)]
        if len(ix) > 3:
            feats += [
                ix.std() / w,
                iy.std() / h,
                np.log((ix.std() + 1) / (iy.std() + 1)),
            ]
        else:
            feats += [0.0, 0.0, 0.0]
    else:
        feats += [0.0] * 8
    return np.asarray(feats, dtype=np.float32)


def extract_geo(folder: str, filenames: List[str], n_jobs: int = 16) -> np.ndarray:
    feats = Parallel(n_jobs=n_jobs)(
        delayed(geo_features)(os.path.join(folder, str(f))) for f in filenames
    )
    return np.stack(feats).astype(np.float32)


def build_inputs(clip: np.ndarray, geo: Optional[np.ndarray],
                 geo_mean: Optional[np.ndarray], geo_std: Optional[np.ndarray]) -> np.ndarray:
    x = clip.astype(np.float32)
    if L2_NORMALIZE_FEATURES:
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    if geo is None:
        return x
    geo_z = ((geo - geo_mean) / np.maximum(geo_std, 1e-6)).astype(np.float32)
    return np.hstack([x, geo_z]).astype(np.float32)


@torch.no_grad()
def predict_proba(models: List[nn.Module], X: np.ndarray, device: str) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float32)
    total: Optional[np.ndarray] = None
    for m in models:
        m.eval()
        out = []
        for i in range(0, len(Xt), BATCH_SIZE):
            out.append(torch.softmax(m(Xt[i:i + BATCH_SIZE].to(device)), dim=1).cpu().numpy())
        p = np.concatenate(out, axis=0)
        total = p if total is None else total + p
    return total / len(models)


# ===================== Load model (seed ensemble) =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

for _path, _name in ((MODEL_PATH, "MODEL_PATH"), (META_PATH, "META_PATH"), (NPZ_PATH, "NPZ_PATH")):
    if not os.path.isfile(_path):
        raise SystemExit(f"{_name} does not exist: {_path}")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
use_letterbox = bool(meta["use_letterbox_square"])
use_geo = bool(meta["use_geo_branch"])
spec = find_spec_by_id(meta["architecture_id"])
model_id = spec.id

ckpt = torch.load(MODEL_PATH, map_location=device)
models = []
for state in ckpt["state_dicts"]:
    model = build_model_from_spec(spec, use_geo, meta.get("geo_hidden", 64)).to(device)
    model.load_state_dict(state)
    model.eval()
    models.append(model)

print(f"Using checkpoint: {MODEL_PATH}")
print(
    f"Architecture: {spec.id} | depth={spec.depth} | widths={spec.widths} | "
    f"BN={spec.use_bn} | dropout={spec.dropout} | act={spec.activation}"
)
print(f"Input: CLIP letterbox={use_letterbox} | geometric branch={use_geo} | seeds={len(models)}")


# ===================== Load NPZ =====================
print(f"\nLoading: {NPZ_PATH}")

data = np.load(NPZ_PATH, allow_pickle=True)
if "letterbox" not in data.files or bool(data["letterbox"]) != use_letterbox:
    raise SystemExit(f"{NPZ_PATH} was not extracted with letterbox={use_letterbox}; "
                     "re-run 1_extract_features_testing_MultiBatchThread_Fastest_V4.py")

features_np = data["features"].astype(np.float32)
filenames = [str(f) for f in data["filenames"]] if "filenames" in data.files else None
aids = data["aids"] if "aids" in data.files else None
color_infos = data["color_infos"] if "color_infos" in data.files else None
y_true = np.asarray(data["labels"], dtype=np.int64) if "labels" in data.files else None

print(f"samples={len(features_np)} | feat_dim={features_np.shape[1]}")


# ===================== Predict =====================
geo = geo_mean = geo_std = None
if use_geo:
    if filenames is None:
        raise SystemExit("NPZ has no 'filenames'; cannot compute geometric features.")
    print(f"Extracting geometric features from: {PATCH_FOLDER}")
    geo = extract_geo(PATCH_FOLDER, filenames)
    geo_mean = np.asarray(meta["geo_mean"], dtype=np.float32)
    geo_std = np.asarray(meta["geo_std"], dtype=np.float32)

X_np = build_inputs(features_np, geo, geo_mean, geo_std)
print(f"Predicting on {len(X_np)} samples...")
preds = predict_proba(models, X_np, device).argmax(axis=1).astype(np.int64)


# ===================== Prediction distribution =====================
class_counts = np.bincount(preds, minlength=len(CLASS_NAMES))

print("\nPredicted class distribution:")
for name, count in zip(CLASS_NAMES, class_counts):
    print(f"{name}: {int(count)}")


# ===================== Metrics =====================
ckpt_stem = os.path.splitext(os.path.basename(MODEL_PATH))[0].lower()

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
    plt.title(f"{model_id} | Acc={acc:.4f} | Macro-F1={f1m:.4f}")
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

    print(f"\nAll JSONs and patches updated using {model_id}")

else:
    print(f"\nAll JSONs updated. Patch saving skipped. Using {model_id}")
