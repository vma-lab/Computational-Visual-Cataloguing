import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from joblib import Parallel, delayed
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============== Config ==============
EPOCHS = 30
BATCH_SIZE = 256
CLIP_DIM = 768
GEO_DIM = 20
OUT_DIM = 3
SEED = 42

DATA_PATH = "orientation_features_clipL14_all.npz"
PATCH_FOLDER = "text_patches_augmented_betterv3_80"
CLASS_NAMES = ["hor", "vrt", "tlt"]

# Final fixed architecture (best RankScore in
# ../2_search_architecture/orientation_mlp_arch_search_V01-V20_rankscore_report.txt)
BEST_VERSION = 16

# Same input and training recipe as in 2_search_architecture/2_train_classifier_V11.py
USE_LETTERBOX_SQUARE = True
USE_GEO_BRANCH = True
GEO_HIDDEN = 64

USE_CLASS_WEIGHTED_CE = True
L2_NORMALIZE_FEATURES = True
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
USE_COSINE_LR = True

USE_MIXUP = True
MIXUP_ALPHA = 0.2

# Seed ensemble (probabilities averaged), as for each fold in the search
N_SEEDS = 3

torch.manual_seed(SEED)
np.random.seed(SEED)


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


def find_spec_by_version(version: int) -> ArchSpec:
    target_id = f"V{version:02d}"
    for spec in get_arch_catalogue():
        if spec.id == target_id:
            return spec
    raise ValueError(f"Unknown architecture {target_id}")


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
                 geo_dim: int = GEO_DIM, geo_hidden: int = GEO_HIDDEN):
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


def build_model_from_spec(spec: ArchSpec, out_dim: int = OUT_DIM) -> nn.Module:
    if USE_GEO_BRANCH:
        return DualBranchOrn(spec, out_dim=out_dim)
    layers, prev = _hidden_layers(spec, CLIP_DIM)
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def describe_model_layers(model: nn.Module) -> str:
    parts: List[str] = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            parts.append(f"Linear({m.in_features},{m.out_features})")
        elif isinstance(m, nn.BatchNorm1d):
            parts.append(f"BN({m.num_features})")
        elif isinstance(m, nn.Dropout):
            parts.append(f"Dropout({m.p})")
        elif isinstance(m, (nn.ReLU, nn.SiLU, nn.GELU)):
            parts.append(m.__class__.__name__)
    return " -> ".join(parts)


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


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def build_inputs(clip: np.ndarray, geo: Optional[np.ndarray],
                 geo_mean: Optional[np.ndarray], geo_std: Optional[np.ndarray]) -> np.ndarray:
    x = l2_normalize(clip) if L2_NORMALIZE_FEATURES else clip.astype(np.float32)
    if geo is None:
        return x
    geo_z = ((geo - geo_mean) / np.maximum(geo_std, 1e-6)).astype(np.float32)
    return np.hstack([x, geo_z]).astype(np.float32)


# ===================== Training =====================
def compute_class_weights(y: np.ndarray, n_classes: int, device: str) -> Optional[torch.Tensor]:
    if not USE_CLASS_WEIGHTED_CE:
        return None
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = counts.sum() / (n_classes * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _loss(model, xb, yb, class_weights):
    if USE_MIXUP and MIXUP_ALPHA > 0:
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        perm = torch.randperm(xb.size(0), device=xb.device)
        logits = model(lam * xb + (1.0 - lam) * xb[perm])
        ce = nn.functional.cross_entropy
        return lam * ce(logits, yb, weight=class_weights) + (1.0 - lam) * ce(logits, yb[perm], weight=class_weights)
    logits = model(xb)
    return nn.functional.cross_entropy(logits, yb, weight=class_weights)


def train_final_model(spec: ArchSpec, X: np.ndarray, y: np.ndarray, seed: int, device: str) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model_from_spec(spec).to(device)
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) if USE_COSINE_LR else None
    class_weights = compute_class_weights(y, OUT_DIM, device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = _loss(model, xb, yb, class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        print(f"Seed {seed} | Epoch {epoch:03d}/{EPOCHS} | loss={total_loss / max(1, n_batches):.6f}")

    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print("Loading data...")

    data = np.load(DATA_PATH, allow_pickle=True)
    if "letterbox" not in data.files or bool(data["letterbox"]) != USE_LETTERBOX_SQUARE:
        raise RuntimeError(
            f"'{DATA_PATH}' was not extracted with letterbox={USE_LETTERBOX_SQUARE}; "
            "re-run 1_extract_features_kfold_V1.py."
        )

    clip = data["features"].astype(np.float32)
    y_np = np.asarray(data["labels"], dtype=np.int64)
    filenames = [str(f) for f in data["filenames"]]
    class_counts = np.bincount(y_np, minlength=OUT_DIM).tolist()

    geo = geo_mean = geo_std = None
    if USE_GEO_BRANCH:
        print(f"Extracting geometric features from: {PATCH_FOLDER}")
        geo = extract_geo(PATCH_FOLDER, filenames)
        geo_mean = geo.mean(axis=0).astype(np.float32)
        geo_std = np.maximum(geo.std(axis=0), 1e-6).astype(np.float32)
    X = build_inputs(clip, geo, geo_mean, geo_std)

    print(f"Samples: {len(X)}")
    print(f"Class names: {CLASS_NAMES}")
    print(f"Class counts: {class_counts}")
    print(f"Input dim: {X.shape[1]} (CLIP letterbox={USE_LETTERBOX_SQUARE}, geometric branch={USE_GEO_BRANCH})")

    spec = find_spec_by_version(BEST_VERSION)

    print(f"\nFixed final architecture: {spec.id}")
    print(
        f"Spec: depth={spec.depth}, widths={spec.widths}, "
        f"use_bn={spec.use_bn}, dropout={spec.dropout}, activation={spec.activation}"
    )

    print(f"\nTraining on ALL augmented training data. No fold / no CV. Seeds: {N_SEEDS}")
    seeds = [SEED + 7919 * s for s in range(N_SEEDS)]
    models = [train_final_model(spec, X, y_np, seed, device) for seed in seeds]
    print(f"Model: {describe_model_layers(models[0])}")

    model_path = f"Orn_classifier_v{BEST_VERSION:02d}_trained_on_all_augmented_training_data.pt"
    torch.save({"state_dicts": [m.state_dict() for m in models]}, model_path)

    meta = {
        "model_path": model_path,
        "architecture_id": spec.id,
        "architecture": {
            "depth": spec.depth,
            "widths": spec.widths,
            "use_bn": spec.use_bn,
            "dropout": spec.dropout,
            "activation": spec.activation,
        },
        "use_letterbox_square": USE_LETTERBOX_SQUARE,
        "use_geo_branch": USE_GEO_BRANCH,
        "geo_hidden": GEO_HIDDEN,
        "geo_mean": None if geo_mean is None else geo_mean.tolist(),
        "geo_std": None if geo_std is None else geo_std.tolist(),
        "seeds": seeds,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "input_dim": int(X.shape[1]),
        "output_dim": OUT_DIM,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "use_cosine_lr": USE_COSINE_LR,
        "use_class_weighted_ce": USE_CLASS_WEIGHTED_CE,
        "use_mixup": USE_MIXUP,
        "mixup_alpha": MIXUP_ALPHA,
        "l2_normalize_features": L2_NORMALIZE_FEATURES,
        "data_path": DATA_PATH,
        "patch_folder": PATCH_FOLDER,
        "class_names": CLASS_NAMES,
        "class_counts": class_counts,
    }

    meta_path = model_path.replace(".pt", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved final model: {model_path} ({len(models)} seeds)")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
