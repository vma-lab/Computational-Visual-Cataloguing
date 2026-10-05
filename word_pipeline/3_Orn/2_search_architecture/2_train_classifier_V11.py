import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============== Config ==============
FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 256
CLIP_DIM = 768
GEO_DIM = 20
OUT_DIM = 3

CM_SAVE_DIR = "confusion_matrices_orientation"
DATA_PATH = "orientation_features_clipL14_all.npz"
PATCH_FOLDER = "text_patches_augmented_betterv3_80"
REPORT_PATH = "orientation_mlp_arch_search_V01-V20_rankscore_report.txt"
CLASS_NAMES = ["hor", "vrt", "tlt"]
SEED = 42

MAX_VERSIONS = 20

# Input: L2-normalised CLIP ViT-L/14 embedding of the letterboxed patch
# (see 1_extract_features_kfold_V1.py) plus 20 geometric features of the word
# polygon and ink, standardised with training-fold statistics.
USE_LETTERBOX_SQUARE = True
USE_GEO_BRANCH = True
GEO_HIDDEN = 64

USE_CLASS_WEIGHTED_CE = True
L2_NORMALIZE_FEATURES = True
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
USE_COSINE_LR = True

# Fixed number of epochs; validation folds are not used during training.
EARLY_STOPPING = False

USE_MIXUP = True
MIXUP_ALPHA = 0.2

N_SEEDS_PER_FOLD = 3
PREDICT_BATCH_SIZE = 512

# Validation folds are scored on original (non-augmented) patches only
SCORE_ORIGINAL_PATCHES_ONLY = True

# Stability-aware ranking:
# rank_score = avg_val_f1_macro - STD_PENALTY_LAMBDA * std_val_f1_macro
RANK_BY = "rank_score"
STD_PENALTY_LAMBDA = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CM_SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ===================== Architecture =====================
@dataclass
class ArchSpec:
    id: str
    kind: str
    depth: int
    widths: List[int]
    use_bn: bool
    dropout: float
    activation: str
    notes: str = ""


def _act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name in ("relu", "-"):
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def get_handcrafted_specs() -> List[ArchSpec]:
    return [
        ArchSpec("V01", "handcrafted", 0, [], False, 0.0, "-", "Linear baseline"),
        ArchSpec("V02", "handcrafted", 1, [128], False, 0.0, "relu"),
        ArchSpec("V03", "handcrafted", 1, [256], False, 0.0, "relu"),
        ArchSpec("V04", "handcrafted", 1, [256], False, 0.2, "relu"),
        ArchSpec("V05", "handcrafted", 2, [512, 256], False, 0.3, "relu"),
        ArchSpec("V06", "handcrafted", 2, [512, 256], False, 0.4, "relu"),
        ArchSpec("V07", "handcrafted", 2, [512, 256], True, 0.3, "relu"),
        ArchSpec("V08", "handcrafted", 2, [1024, 512], False, 0.4, "relu"),
        ArchSpec("V09", "handcrafted", 3, [1024, 512, 256], True, 0.4, "relu"),
        ArchSpec("V10", "handcrafted", 4, [1024, 512, 256, 256], False, 0.5, "relu"),
    ]


def get_parametric_specs_v11_v20_diverse() -> List[ArchSpec]:
    specs = []
    widths = [128, 160, 192, 224, 256]

    for i, w in enumerate(widths, start=11):
        specs.append(ArchSpec(f"V{i:02d}", "grid", 2, [w, w], False, 0.2, "relu"))

    specs.append(ArchSpec("V16", "grid", 2, [256, 256], True, 0.2, "relu"))
    specs.append(ArchSpec("V17", "grid", 3, [192, 192, 192], True, 0.2, "relu"))
    specs.append(ArchSpec("V18", "grid", 3, [224, 224, 224], False, 0.3, "relu"))
    specs.append(ArchSpec("V19", "grid", 4, [160, 160, 160, 160], True, 0.3, "relu"))
    specs.append(ArchSpec("V20", "grid", 3, [256, 256, 256], True, 0.4, "relu"))

    return specs


def build_search_space() -> List[ArchSpec]:
    specs = get_handcrafted_specs() + get_parametric_specs_v11_v20_diverse()
    return sorted(specs, key=lambda s: int(s.id[1:]))[:MAX_VERSIONS]


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


def print_built_architectures(specs: List[ArchSpec]) -> None:
    in_dim = CLIP_DIM + (GEO_DIM if USE_GEO_BRANCH else 0)
    print("=== Built MLP architectures (actual code) ===")
    print(f"in_dim={in_dim}  out_dim={OUT_DIM}")
    for spec in specs:
        model = build_model_from_spec(spec)
        widths_str = "-".join(str(w) for w in spec.widths) if spec.widths else "-"
        drop_str = "-" if spec.dropout <= 0 else str(spec.dropout)
        bn_str = "Yes" if spec.use_bn else "No"
        print(f"{spec.id}  depth={spec.depth}  widths={widths_str}  drop={drop_str}  bn={bn_str}")
        print(f"  {describe_model_layers(model)}")
    print("=== End built architectures ===")


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


def geo_stats(geo: Optional[np.ndarray], idx) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if geo is None:
        return None, None
    g = geo[np.asarray(idx)]
    return g.mean(axis=0).astype(np.float32), np.maximum(g.std(axis=0), 1e-6).astype(np.float32)


def build_inputs(clip: np.ndarray, geo: Optional[np.ndarray],
                 geo_mean: Optional[np.ndarray], geo_std: Optional[np.ndarray]) -> np.ndarray:
    x = l2_normalize(clip) if L2_NORMALIZE_FEATURES else clip.astype(np.float32)
    if geo is None:
        return x
    geo_z = ((geo - geo_mean) / np.maximum(geo_std, 1e-6)).astype(np.float32)
    return np.hstack([x, geo_z]).astype(np.float32)


# ===================== Training =====================
def compute_class_weights(y: np.ndarray, n_classes: int) -> Optional[torch.Tensor]:
    if not USE_CLASS_WEIGHTED_CE:
        return None
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = counts.sum() / (n_classes * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def _loss(model, xb, yb, class_weights):
    if USE_MIXUP and MIXUP_ALPHA > 0:
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        perm = torch.randperm(xb.size(0), device=xb.device)
        logits = model(lam * xb + (1.0 - lam) * xb[perm])
        ce = nn.functional.cross_entropy
        return lam * ce(logits, yb, weight=class_weights) + (1.0 - lam) * ce(logits, yb[perm], weight=class_weights)
    logits = model(xb)
    return nn.functional.cross_entropy(logits, yb, weight=class_weights)


def _train_one_seed(spec: ArchSpec, X: np.ndarray, y: np.ndarray, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model_from_spec(spec).to(DEVICE)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) if USE_COSINE_LR else None
    class_weights = compute_class_weights(y, OUT_DIM)

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = _loss(model, xb, yb, class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    model.eval()
    return model


@torch.no_grad()
def predict_proba(models: List[nn.Module], X: np.ndarray) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float32)
    total: Optional[np.ndarray] = None
    for m in models:
        m.eval()
        out = []
        for i in range(0, len(Xt), PREDICT_BATCH_SIZE):
            out.append(torch.softmax(m(Xt[i:i + PREDICT_BATCH_SIZE].to(DEVICE)), dim=1).cpu().numpy())
        p = np.concatenate(out, axis=0)
        total = p if total is None else total + p
    return total / len(models)


def _f1_from_preds(gts: np.ndarray, preds: np.ndarray) -> Tuple[float, float, np.ndarray]:
    acc = float(accuracy_score(gts, preds))
    f1_mac = float(f1_score(gts, preds, average="macro", zero_division=0))
    f1_per_class = f1_score(gts, preds, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    return acc, f1_mac, f1_per_class


def train_one_fold(spec, clip, geo, y, is_orig, train_idx, val_idx, base_seed):
    g_mean, g_std = geo_stats(geo, train_idx)
    X_tr = build_inputs(clip[train_idx], None if geo is None else geo[train_idx], g_mean, g_std)
    X_va = build_inputs(clip[val_idx], None if geo is None else geo[val_idx], g_mean, g_std)

    models = [
        _train_one_seed(spec, X_tr, y[train_idx], seed=base_seed + 7919 * s)
        for s in range(max(1, N_SEEDS_PER_FOLD))
    ]
    preds = predict_proba(models, X_va).argmax(axis=1)

    scored = is_orig[val_idx] if SCORE_ORIGINAL_PATCHES_ONLY else np.ones(len(val_idx), dtype=bool)
    gts = y[val_idx]
    acc, f1_mac, f1_per_class = _f1_from_preds(gts[scored], preds[scored])
    return acc, f1_mac, f1_per_class, preds[scored].tolist(), gts[scored].tolist()


# ===================== Reports =====================
def save_confusion_matrix(y_true, y_pred, arch_id, fold, acc=None, f1_macro=None, path_prefix=CM_SAVE_DIR):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    extras = []
    if acc is not None:
        extras.append(f"acc={acc:.4f}")
    if f1_macro is not None:
        extras.append(f"f1={f1_macro:.4f}")

    title = f"Fold {fold}" if fold is not None else ""
    if extras:
        title = (title + " | " if title else "") + ", ".join(extras)
    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(f"{path_prefix}/confmat_{arch_id}_fold{fold}.png")
    plt.close()
    return cm


def notebooks_per_fold(groups: np.ndarray, splits) -> List[int]:
    return [int(len(np.unique(groups[val_idx]))) for _, val_idx in splits]


def _fmt_fold_counts(counts: List[int]) -> str:
    parts = [str(n) for n in counts]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def fold_assignment_sentence(counts: List[int]) -> str:
    return (
        f"The {sum(counts)} training notebooks stay whole and form five folds of "
        f"{_fmt_fold_counts(counts)} notebooks."
    )


def write_f1_version_report_txt(results: List[Dict[str, Any]], out_path: str, fold_note: str) -> Dict[str, Any]:
    lines = [
        "=== Orientation MLP architecture search report ===",
        f"FOLDS={FOLDS} (notebook-level, stratified)  N_SEEDS_PER_FOLD={N_SEEDS_PER_FOLD} (probabilities averaged)",
        f"RankScore = Avg F1(macro) - {STD_PENALTY_LAMBDA} * Std F1(macro), computed on original validation patches",
        fold_note,
        f"Input: CLIP ViT-L/14 (letterbox={USE_LETTERBOX_SQUARE}) + geometric branch={USE_GEO_BRANCH} | "
        f"Loss: class-weighted CE, mixup={USE_MIXUP}(alpha={MIXUP_ALPHA}) | "
        f"fixed {EPOCHS} epochs, no early stopping",
        "",
    ]
    header = (f"{'ID':<4}  {'Depth':>5}  {'Widths':<18}  {'Drop':>4}  {'BN':<3}  "
              f"{'Avg F1(macro) ± std':>22}  {'RankScore':>10}  {'Combined F1':>11}")
    lines += [header, "-" * len(header)]
    for r in results:
        drop = "-" if r["dropout"] <= 0 else f"{r['dropout']:g}"
        lines.append(
            f"{r['arch_id']:<4}  {r['depth']:>5}  {r['widths']:<18}  {drop:>4}  {'Yes' if r['use_bn'] else 'No':<3}  "
            f"{r['avg_val_f1_macro']:>10.4f} ± {r['std_val_f1_macro']:<8.4f}  {r['rank_score']:>10.4f}  "
            f"{r['combined_f1_macro']:>11.4f}"
        )
    best = max(results, key=lambda r: r[RANK_BY])
    best_avg = max(results, key=lambda r: r["avg_val_f1_macro"])
    lines += [
        "",
        "--- Summary ---",
        f"Best RankScore:          {best['arch_id']} = {best['rank_score']:.4f} "
        f"(Avg={best['avg_val_f1_macro']:.4f}, Std={best['std_val_f1_macro']:.4f})",
        f"Best Avg val F1(macro):  {best_avg['arch_id']} = {best_avg['avg_val_f1_macro']:.4f} ± "
        f"{best_avg['std_val_f1_macro']:.4f}",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return best


# ===================== Notebook-level folds =====================
def extract_notebook_ids(filenames: List[str]) -> np.ndarray:
    groups = []

    for fname in filenames:
        base = os.path.basename(str(fname))
        notebook_id = base.split("_", 1)[0]
        groups.append(notebook_id)

    return np.array(groups)


def is_original_patch(fname: str) -> bool:
    """True for non-augmented patches (no _augN / _balN / _tltN suffix)."""
    stem = os.path.splitext(os.path.basename(str(fname)))[0]
    last = stem.rsplit("_", 1)[-1]
    kind = "".join(c for c in last if not c.isdigit())
    return kind not in {"aug", "bal", "tlt"}


def _local_search_balance(
    fold_of_group: Dict[str, int],
    group_class_counts: Dict[str, np.ndarray],
    n_splits: int,
    n_classes: int,
    target_size: float,
    target_class: np.ndarray,
    size_weight: float,
    class_weight: float,
    rng: np.random.Generator,
    max_passes: int,
    min_per_class: int = 0,
    floor_weight: float = 10.0,
) -> Tuple[Dict[str, int], np.ndarray, float]:
    fold_class_counts = np.zeros((n_splits, n_classes), dtype=np.int64)

    for g, f in fold_of_group.items():
        fold_class_counts[f] += group_class_counts[g]

    target_class_safe = np.maximum(target_class, 1.0)

    def fold_cost(counts: np.ndarray) -> float:
        rel_size = abs(float(counts.sum()) - target_size) / target_size
        rel_class = float(np.mean(np.abs(counts - target_class) / target_class_safe))

        floor = 0.0
        if min_per_class > 0:
            for c in counts:
                if c < min_per_class:
                    floor += float(min_per_class - c)

        return size_weight * rel_size + class_weight * rel_class + floor_weight * floor

    def total_cost(mat: np.ndarray) -> float:
        return float(sum(fold_cost(mat[f]) for f in range(n_splits)))

    for _ in range(max_passes):
        improved = False

        order = list(fold_of_group.keys())
        rng.shuffle(order)

        for g in order:
            src = fold_of_group[g]
            cls = group_class_counts[g]

            old_src = fold_cost(fold_class_counts[src])
            best_dst = src
            best_delta = 0.0

            for dst in range(n_splits):
                if dst == src:
                    continue

                old_dst = fold_cost(fold_class_counts[dst])
                new_src = fold_cost(fold_class_counts[src] - cls)
                new_dst = fold_cost(fold_class_counts[dst] + cls)
                delta = (new_src + new_dst) - (old_src + old_dst)

                if delta + 1e-12 < best_delta:
                    best_delta = delta
                    best_dst = dst

            if best_dst != src:
                fold_class_counts[src] -= cls
                fold_class_counts[best_dst] += cls
                fold_of_group[g] = best_dst
                improved = True

        pair_order = list(fold_of_group.keys())
        rng.shuffle(pair_order)

        for i in range(len(pair_order)):
            gi = pair_order[i]

            for j in range(i + 1, len(pair_order)):
                gj = pair_order[j]
                fi = fold_of_group[gi]
                fj = fold_of_group[gj]

                if fi == fj:
                    continue

                ci = group_class_counts[gi]
                cj = group_class_counts[gj]

                old = fold_cost(fold_class_counts[fi]) + fold_cost(fold_class_counts[fj])
                new = (
                    fold_cost(fold_class_counts[fi] - ci + cj)
                    + fold_cost(fold_class_counts[fj] - cj + ci)
                )

                if new + 1e-12 < old:
                    fold_class_counts[fi] = fold_class_counts[fi] - ci + cj
                    fold_class_counts[fj] = fold_class_counts[fj] - cj + ci
                    fold_of_group[gi] = fj
                    fold_of_group[gj] = fi
                    improved = True

        if not improved:
            break

    return fold_of_group, fold_class_counts, total_cost(fold_class_counts)


def make_balanced_grouped_splits(
    y_np: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    size_weight: float = 1.0,
    class_weight: float = 1.0,
    n_restarts: int = 32,
    n_kicks_per_restart: int = 30,
    kick_strength: int = 3,
    max_passes: int = 100,
    min_per_class: int = 5,
    floor_weight: float = 10.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_classes = int(y_np.max()) + 1

    unique_groups = np.unique(groups)
    group_class_counts: Dict[str, np.ndarray] = {}

    for g in unique_groups:
        mask = groups == g
        group_class_counts[str(g)] = np.bincount(
            y_np[mask],
            minlength=n_classes,
        ).astype(np.int64)

    target_size = len(y_np) / n_splits
    target_class = np.bincount(y_np, minlength=n_classes).astype(np.float64) / n_splits

    def stratified_init(init_seed: int) -> Dict[str, int]:
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=init_seed,
        )

        mapping: Dict[str, int] = {}

        for fold_idx, (_, val_idx) in enumerate(
            sgkf.split(np.zeros(len(y_np)), y_np, groups=groups)
        ):
            for g in np.unique(groups[val_idx]):
                mapping[str(g)] = fold_idx

        return mapping

    def random_init(init_seed: int) -> Dict[str, int]:
        rng_local = np.random.default_rng(init_seed)
        order = list(map(str, unique_groups))
        sizes = np.array([group_class_counts[g].sum() for g in order])

        idx = np.argsort(-sizes)
        order = [order[i] for i in idx]

        fold_sizes = np.zeros(n_splits, dtype=np.float64)
        mapping: Dict[str, int] = {}

        for g in order:
            ranked = np.argsort(fold_sizes + rng_local.uniform(0, 1e-3, size=n_splits))
            mapping[g] = int(ranked[0])
            fold_sizes[mapping[g]] += group_class_counts[g].sum()

        return mapping

    inits: List[Dict[str, int]] = [stratified_init(seed)]

    for k in range(n_restarts - 1):
        sub = seed + 7919 * (k + 1)
        if k % 2 == 0:
            inits.append(stratified_init(sub))
        else:
            inits.append(random_init(sub))

    best_mapping: Optional[Dict[str, int]] = None
    best_cost = float("inf")
    group_keys = list(map(str, unique_groups))

    for k, init in enumerate(inits):
        rng = np.random.default_rng(seed + 1000003 * (k + 1))

        mapping, _, cost = _local_search_balance(
            fold_of_group=dict(init),
            group_class_counts=group_class_counts,
            n_splits=n_splits,
            n_classes=n_classes,
            target_size=target_size,
            target_class=target_class,
            size_weight=size_weight,
            class_weight=class_weight,
            rng=rng,
            max_passes=max_passes,
            min_per_class=min_per_class,
            floor_weight=floor_weight,
        )

        local_best_mapping = dict(mapping)
        local_best_cost = cost

        for _ in range(n_kicks_per_restart):
            kicked = dict(local_best_mapping)
            n_kick = max(1, kick_strength)

            kick_groups = rng.choice(
                group_keys,
                size=min(n_kick, len(group_keys)),
                replace=False,
            )

            for kg in kick_groups:
                cur = kicked[str(kg)]
                choices = [f for f in range(n_splits) if f != cur]
                kicked[str(kg)] = int(rng.choice(choices))

            mapping2, _, cost2 = _local_search_balance(
                fold_of_group=kicked,
                group_class_counts=group_class_counts,
                n_splits=n_splits,
                n_classes=n_classes,
                target_size=target_size,
                target_class=target_class,
                size_weight=size_weight,
                class_weight=class_weight,
                rng=rng,
                max_passes=max_passes,
                min_per_class=min_per_class,
                floor_weight=floor_weight,
            )

            if cost2 + 1e-12 < local_best_cost:
                local_best_cost = cost2
                local_best_mapping = dict(mapping2)

        if local_best_cost < best_cost - 1e-12:
            best_cost = local_best_cost
            best_mapping = local_best_mapping

    assert best_mapping is not None

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(y_np))

    for fold_idx in range(n_splits):
        val_mask = np.array([best_mapping[str(g)] == fold_idx for g in groups])
        val_idx = all_indices[val_mask]
        train_idx = all_indices[~val_mask]
        splits.append((train_idx, val_idx))

    return splits


# ===================== Main =====================
def main():
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
    groups = extract_notebook_ids(filenames)
    is_orig = np.array([is_original_patch(f) for f in filenames])

    geo = None
    if USE_GEO_BRANCH:
        print(f"Extracting geometric features from: {PATCH_FOLDER}")
        geo = extract_geo(PATCH_FOLDER, filenames)

    print(f"Samples: {len(y_np)} (original={int(is_orig.sum())})")
    print(f"Class counts: {np.bincount(y_np).tolist()}")
    print(f"Notebooks: {len(np.unique(groups))}")
    print("Split mode: notebook-grouped stratified k-fold (size-balanced)")
    print(f"Device: {DEVICE}")

    if USE_CLASS_WEIGHTED_CE:
        global_counts = np.bincount(y_np, minlength=OUT_DIM).astype(np.float64)
        global_weights = len(y_np) / (OUT_DIM * np.maximum(global_counts, 1.0))
        print(
            "Loss: class-weighted CrossEntropy "
            f"(global ref weights={[round(float(w), 3) for w in global_weights]})"
        )
    else:
        print("Loss: vanilla CrossEntropy")

    print(
        f"Optim: AdamW lr={LEARNING_RATE} wd={WEIGHT_DECAY} "
        f"cosine_lr={USE_COSINE_LR} epochs={EPOCHS} "
        f"early_stop={EARLY_STOPPING} "
        f"L2_norm_features={L2_NORMALIZE_FEATURES}"
    )
    print(f"Aug: mixup={USE_MIXUP}(alpha={MIXUP_ALPHA}) seeds_per_fold={N_SEEDS_PER_FOLD}")
    print(
        f"Input: CLIP ViT-L/14 letterbox={USE_LETTERBOX_SQUARE} | "
        f"geometric branch={USE_GEO_BRANCH} ({GEO_DIM} features, hidden={GEO_HIDDEN})"
    )
    print(f"Scoring: original validation patches only={SCORE_ORIGINAL_PATCHES_ONLY}")
    print(f"Ranking: {RANK_BY} = avg_val_f1_macro - {STD_PENALTY_LAMBDA} * std_val_f1_macro")

    specs = build_search_space()
    print_built_architectures(specs)
    print(f"Search space size: {len(specs)} architectures (V01-V{MAX_VERSIONS:02d})")

    splits = make_balanced_grouped_splits(y_np=y_np, groups=groups, n_splits=FOLDS, seed=SEED)
    fold_note = fold_assignment_sentence(notebooks_per_fold(groups, splits))
    print(fold_note)

    print("\nFold composition:")
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        val_groups = np.unique(groups[val_idx])
        overlap = np.intersect1d(val_groups, np.unique(groups[train_idx]))
        val_orig = val_idx[is_orig[val_idx]]
        print(
            f"  Fold {fold_idx + 1}: "
            f"val_size={len(val_idx)} (original={len(val_orig)}), "
            f"val_notebooks={len(val_groups)}, "
            f"val_class_counts (original)={np.bincount(y_np[val_orig], minlength=len(CLASS_NAMES)).tolist()}, "
            f"group_overlap_with_train={len(overlap)}"
        )

    results = []

    for spec in specs:
        print(
            f"\n{spec.id} | kind={spec.kind} depth={spec.depth} "
            f"widths={spec.widths if spec.widths else '-'} "
            f"BN={spec.use_bn} dropout={spec.dropout} act={spec.activation}"
        )

        fold_accs, fold_f1s = [], []
        fold_f1_hor, fold_f1_vrt, fold_f1_tlt = [], [], []
        all_preds, all_gts = [], []

        for fold, (train_idx, val_idx) in enumerate(splits):
            val_orig = val_idx[is_orig[val_idx]]
            print(
                f"  Fold {fold + 1}: "
                f"train samples={len(train_idx)}, "
                f"val samples={len(val_idx)} (original={len(val_orig)}), "
                f"val counts={np.bincount(y_np[val_orig], minlength=len(CLASS_NAMES)).tolist()}"
            )

            acc, f1_mac, f1_per_class, preds, gts = train_one_fold(
                spec, clip, geo, y_np, is_orig, train_idx, val_idx,
                base_seed=SEED + 1009 * (fold + 1),
            )

            fold_accs.append(acc)
            fold_f1s.append(f1_mac)
            fold_f1_hor.append(float(f1_per_class[0]))
            fold_f1_vrt.append(float(f1_per_class[1]))
            fold_f1_tlt.append(float(f1_per_class[2]))
            all_preds.extend(preds)
            all_gts.extend(gts)

            save_confusion_matrix(gts, preds, arch_id=spec.id, fold=fold + 1, acc=acc, f1_macro=f1_mac)

            print(
                f"    Acc={acc:.4f} | "
                f"F1(macro)={f1_mac:.4f} | "
                f"hor={f1_per_class[0]:.4f} "
                f"vrt={f1_per_class[1]:.4f} "
                f"tlt={f1_per_class[2]:.4f}"
            )

        avg_acc = float(np.mean(fold_accs))
        avg_f1_macro = float(np.mean(fold_f1s))
        std_f1_macro = float(np.std(fold_f1s))
        rank_score = avg_f1_macro - STD_PENALTY_LAMBDA * std_f1_macro

        combined_acc = float(accuracy_score(all_gts, all_preds))
        combined_f1_macro = float(f1_score(all_gts, all_preds, average="macro", zero_division=0))

        save_confusion_matrix(all_gts, all_preds, arch_id=spec.id, fold="all",
                              acc=combined_acc, f1_macro=combined_f1_macro)

        results.append({
            "arch_id": spec.id,
            "kind": spec.kind,
            "depth": spec.depth,
            "widths": "-".join(map(str, spec.widths)) if spec.widths else "-",
            "use_bn": int(spec.use_bn),
            "dropout": float(spec.dropout),
            "activation": spec.activation,
            "avg_val_accuracy": avg_acc,
            "avg_val_f1_macro": avg_f1_macro,
            "std_val_f1_macro": std_f1_macro,
            "rank_score": rank_score,
            "avg_val_f1_hor": float(np.mean(fold_f1_hor)),
            "avg_val_f1_vrt": float(np.mean(fold_f1_vrt)),
            "avg_val_f1_tlt": float(np.mean(fold_f1_tlt)),
            "combined_accuracy": combined_acc,
            "combined_f1_macro": combined_f1_macro,
        })

        print(
            f"Avg Acc={avg_acc:.4f} | "
            f"Avg F1(macro)={avg_f1_macro:.4f} ± {std_f1_macro:.4f} | "
            f"RankScore={rank_score:.4f} "
            f"(avg - {STD_PENALTY_LAMBDA}*std) | "
            f"Combined Acc={combined_acc:.4f} | "
            f"Combined F1(macro)={combined_f1_macro:.4f}"
        )

    best = write_f1_version_report_txt(results, REPORT_PATH, fold_note)

    print(f"\nF1 summary report (txt): {REPORT_PATH}")
    print(
        f"Best Arch by {RANK_BY}: {best['arch_id']} "
        f"with {RANK_BY}={best['rank_score']:.4f} "
        f"(avg_val_f1_macro={best['avg_val_f1_macro']:.4f}, "
        f"std_val_f1_macro={best['std_val_f1_macro']:.4f})"
    )


if __name__ == "__main__":
    main()
