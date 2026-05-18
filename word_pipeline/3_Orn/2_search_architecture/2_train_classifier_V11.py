import os
import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional


# ============== Config ==============
FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 256
IN_DIM = 768
OUT_DIM = 3

CM_SAVE_DIR = "confusion_matrices_orientation"
DATA_PATH = "orientation_features_clipL14_all.npz"
CLASS_NAMES = ["hor", "vrt", "tlt"]
SEED = 42

MAX_VERSIONS = 20
SEARCH_MODE = "ORIGINAL_30"

USE_CLASS_WEIGHTED_CE = True
L2_NORMALIZE_FEATURES = True
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
USE_COSINE_LR = True

EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 8

USE_MIXUP = True
MIXUP_ALPHA = 0.2

N_SEEDS_PER_FOLD = 3

# Stability-aware ranking:
# rank_score = avg_val_f1_macro - STD_PENALTY_LAMBDA * std_val_f1_macro
RANK_BY = "rank_score"
STD_PENALTY_LAMBDA = 0.5

GRID_CONFIG = {
    "depths": [2, 3, 4, 5],
    "widths": [128, 160, 192, 224, 256],
    "use_bn": [False, True],
    "dropouts": [0.2],
    "activation": "relu",
}

os.makedirs(CM_SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


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


def _act(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def get_handcrafted_specs() -> List[ArchSpec]:
    return [
        ArchSpec("V01", "handcrafted", 0, [], False, 0.0, "-", "Linear baseline 768->3"),
        ArchSpec("V02", "handcrafted", 1, [128], False, 0.0, "relu"),
        ArchSpec("V03", "handcrafted", 1, [256], False, 0.0, "relu"),
        ArchSpec("V04", "handcrafted", 1, [256], False, 0.2, "relu"),
        ArchSpec("V05", "handcrafted", 2, [512, 256], False, 0.3, "relu"),
        ArchSpec("V06", "handcrafted", 2, [512, 256], False, 0.3, "relu"),
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


def get_parametric_specs_extended(cfg: Dict[str, Any]) -> List[ArchSpec]:
    specs = []
    counter = 1

    for d in cfg["depths"]:
        for w in cfg["widths"]:
            for bn in cfg["use_bn"]:
                for p in cfg["dropouts"]:
                    specs.append(
                        ArchSpec(
                            id=f"G{counter:03d}",
                            kind="grid",
                            depth=d,
                            widths=[w] * d,
                            use_bn=bn,
                            dropout=float(p),
                            activation=cfg["activation"],
                        )
                    )
                    counter += 1

    return specs


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


def save_confusion_matrix(
    y_true,
    y_pred,
    arch_id,
    fold,
    acc=None,
    f1_macro=None,
    path_prefix=CM_SAVE_DIR,
):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

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
    path = f"{path_prefix}/confmat_{arch_id}_fold{fold}.png"
    plt.savefig(path)
    plt.close()

    return cm


def compute_class_weights(y_subset: torch.Tensor, n_classes: int) -> Optional[torch.Tensor]:
    if not USE_CLASS_WEIGHTED_CE:
        return None

    counts = np.bincount(y_subset.cpu().numpy(), minlength=n_classes).astype(np.float64)
    n_total = counts.sum()

    weights = np.zeros(n_classes, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = n_total / (n_classes * counts[nonzero])

    return torch.tensor(weights, dtype=torch.float32, device="cuda")


def _mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, y, 1.0

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]

    return mixed_x, y, y[perm], lam


@torch.no_grad()
def _eval_logits(model, val_loader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []
    gts_list: List[np.ndarray] = []

    for xb, yb in val_loader:
        xb = xb.cuda()
        logits_list.append(model(xb).cpu().numpy())
        gts_list.append(yb.numpy())

    return np.concatenate(logits_list, axis=0), np.concatenate(gts_list, axis=0)


def _f1_from_preds(gts: np.ndarray, preds: np.ndarray) -> Tuple[float, float, np.ndarray]:
    acc = float(accuracy_score(gts, preds))
    f1_mac = float(f1_score(gts, preds, average="macro", zero_division=0))
    f1_per_class = f1_score(
        gts,
        preds,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    return acc, f1_mac, f1_per_class


def _train_one_seed(
    spec: ArchSpec,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model_from_spec(spec, IN_DIM, OUT_DIM).cuda()

    train_ds = Subset(TensorDataset(X, y), train_idx)
    val_ds = Subset(TensorDataset(X, y), val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        if USE_COSINE_LR
        else None
    )

    class_weights = compute_class_weights(y[train_idx], n_classes=OUT_DIM)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_f1 = -1.0
    epochs_no_improve = 0

    for _ in range(EPOCHS):
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.cuda(), yb.cuda()

            if USE_MIXUP and MIXUP_ALPHA > 0:
                mx, ya, yb_perm, lam = _mixup_batch(xb, yb, MIXUP_ALPHA)
                logits = model(mx)
                loss = lam * nn.functional.cross_entropy(
                    logits,
                    ya,
                    weight=class_weights,
                ) + (1.0 - lam) * nn.functional.cross_entropy(
                    logits,
                    yb_perm,
                    weight=class_weights,
                )
            else:
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits, yb, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        logits, gts = _eval_logits(model, val_loader)
        preds = logits.argmax(axis=1)
        f1_mac = float(f1_score(gts, preds, average="macro", zero_division=0))

        if f1_mac > best_f1 + 1e-6:
            best_f1 = f1_mac
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if EARLY_STOPPING and epochs_no_improve >= EARLY_STOP_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return _eval_logits(model, val_loader)


def train_one_fold(
    spec: ArchSpec,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    base_seed: int,
):
    n_seeds = max(1, N_SEEDS_PER_FOLD)
    avg_probs: Optional[np.ndarray] = None
    gts_ref: Optional[np.ndarray] = None

    for s in range(n_seeds):
        logits, gts = _train_one_seed(
            spec,
            X,
            y,
            train_idx,
            val_idx,
            seed=base_seed + 7919 * s,
        )

        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        avg_probs = probs if avg_probs is None else avg_probs + probs
        gts_ref = gts

    assert avg_probs is not None and gts_ref is not None

    avg_probs /= n_seeds
    preds = avg_probs.argmax(axis=1)

    acc, f1_mac, f1_per_class = _f1_from_preds(gts_ref, preds)
    return acc, f1_mac, f1_per_class, preds.tolist(), gts_ref.tolist()


def train_final_model(model, X, y, n_epochs: Optional[int] = None):
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    epochs = n_epochs if n_epochs is not None else EPOCHS

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if USE_COSINE_LR
        else None
    )

    class_weights = compute_class_weights(y, n_classes=OUT_DIM)

    for _ in range(epochs):
        model.train()

        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda()

            if USE_MIXUP and MIXUP_ALPHA > 0:
                mx, ya, yb_perm, lam = _mixup_batch(xb, yb, MIXUP_ALPHA)
                logits = model(mx)
                loss = lam * nn.functional.cross_entropy(
                    logits,
                    ya,
                    weight=class_weights,
                ) + (1.0 - lam) * nn.functional.cross_entropy(
                    logits,
                    yb_perm,
                    weight=class_weights,
                )
            else:
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits, yb, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()


def build_search_space() -> List[ArchSpec]:
    if SEARCH_MODE == "ORIGINAL_30":
        specs = get_handcrafted_specs() + get_parametric_specs_v11_v20_diverse()

        def _id_num(sid: str) -> int:
            try:
                return int(sid[1:])
            except Exception:
                return 10**9

        specs = sorted(specs, key=lambda s: _id_num(s.id))
        specs = [
            s
            for s in specs
            if s.id.startswith("V") and _id_num(s.id) <= 20
        ][:MAX_VERSIONS]

    elif SEARCH_MODE == "EXTENDED_GRID":
        specs = get_parametric_specs_extended(GRID_CONFIG)
        specs = specs[:MAX_VERSIONS]

    else:
        raise ValueError(f"Unknown SEARCH_MODE: {SEARCH_MODE}")

    return specs


def write_f1_version_report_txt(
    results: List[Dict[str, Any]],
    out_path: str,
    max_version: int = 20,
) -> None:
    by_num: Dict[int, Dict[str, Any]] = {}

    for rec in results:
        aid = rec.get("arch_id", "")
        if not (isinstance(aid, str) and aid.startswith("V")):
            continue

        suffix = aid[1:]
        if not suffix.isdigit():
            continue

        n = int(suffix)
        if 1 <= n <= max_version:
            by_num[n] = rec

    lines: List[str] = []
    lines.append("=== Báo cáo F1 theo version (orientation classifier) ===")
    lines.append(
        f"SEARCH_MODE={SEARCH_MODE}  FOLDS={FOLDS}  "
        f"N_SEEDS_PER_FOLD={N_SEEDS_PER_FOLD}"
    )
    lines.append(
        f"RankScore = Avg F1(macro) - {STD_PENALTY_LAMBDA} * Std F1(macro)"
    )
    lines.append("")
    lines.append(
        "Avg F1(macro) ± std: trung bình F1 macro trên các fold val ± độ lệch chuẩn theo fold."
    )
    lines.append(
        "RankScore: điểm xếp hạng ưu tiên model vừa cao vừa ổn định giữa folds."
    )
    lines.append(
        "Combined F1(macro): F1 macro trên toàn bộ val khi gộp mọi fold."
    )
    lines.append("")
    lines.append(
        f"{'Version':<8} | {'Avg F1(macro) ± std':>28} | "
        f"{'RankScore':>10} | {'Combined F1(macro)':>20}"
    )
    lines.append("-" * 82)

    for v in range(1, max_version + 1):
        vid = f"V{v:02d}"
        rec = by_num.get(v)

        if rec is None:
            lines.append(
                f"{vid:<8} | {'(không có trong kết quả)':>28} | "
                f"{'-':>10} | {'-':>20}"
            )
            continue

        avg = float(rec["avg_val_f1_macro"])
        std = float(rec["std_val_f1_macro"])
        rank = float(rec["rank_score"])
        comb = float(rec["combined_f1_macro"])

        lines.append(
            f"{vid:<8} | {avg:>12.4f} ± {std:>10.4f} | "
            f"{rank:>10.4f} | {comb:>20.4f}"
        )

    present = [by_num[v] for v in range(1, max_version + 1) if v in by_num]

    if present:
        lines.append("")
        lines.append("--- Tóm tắt ---")

        best_rank = max(present, key=lambda r: float(r["rank_score"]))
        best_avg = max(present, key=lambda r: float(r["avg_val_f1_macro"]))
        best_comb = max(present, key=lambda r: float(r["combined_f1_macro"]))

        lines.append(
            f"Best RankScore:          {best_rank['arch_id']} = "
            f"{float(best_rank['rank_score']):.4f} "
            f"(Avg={float(best_rank['avg_val_f1_macro']):.4f}, "
            f"Std={float(best_rank['std_val_f1_macro']):.4f})"
        )
        lines.append(
            f"Best Avg val F1(macro):  {best_avg['arch_id']} = "
            f"{float(best_avg['avg_val_f1_macro']):.4f} ± "
            f"{float(best_avg['std_val_f1_macro']):.4f}"
        )
        lines.append(
            f"Best Combined F1(macro): {best_comb['arch_id']} = "
            f"{float(best_comb['combined_f1_macro']):.4f}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def extract_notebook_ids(filenames: List[str]) -> np.ndarray:
    groups = []

    for fname in filenames:
        base = os.path.basename(str(fname))
        notebook_id = base.split("_", 1)[0]
        groups.append(notebook_id)

    return np.array(groups)


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


def main():
    print("Loading data...")

    data = np.load(DATA_PATH, allow_pickle=True)

    X = torch.tensor(data["features"], dtype=torch.float32)

    if L2_NORMALIZE_FEATURES:
        X = nn.functional.normalize(X, p=2, dim=1)

    y = torch.tensor(data["labels"], dtype=torch.long)
    y_np = y.numpy()

    if "filenames" not in data.files:
        raise RuntimeError(
            f"'{DATA_PATH}' has no 'filenames' field; re-run feature extraction."
        )

    filenames = list(data["filenames"])
    groups = extract_notebook_ids(filenames)
    unique_notebooks = np.unique(groups)

    print(f"Samples: {len(X)}")
    print(f"Class counts: {np.bincount(y_np).tolist()}")
    print(f"Notebooks: {len(unique_notebooks)}")
    print("Split mode: notebook-grouped stratified k-fold (size-balanced)")

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
        f"early_stop={EARLY_STOPPING}(patience={EARLY_STOP_PATIENCE}) "
        f"L2_norm_features={L2_NORMALIZE_FEATURES}"
    )

    print(
        f"Aug: mixup={USE_MIXUP}(alpha={MIXUP_ALPHA}) "
        f"seeds_per_fold={N_SEEDS_PER_FOLD}"
    )

    print(
        f"Ranking: {RANK_BY} = avg_val_f1_macro "
        f"- {STD_PENALTY_LAMBDA} * std_val_f1_macro"
    )

    specs = build_search_space()

    print(
        f"Search space size: {len(specs)} architectures "
        f"({SEARCH_MODE}; limited to first {MAX_VERSIONS})"
    )

    splits = make_balanced_grouped_splits(
        y_np=y_np,
        groups=groups,
        n_splits=FOLDS,
        seed=SEED,
    )

    print("\nFold composition:")
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        val_groups = np.unique(groups[val_idx])
        train_groups = np.unique(groups[train_idx])
        overlap = np.intersect1d(val_groups, train_groups)
        val_class_counts = np.bincount(
            y_np[val_idx],
            minlength=len(CLASS_NAMES),
        ).tolist()

        print(
            f"  Fold {fold_idx + 1}: "
            f"val_size={len(val_idx)}, "
            f"val_notebooks={len(val_groups)}, "
            f"val_class_counts={val_class_counts}, "
            f"group_overlap_with_train={len(overlap)}"
        )

    results = []
    best_arch_id: Optional[str] = None
    best_score: float = -1.0
    best_avg_f1: float = -1.0
    best_std_f1: float = 0.0

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
            val_counts = np.bincount(y_np[val_idx], minlength=len(CLASS_NAMES))

            print(
                f"  Fold {fold + 1}: "
                f"train samples={len(train_idx)}, "
                f"val samples={len(val_idx)}, "
                f"val counts={val_counts.tolist()}"
            )

            acc, f1_mac, f1_per_class, preds, gts = train_one_fold(
                spec,
                X,
                y,
                train_idx,
                val_idx,
                base_seed=SEED + 1009 * (fold + 1),
            )

            fold_accs.append(acc)
            fold_f1s.append(f1_mac)

            fold_f1_hor.append(float(f1_per_class[0]))
            fold_f1_vrt.append(float(f1_per_class[1]))
            fold_f1_tlt.append(float(f1_per_class[2]))

            all_preds.extend(preds)
            all_gts.extend(gts)

            save_confusion_matrix(
                gts,
                preds,
                arch_id=spec.id,
                fold=fold + 1,
                acc=acc,
                f1_macro=f1_mac,
            )

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

        avg_f1_hor = float(np.mean(fold_f1_hor))
        avg_f1_vrt = float(np.mean(fold_f1_vrt))
        avg_f1_tlt = float(np.mean(fold_f1_tlt))

        combined_acc = float(accuracy_score(all_gts, all_preds))
        combined_f1_macro = float(
            f1_score(all_gts, all_preds, average="macro", zero_division=0)
        )
        combined_f1_per_class = f1_score(
            all_gts,
            all_preds,
            average=None,
            labels=list(range(len(CLASS_NAMES))),
            zero_division=0,
        )

        save_confusion_matrix(
            all_gts,
            all_preds,
            arch_id=spec.id,
            fold="all",
            acc=combined_acc,
            f1_macro=combined_f1_macro,
        )

        rec = {
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
            "std_penalty_lambda": STD_PENALTY_LAMBDA,

            "avg_val_f1_hor": avg_f1_hor,
            "avg_val_f1_vrt": avg_f1_vrt,
            "avg_val_f1_tlt": avg_f1_tlt,

            "combined_accuracy": combined_acc,
            "combined_f1_macro": combined_f1_macro,
            "combined_f1_hor": float(combined_f1_per_class[0]),
            "combined_f1_vrt": float(combined_f1_per_class[1]),
            "combined_f1_tlt": float(combined_f1_per_class[2]),
        }

        if spec.notes:
            rec["notes"] = spec.notes

        results.append(rec)

        print(
            f"Avg Acc={avg_acc:.4f} | "
            f"Avg F1(macro)={avg_f1_macro:.4f} ± {std_f1_macro:.4f} | "
            f"RankScore={rank_score:.4f} "
            f"(avg - {STD_PENALTY_LAMBDA}*std) | "
            f"Combined Acc={combined_acc:.4f} | "
            f"Combined F1(macro)={combined_f1_macro:.4f}"
        )

        score = float(rec[RANK_BY])

        if score > best_score:
            best_score = score
            best_avg_f1 = avg_f1_macro
            best_std_f1 = std_f1_macro
            best_arch_id = spec.id

    sort_keys = [RANK_BY, "avg_val_f1_macro", "combined_f1_macro"]
    df = pd.DataFrame(results).sort_values(by=sort_keys, ascending=False)

    out_csv = (
        f"orientation_classifier_search_{SEARCH_MODE.lower()}"
        f"_V1to{MAX_VERSIONS}_rankscore.csv"
    )

    df.to_csv(out_csv, index=False)

    out_f1_txt = os.path.splitext(out_csv)[0] + "_f1_report.txt"
    write_f1_version_report_txt(results, out_f1_txt, max_version=20)

    print(f"\nResults saved to {out_csv}")
    print(f"F1 summary report (txt): {out_f1_txt}")

    print(
        f"Best Arch by {RANK_BY}: {best_arch_id} "
        f"with {RANK_BY}={best_score:.4f} "
        f"(avg_val_f1_macro={best_avg_f1:.4f}, "
        f"std_val_f1_macro={best_std_f1:.4f})"
    )

    print("\nTraining best model on all data...")

    best_spec = next(s for s in specs if s.id == best_arch_id)
    best_model = build_model_from_spec(best_spec, IN_DIM, OUT_DIM).cuda()

    train_final_model(best_model, X, y)

    score_str = f"{best_score:.4f}".replace(".", "_")
    dynamic_model_path = (
        f"orientation_classifier_{best_spec.id}"
        f"_{RANK_BY}_{score_str}.pt"
    )

    torch.save(best_model.state_dict(), dynamic_model_path)

    meta = {
        "model_path": dynamic_model_path,
        "best_arch_id": best_arch_id,
        "rank_by": RANK_BY,
        "rank_score": best_score,
        "avg_val_f1_macro": best_avg_f1,
        "std_val_f1_macro": best_std_f1,
        "std_penalty_lambda": STD_PENALTY_LAMBDA,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "n_seeds_per_fold": N_SEEDS_PER_FOLD,
        "use_mixup": USE_MIXUP,
        "mixup_alpha": MIXUP_ALPHA,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "class_names": CLASS_NAMES,
        "class_counts": np.bincount(y_np, minlength=OUT_DIM).tolist(),
    }

    meta_path = dynamic_model_path.replace(".pt", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved best model as {dynamic_model_path}")
    print(f"Saved metadata as {meta_path}")


if __name__ == "__main__":
    main()
