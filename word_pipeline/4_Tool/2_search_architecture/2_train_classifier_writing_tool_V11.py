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
TASK_NAME = "writing_tool"
CLASS_NAMES = ["pcl", "ink"]   # 0=pencil, 1=ink

FOLDS = 5
EPOCHS = 50
BATCH_SIZE = 64
IN_DIM = 768
OUT_DIM = len(CLASS_NAMES)
CM_SAVE_DIR = f"confusion_matrices_{TASK_NAME}"
DATA_PATH = f"{TASK_NAME}_features_clipL14_all.npz"
SEED = 42

MAX_VERSIONS = 20
SEARCH_MODE = "ORIGINAL_30"

USE_CLASS_WEIGHTED_CE = True
L2_NORMALIZE_FEATURES = True
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
USE_COSINE_LR = True
EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 12
RANK_BY = "combined_f1_macro"

USE_MIXUP = True
MIXUP_ALPHA = 0.3

N_SEEDS_PER_FOLD = 5
N_SEEDS_FINAL_MODEL = 1

USE_FOCAL_LOSS = False
FOCAL_GAMMA = 1.0
LABEL_SMOOTHING = 0.0

USE_EMA = True
EMA_DECAY = 0.999

TUNE_THRESHOLDS = True
REPORT_TOP_K_ENSEMBLE = 1

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


def build_model_from_spec(spec: ArchSpec, in_dim=IN_DIM, out_dim=OUT_DIM) -> nn.Module:
    layers: List[nn.Module] = []

    if spec.depth == 0:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    hidden = spec.widths[0]

    layers.append(nn.Linear(in_dim, hidden))
    if spec.use_bn:
        layers.append(nn.BatchNorm1d(hidden))
    layers.append(_act(spec.activation))
    if spec.dropout and spec.dropout > 0:
        layers.append(nn.Dropout(spec.dropout))

    for _ in range(1, spec.depth):
        layers.append(nn.Linear(hidden, hidden))
        if spec.use_bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(_act(spec.activation))
        if spec.dropout and spec.dropout > 0:
            layers.append(nn.Dropout(spec.dropout))

    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def save_confusion_matrix(y_true, y_pred, arch_id, fold, acc=None, f1_macro=None, path_prefix=CM_SAVE_DIR):
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


def loss_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor],
    *,
    use_focal: bool = USE_FOCAL_LOSS,
    gamma: float = FOCAL_GAMMA,
    label_smoothing: float = LABEL_SMOOTHING,
) -> torch.Tensor:
    n_classes = logits.size(1)
    log_probs = nn.functional.log_softmax(logits, dim=1)

    if label_smoothing > 0:
        with torch.no_grad():
            true_dist = torch.full_like(
                log_probs,
                label_smoothing / max(1, n_classes - 1),
            )
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
    else:
        true_dist = nn.functional.one_hot(targets, n_classes).float()

    if use_focal and gamma > 0:
        focal_mod = (1.0 - log_probs.exp()) ** gamma
        per_class = -(true_dist * focal_mod * log_probs).sum(dim=1)
    else:
        per_class = -(true_dist * log_probs).sum(dim=1)

    if weight is not None:
        sample_w = weight[targets]
        return (sample_w * per_class).sum() / sample_w.sum().clamp(min=1e-8)

    return per_class.mean()


def _is_bn_running_stat(key: str) -> bool:
    return (
        key.endswith("running_mean")
        or key.endswith("running_var")
        or key.endswith("num_batches_tracked")
    )


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            new_v = msd[k].detach()
            if v.dtype.is_floating_point and not _is_bn_running_stat(k):
                v.mul_(self.decay).add_(new_v, alpha=1.0 - self.decay)
            else:
                v.copy_(new_v)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def tune_thresholds_oof(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_classes: int,
    *,
    grid: Optional[np.ndarray] = None,
    n_sweeps: int = 3,
) -> Tuple[np.ndarray, float]:
    if grid is None:
        grid = np.linspace(-3.0, 3.0, 61)

    log_p = np.log(np.clip(probs, 1e-8, 1.0))
    biases = np.zeros(n_classes, dtype=np.float64)

    def macro_f1(b: np.ndarray) -> float:
        preds = np.argmax(log_p + b, axis=1)
        return float(
            f1_score(
                y_true,
                preds,
                average="macro",
                labels=list(range(n_classes)),
                zero_division=0,
            )
        )

    best = macro_f1(biases)

    for _ in range(n_sweeps):
        improved = False
        for c in range(1, n_classes):
            best_b = biases[c]
            for b_c in grid:
                trial = biases.copy()
                trial[c] = b_c
                f = macro_f1(trial)
                if f > best + 1e-6:
                    best = f
                    best_b = b_c
                    improved = True
            biases[c] = best_b

        if not improved:
            break

    return biases, best


@torch.no_grad()
def _eval_logits(model, val_loader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []
    gts_list: List[np.ndarray] = []

    for xb, yb in val_loader:
        xb = xb.cuda()
        out = model(xb).cpu().numpy()
        logits_list.append(out)
        gts_list.append(yb.numpy())

    return np.concatenate(logits_list, axis=0), np.concatenate(gts_list, axis=0)


def _f1_from_preds(gts: np.ndarray, preds: np.ndarray) -> Tuple[float, float, np.ndarray]:
    acc = float(accuracy_score(gts, preds))
    f1_mac = float(f1_score(gts, preds, average="macro"))
    f1_per_class = f1_score(
        gts,
        preds,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
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

    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None
    eval_model = build_model_from_spec(spec, IN_DIM, OUT_DIM).cuda() if ema else None

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_f1 = -1.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.cuda(), yb.cuda()

            if USE_MIXUP and MIXUP_ALPHA > 0:
                mx, ya, yb_perm, lam = _mixup_batch(xb, yb, MIXUP_ALPHA)
                pred = model(mx)
                loss = (
                    lam * loss_fn(pred, ya, class_weights)
                    + (1.0 - lam) * loss_fn(pred, yb_perm, class_weights)
                )
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb, class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema is not None:
                ema.update(model)

        if scheduler is not None:
            scheduler.step()

        if ema is not None and eval_model is not None:
            ema.apply_to(eval_model)
            logits, gts = _eval_logits(eval_model, val_loader)
        else:
            logits, gts = _eval_logits(model, val_loader)

        preds = logits.argmax(axis=1)
        f1_mac = float(f1_score(gts, preds, average="macro"))

        if f1_mac > best_f1 + 1e-6:
            best_f1 = f1_mac
            src = ema.state_dict() if ema is not None else model.state_dict()
            best_state = {k: v.detach().cpu().clone() for k, v in src.items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if EARLY_STOPPING and epochs_no_improve >= EARLY_STOP_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logits, gts = _eval_logits(model, val_loader)
    return logits, gts


def train_one_fold(spec: ArchSpec, X, y, train_idx, val_idx, base_seed: int):
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

    return (
        acc,
        f1_mac,
        f1_per_class,
        preds.tolist(),
        gts_ref.tolist(),
        avg_probs,
    )


def _train_final_one_seed(
    spec: ArchSpec,
    X: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    n_epochs: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model_from_spec(spec, IN_DIM, OUT_DIM).cuda()

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
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    for _ in range(epochs):
        model.train()

        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda()

            if USE_MIXUP and MIXUP_ALPHA > 0:
                mx, ya, yb_perm, lam = _mixup_batch(xb, yb, MIXUP_ALPHA)
                pred = model(mx)
                loss = (
                    lam * loss_fn(pred, ya, class_weights)
                    + (1.0 - lam) * loss_fn(pred, yb_perm, class_weights)
                )
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb, class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema is not None:
                ema.update(model)

        if scheduler is not None:
            scheduler.step()

    return ema.state_dict() if ema is not None else {
        k: v.detach().clone() for k, v in model.state_dict().items()
    }


def train_final_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: Optional[int] = None,
    *,
    spec: Optional[ArchSpec] = None,
    n_seeds: int = N_SEEDS_FINAL_MODEL,
    base_seed: int = SEED + 99991,
) -> nn.Module:
    if spec is None:
        sd = _train_final_one_seed(
            spec=_ad_hoc_spec_from_model(model),
            X=X,
            y=y,
            seed=base_seed,
            n_epochs=n_epochs,
        )
        model.load_state_dict(sd)
        return model

    if n_seeds <= 1:
        sd = _train_final_one_seed(
            spec=spec,
            X=X,
            y=y,
            seed=base_seed,
            n_epochs=n_epochs,
        )
        model.load_state_dict(sd)
        return model

    best_sd: Optional[Dict[str, torch.Tensor]] = None
    best_f1 = -1.0
    eval_model = build_model_from_spec(spec, IN_DIM, OUT_DIM).cuda()

    for s in range(n_seeds):
        sd = _train_final_one_seed(spec, X, y, base_seed + 7919 * s, n_epochs)
        eval_model.load_state_dict(sd)
        eval_model.eval()

        with torch.no_grad():
            preds = []
            X_dev = X.cuda()
            for i in range(0, len(X_dev), BATCH_SIZE):
                preds.append(
                    eval_model(X_dev[i:i + BATCH_SIZE]).argmax(dim=1).cpu()
                )
            preds = torch.cat(preds).numpy()

        _, f1m, _ = _f1_from_preds(y.cpu().numpy(), preds)

        print(f"  • seed {s}: train F1(macro)={f1m:.4f}")

        if f1m > best_f1:
            best_f1 = f1m
            best_sd = {k: v.detach().clone() for k, v in sd.items()}

    assert best_sd is not None

    print(f"  → keeping best seed (train F1={best_f1:.4f})")
    model.load_state_dict(best_sd)

    return model


def _ad_hoc_spec_from_model(model: nn.Module) -> ArchSpec:
    return ArchSpec(
        id="ADHOC",
        kind="legacy",
        depth=0,
        widths=[],
        use_bn=False,
        dropout=0.0,
        activation="relu",
    )


def evaluate_on_test(model, test_path):
    if not os.path.exists(test_path):
        print("No test file provided.")
        return

    test_data = np.load(test_path, allow_pickle=True)
    X_test = torch.tensor(test_data["features"], dtype=torch.float32).cuda()
    y_test = torch.tensor(test_data["labels"], dtype=torch.long).cpu().numpy()

    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            xb = X_test[i:i + BATCH_SIZE]
            out = model(xb).argmax(dim=1).cpu().numpy()
            preds.extend(out)

    acc = accuracy_score(y_test, preds)
    f1_mac = f1_score(y_test, preds, average="macro")

    print(f"Test Accuracy: {acc:.4f} | F1(macro): {f1_mac:.4f}")

    save_confusion_matrix(
        y_test,
        preds,
        arch_id="TEST",
        fold="test",
        acc=acc,
        f1_macro=f1_mac,
    )


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
        raise ValueError("Unknown SEARCH_MODE")

    return specs


def compute_rankscore(avg_f1_macro: float, std_f1_macro: float) -> float:
    """RankScore used only for TXT report: Avg F1(macro) - 0.5 * Std."""
    return float(avg_f1_macro) - 0.5 * float(std_f1_macro)


def write_f1_version_report_txt(
    results: List[Dict[str, Any]],
    out_path: str,
    max_version: int = 20,
) -> None:
    """
    Export TXT report.

    IMPORTANT:
    This function only changes the TXT report format.
    It adds RankScore shown as:
        0.9269 - 0.5*0.0193 = 0.9173

    Training, CSV export, model selection, and saved model logic remain unchanged.
    """

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

    lines.append(f"=== F1 REPORT ({TASK_NAME} classifier) ===")
    lines.append(
        f"SEARCH_MODE={SEARCH_MODE} | "
        f"FOLDS={FOLDS} | "
        f"N_SEEDS_PER_FOLD={N_SEEDS_PER_FOLD}"
    )
    lines.append(f"RANK_BY_FOR_TRAINING_AND_MODEL_SELECTION={RANK_BY}")
    lines.append("TXT_RankScore = Avg F1(macro) - 0.5 * Std F1(macro)")
    lines.append("")

    lines.append("Avg F1(macro) = mean macro F1 across validation folds.")
    lines.append("Std F1(macro) = standard deviation across validation folds.")
    lines.append("Combined F1(macro) = F1 over all merged validation predictions.")
    lines.append("RankScore below is only for TXT reporting.")
    lines.append("")

    header = (
        f"{'Version':<8} | "
        f"{'Avg F1(macro)':>14} | "
        f"{'Std':>8} | "
        f"{'Combined F1':>12} | "
        f"{'RankScore':>34}"
    )

    lines.append(header)
    lines.append("-" * len(header))

    for v in range(1, max_version + 1):
        vid = f"V{v:02d}"
        rec = by_num.get(v)

        if rec is None:
            lines.append(
                f"{vid:<8} | "
                f"{'(missing)':>14} | "
                f"{'-':>8} | "
                f"{'-':>12} | "
                f"{'-':>34}"
            )
            continue

        avg = float(rec["avg_val_f1_macro"])
        std = float(rec["std_val_f1_macro"])
        comb = float(rec["combined_f1_macro"])
        rankscore = compute_rankscore(avg, std)

        lines.append(
            f"{vid:<8} | "
            f"{avg:>14.4f} | "
            f"{std:>8.4f} | "
            f"{comb:>12.4f} | "
            f"{avg:.4f} - 0.5*{std:.4f} = {rankscore:.4f}"
        )

    present = [by_num[v] for v in range(1, max_version + 1) if v in by_num]

    if present:
        lines.append("")
        lines.append("=" * 100)
        lines.append("SUMMARY BY TXT RankScore")
        lines.append("=" * 100)

        best_rankscore = max(
            present,
            key=lambda r: compute_rankscore(
                float(r["avg_val_f1_macro"]),
                float(r["std_val_f1_macro"]),
            ),
        )

        best_comb = max(
            present,
            key=lambda r: float(r["combined_f1_macro"]),
        )

        best_avg = max(
            present,
            key=lambda r: float(r["avg_val_f1_macro"]),
        )

        br_avg = float(best_rankscore["avg_val_f1_macro"])
        br_std = float(best_rankscore["std_val_f1_macro"])
        br_score = compute_rankscore(br_avg, br_std)

        lines.append(
            f"Best by TXT RankScore: {best_rankscore['arch_id']} = "
            f"{br_avg:.4f} - 0.5*{br_std:.4f} = {br_score:.4f}"
        )

        lines.append(
            f"Best Combined F1(macro): "
            f"{best_comb['arch_id']} = "
            f"{float(best_comb['combined_f1_macro']):.4f}"
        )

        lines.append(
            f"Best Avg val F1(macro): "
            f"{best_avg['arch_id']} = "
            f"{float(best_avg['avg_val_f1_macro']):.4f} ± "
            f"{float(best_avg['std_val_f1_macro']):.4f}"
        )

        lines.append("")
        lines.append("=" * 100)
        lines.append("RANKING ORDER BY TXT RankScore")
        lines.append("=" * 100)

        ranked = sorted(
            present,
            key=lambda r: compute_rankscore(
                float(r["avg_val_f1_macro"]),
                float(r["std_val_f1_macro"]),
            ),
            reverse=True,
        )

        rank_header = (
            f"{'Rank':<6} | "
            f"{'Version':<8} | "
            f"{'Avg F1':>10} | "
            f"{'Std':>8} | "
            f"{'RankScore':>10} | "
            f"{'Combined F1':>12}"
        )

        lines.append(rank_header)
        lines.append("-" * len(rank_header))

        for i, rec in enumerate(ranked, start=1):
            avg = float(rec["avg_val_f1_macro"])
            std = float(rec["std_val_f1_macro"])
            comb = float(rec["combined_f1_macro"])
            rankscore = compute_rankscore(avg, std)

            lines.append(
                f"{i:<6} | "
                f"{rec['arch_id']:<8} | "
                f"{avg:>10.4f} | "
                f"{std:>8.4f} | "
                f"{rankscore:>10.4f} | "
                f"{comb:>12.4f}"
            )

        lines.append("")
        lines.append("=" * 100)
        lines.append("DETAILED RANKING BY TXT RankScore")
        lines.append("=" * 100)

        detailed_header = (
            f"{'Rank':<6} | "
            f"{'Version':<8} | "
            f"{'RankScore':>10} | "
            f"{'AvgMacro':>10} | "
            f"{'Std':>8} | "
            f"{'CombMacro':>10} | "
            + " | ".join([f"{c:>10}" for c in CLASS_NAMES])
        )

        lines.append(detailed_header)
        lines.append("-" * len(detailed_header))

        for i, rec in enumerate(ranked, start=1):
            avg = float(rec["avg_val_f1_macro"])
            std = float(rec["std_val_f1_macro"])
            comb = float(rec["combined_f1_macro"])
            rankscore = compute_rankscore(avg, std)

            class_scores = []
            for cname in CLASS_NAMES:
                key = f"combined_f1_{cname}"
                if key in rec:
                    class_scores.append(f"{float(rec[key]):>10.4f}")
                else:
                    class_scores.append(f"{'-':>10}")

            lines.append(
                f"{i:<6} | "
                f"{rec['arch_id']:<8} | "
                f"{rankscore:>10.4f} | "
                f"{avg:>10.4f} | "
                f"{std:>8.4f} | "
                f"{comb:>10.4f} | "
                + " | ".join(class_scores)
            )

        lines.append("")
        lines.append("=" * 100)
        lines.append("ARCHITECTURE DETAILS")
        lines.append("=" * 100)

        arch_header = (
            f"{'Version':<8} | "
            f"{'Kind':>12} | "
            f"{'Depth':>5} | "
            f"{'Widths':>20} | "
            f"{'BN':>4} | "
            f"{'Dropout':>8} | "
            f"{'Activation':>10}"
        )

        lines.append(arch_header)
        lines.append("-" * len(arch_header))

        for rec in ranked:
            lines.append(
                f"{rec['arch_id']:<8} | "
                f"{str(rec.get('kind', '-')):>12} | "
                f"{int(rec['depth']):>5} | "
                f"{str(rec['widths']):>20} | "
                f"{int(rec['use_bn']):>4} | "
                f"{float(rec['dropout']):>8.2f} | "
                f"{str(rec['activation']):>10}"
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
            ranked = np.argsort(
                fold_sizes + rng_local.uniform(0, 1e-3, size=n_splits)
            )
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
        val_mask = np.array(
            [best_mapping[str(g)] == fold_idx for g in groups]
        )

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
            f"'{DATA_PATH}' has no 'filenames' field; re-run feature "
            f"extraction so notebook-aware grouping is available."
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
            "Loss: class-weighted CrossEntropy (per-fold weights, "
            f"global ref weights={[round(float(w), 3) for w in global_weights]})"
        )
    else:
        print("Loss: vanilla CrossEntropy")

    print(
        f"Optim: AdamW lr={LEARNING_RATE} wd={WEIGHT_DECAY} "
        f"cosine_lr={USE_COSINE_LR} epochs={EPOCHS} "
        f"early_stop={EARLY_STOPPING}(patience={EARLY_STOP_PATIENCE}) "
        f"L2_norm_features={L2_NORMALIZE_FEATURES} rank_by={RANK_BY}"
    )

    print(
        f"Aug: mixup={USE_MIXUP}(alpha={MIXUP_ALPHA}) "
        f"seeds_per_fold={N_SEEDS_PER_FOLD}"
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
    best_score_f1: float = -1.0
    best_avg_f1: float = -1.0

    N = len(y_np)
    oof_probs_per_spec: Dict[str, np.ndarray] = {}

    for spec in specs:
        print(
            f"\n{spec.id} | kind={spec.kind} depth={spec.depth} "
            f"widths={spec.widths if spec.widths else '-'} "
            f"BN={spec.use_bn} dropout={spec.dropout} act={spec.activation}"
        )

        fold_accs = []
        fold_f1s = []
        fold_f1_per_class: Dict[str, List[float]] = {n: [] for n in CLASS_NAMES}

        all_preds = []
        all_gts = []

        oof_probs = np.zeros((N, OUT_DIM), dtype=np.float64)

        for fold, (train_idx, val_idx) in enumerate(splits):
            val_counts = np.bincount(y_np[val_idx], minlength=len(CLASS_NAMES))

            print(
                f"  Fold {fold + 1}: "
                f"train samples={len(train_idx)}, "
                f"val samples={len(val_idx)}, "
                f"val counts={val_counts.tolist()}"
            )

            acc, f1_mac, f1_per_class, preds, gts, fold_probs = train_one_fold(
                spec,
                X,
                y,
                train_idx,
                val_idx,
                base_seed=SEED + 1009 * (fold + 1),
            )

            fold_accs.append(acc)
            fold_f1s.append(f1_mac)

            for ci, name in enumerate(CLASS_NAMES):
                fold_f1_per_class[name].append(float(f1_per_class[ci]))

            all_preds.extend(preds)
            all_gts.extend(gts)

            oof_probs[val_idx] = fold_probs

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
                + " ".join(
                    f"{name}={f1_per_class[ci]:.4f}"
                    for ci, name in enumerate(CLASS_NAMES)
                )
            )

        avg_acc = float(np.mean(fold_accs))
        avg_f1_macro = float(np.mean(fold_f1s))
        std_f1_macro = float(np.std(fold_f1s))

        avg_f1_per_class = {
            name: float(np.mean(fold_f1_per_class[name]))
            for name in CLASS_NAMES
        }

        combined_acc = accuracy_score(all_gts, all_preds)
        combined_f1_macro = f1_score(all_gts, all_preds, average="macro")

        combined_f1_per_class = f1_score(
            all_gts,
            all_preds,
            average=None,
            labels=list(range(len(CLASS_NAMES))),
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
        }

        for name in CLASS_NAMES:
            rec[f"avg_val_f1_{name}"] = avg_f1_per_class[name]

        rec["combined_accuracy"] = float(combined_acc)
        rec["combined_f1_macro"] = float(combined_f1_macro)

        for ci, name in enumerate(CLASS_NAMES):
            rec[f"combined_f1_{name}"] = float(combined_f1_per_class[ci])

        if spec.notes:
            rec["notes"] = spec.notes

        results.append(rec)

        print(
            f"Avg Acc={avg_acc:.4f} | "
            f"Avg F1(macro)={avg_f1_macro:.4f} ± {std_f1_macro:.4f} | "
            f"Combined Acc={combined_acc:.4f} | "
            f"Combined F1(macro)={combined_f1_macro:.4f}"
        )

        score = rec.get(RANK_BY, avg_f1_macro)

        if score > best_score_f1:
            best_score_f1 = float(score)
            best_avg_f1 = avg_f1_macro
            best_arch_id = spec.id

        oof_probs_per_spec[spec.id] = oof_probs

    sort_keys = [RANK_BY] if RANK_BY != "avg_val_f1_macro" else []
    sort_keys += ["avg_val_f1_macro", "combined_f1_macro"]

    seen = set()
    sort_keys = [k for k in sort_keys if not (k in seen or seen.add(k))]

    df = pd.DataFrame(results).sort_values(by=sort_keys, ascending=False)

    out_csv = (
        f"{TASK_NAME}_classifier_search_{SEARCH_MODE.lower()}"
        f"_V1to{MAX_VERSIONS}_random_stratified.csv"
    )

    df.to_csv(out_csv, index=False)

    out_f1_txt = os.path.splitext(out_csv)[0] + "_f1_report.txt"
    write_f1_version_report_txt(results, out_f1_txt, max_version=20)

    print(f"\nResults saved to {out_csv}")
    print(f"F1 summary report (txt): {out_f1_txt}")

    print(
        f"Best Arch by {RANK_BY}: {best_arch_id} "
        f"with {RANK_BY}={best_score_f1:.4f} "
        f"(avg_val_f1_macro={best_avg_f1:.4f})"
    )

    ranked = sorted(
        results,
        key=lambda r: r.get("combined_f1_macro", 0.0),
        reverse=True,
    )

    top_ids = [r["arch_id"] for r in ranked[:max(1, REPORT_TOP_K_ENSEMBLE)]]

    best_id = top_ids[0]
    best_oof = oof_probs_per_spec[best_id]

    sidecar: Dict[str, Any] = {
        "best_arch_id": best_id,
        "rank_by": RANK_BY,
        "untuned_combined_f1_macro": float(best_score_f1),
    }

    if TUNE_THRESHOLDS:
        biases, tuned_f1 = tune_thresholds_oof(best_oof, y_np, OUT_DIM)

        tuned_preds = np.argmax(
            np.log(best_oof.clip(1e-8)) + biases,
            axis=1,
        )

        tuned_per_class = f1_score(
            y_np,
            tuned_preds,
            average=None,
            labels=list(range(len(CLASS_NAMES))),
            zero_division=0,
        )

        print(
            f"\nThreshold tuning ({best_id}): biases="
            f"[{', '.join(f'{b:+.2f}' for b in biases)}] "
            f"combined_f1_macro: {best_score_f1:.4f} -> {tuned_f1:.4f}"
        )

        print(
            "   tuned per-class F1: "
            + "  ".join(
                f"{n}={v:.4f}"
                for n, v in zip(CLASS_NAMES, tuned_per_class)
            )
        )

        sidecar["thresholds"] = {
            "biases": biases.tolist(),
            "tuned_combined_f1_macro": float(tuned_f1),
            "tuned_combined_f1_per_class": [float(v) for v in tuned_per_class],
        }

    if len(top_ids) >= 2:
        ens_probs = np.mean(
            [oof_probs_per_spec[sid] for sid in top_ids],
            axis=0,
        )

        ens_preds = ens_probs.argmax(axis=1)
        ens_f1 = f1_score(y_np, ens_preds, average="macro")

        print(
            f"\nTop-{len(top_ids)} ensemble {top_ids}: "
            f"combined_f1_macro={ens_f1:.4f}"
        )

        if TUNE_THRESHOLDS:
            ens_biases, ens_tuned_f1 = tune_thresholds_oof(
                ens_probs,
                y_np,
                OUT_DIM,
            )

            print(
                f"   + threshold tuning: "
                f"[{', '.join(f'{b:+.2f}' for b in ens_biases)}] "
                f"combined_f1_macro={ens_tuned_f1:.4f}"
            )

            sidecar["ensemble"] = {
                "members": top_ids,
                "combined_f1_macro": float(ens_f1),
                "tuned_biases": ens_biases.tolist(),
                "tuned_combined_f1_macro": float(ens_tuned_f1),
            }

        else:
            sidecar["ensemble"] = {
                "members": top_ids,
                "combined_f1_macro": float(ens_f1),
            }

    print(
        f"\nTraining best model on all data "
        f"(multi-seed averaging, n_seeds={N_SEEDS_FINAL_MODEL})..."
    )

    best_spec = next(s for s in specs if s.id == best_arch_id)

    best_model = build_model_from_spec(best_spec, IN_DIM, OUT_DIM).cuda()

    train_final_model(
        best_model,
        X,
        y,
        spec=best_spec,
        n_seeds=N_SEEDS_FINAL_MODEL,
    )

    f1_str = f"{best_score_f1:.4f}".replace(".", "_")

    dynamic_model_path = (
        f"{TASK_NAME}_classifier_{best_spec.id}"
        f"_{RANK_BY}_{f1_str}.pt"
    )

    torch.save(best_model.state_dict(), dynamic_model_path)

    print(f"Saved best model as {dynamic_model_path}")

    sidecar_path = dynamic_model_path.replace(".pt", ".meta.json")

    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Saved metadata + tuned thresholds to {sidecar_path}")


if __name__ == "__main__":
    main()

