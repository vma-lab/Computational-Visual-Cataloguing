import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional


# ============== Config ==============
# Same training recipe as the cross-validation search in
# ../2_search_architecture/2_train_classifier_writing_tool_V11.py, so the
# retrained model is the model that was evaluated.
EPOCHS = 30
BATCH_SIZE = 64
IN_DIM = 768
OUT_DIM = 2

SEED = 42
DATA_PATH = "writing_tool_features_clipL14_all.npz"

CLASS_NAMES = ["pcl", "ink"]

# Best RankScore in ../2_search_architecture/writing_tool_mlp_arch_search_V01-V20_rankscore_report.txt
BEST_VERSION = 9

USE_CLASS_WEIGHTED_CE = True
L2_NORMALIZE_FEATURES = True
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
USE_COSINE_LR = True

USE_MIXUP = True
MIXUP_ALPHA = 0.3

USE_EMA = True
EMA_DECAY = 0.999

# Same as orientation: one model per seed; inference averages softmax probabilities.
N_SEEDS = 3

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============== Architecture catalogue ==============
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

    if len(spec.widths) != spec.depth:
        raise ValueError(
            f"{spec.id}: depth={spec.depth} != len(widths)={len(spec.widths)}"
        )

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

    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def compute_class_weights(
    y: torch.Tensor,
    n_classes: int,
    device: str,
) -> Optional[torch.Tensor]:
    if not USE_CLASS_WEIGHTED_CE:
        return None

    counts = np.bincount(y.cpu().numpy(), minlength=n_classes).astype(np.float64)
    total = counts.sum()

    weights = np.zeros(n_classes, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = total / (n_classes * counts[nonzero])

    return torch.tensor(weights, dtype=torch.float32, device=device)


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
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}


def seed_list(base_seed: int = SEED) -> List[int]:
    return [base_seed + 7919 * s for s in range(N_SEEDS)]


def train_one_seed(
    spec: ArchSpec,
    X: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Fixed number of epochs on all data; returns the final (EMA) weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model_from_spec(spec, IN_DIM, OUT_DIM).to(device)

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    class_weights = compute_class_weights(y, OUT_DIM, device)
    ce = nn.functional.cross_entropy

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

    ema = ModelEMA(model) if USE_EMA else None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if USE_MIXUP and MIXUP_ALPHA > 0:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm = torch.randperm(xb.size(0), device=xb.device)
                logits = model(lam * xb + (1.0 - lam) * xb[perm])
                loss = (
                    lam * ce(logits, yb, weight=class_weights)
                    + (1.0 - lam) * ce(logits, yb[perm], weight=class_weights)
                )
            else:
                logits = model(xb)
                loss = ce(logits, yb, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema is not None:
                ema.update(model)

            total_loss += float(loss.item())
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
            avg_loss = total_loss / max(1, n_batches)
            print(f"  seed={seed} epoch {epoch:03d}/{EPOCHS} | loss={avg_loss:.6f}")

    if ema is not None:
        return ema.state_dict()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print("Loading data...")

    data = np.load(DATA_PATH, allow_pickle=True)

    X = torch.tensor(data["features"], dtype=torch.float32)

    if L2_NORMALIZE_FEATURES:
        X = nn.functional.normalize(X, p=2, dim=1)

    y = torch.tensor(data["labels"], dtype=torch.long)
    y_np = y.numpy()

    class_counts = np.bincount(y_np, minlength=OUT_DIM).tolist()

    print(f"Samples: {len(X)}")
    print(f"Class names: {CLASS_NAMES}")
    print(f"Class counts: {class_counts}")

    spec = find_spec_by_version(BEST_VERSION)

    print(f"\nTraining on ALL data with architecture: {spec.id}")
    print(
        f"Spec: depth={spec.depth}, widths={spec.widths}, "
        f"use_bn={spec.use_bn}, dropout={spec.dropout}, activation={spec.activation}"
    )

    print("Training on all augmented training data (no validation split).")

    seeds = seed_list()
    state_dicts = [train_one_seed(spec, X, y, seed=s, device=device) for s in seeds]

    model_path = f"writing_tool_classifier_{spec.id}_trained_on_all_training_data.pt"
    torch.save({"state_dicts": state_dicts}, model_path)

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
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "input_dim": IN_DIM,
        "output_dim": OUT_DIM,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "use_cosine_lr": USE_COSINE_LR,
        "use_class_weighted_ce": USE_CLASS_WEIGHTED_CE,
        "use_mixup": USE_MIXUP,
        "mixup_alpha": MIXUP_ALPHA,
        "use_ema": USE_EMA,
        "ema_decay": EMA_DECAY,
        "l2_normalize_features": L2_NORMALIZE_FEATURES,
        "seeds": seeds,
        "data_path": DATA_PATH,
        "class_names": CLASS_NAMES,
        "class_counts": class_counts,
    }

    meta_path = model_path.replace(".pt", ".meta.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved final model: {model_path} ({len(state_dicts)} seeds)")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
