import os
import re
import torch
import numpy as np
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


# ===================== Config =====================
MODEL_PATH = "writing_tool_classifier_V09_trained_on_all_training_data.pt"
TEST_PATH = "writing_tool_features_clipL14_all.npz"

IN_DIM = 768
OUT_DIM = 2
BATCH_SIZE = 64

CLASS_NAMES = ["pcl", "ink"]
L2_NORMALIZE_FEATURES = True

VERSION: Optional[int] = None


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


def parse_version_from_path(path: str) -> Optional[int]:
    name = os.path.basename(path).lower()
    m = re.search(r"v(\d+)", name)
    if m is None:
        return None
    return int(m.group(1))


def state_dict_has_bn(state: dict) -> bool:
    return any(".running_mean" in k or ".running_var" in k for k in state.keys())


def load_state_dicts(path: str, device: str) -> List[dict]:
    """One state dict per seed ({"state_dicts": [...]}) or a single state dict."""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dicts" in ckpt:
        return list(ckpt["state_dicts"])
    return [ckpt]


# ===================== Main =====================
def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {MODEL_PATH}")

    if not os.path.isfile(TEST_PATH):
        raise FileNotFoundError(f"TEST_PATH does not exist: {TEST_PATH}")

    parsed_version = parse_version_from_path(MODEL_PATH)
    chosen_version = VERSION if VERSION is not None else parsed_version

    if chosen_version is None:
        raise ValueError(
            "Cannot infer version from MODEL_PATH. "
            "Set VERSION = 9 or rename checkpoint with V09/v09."
        )

    if VERSION is not None and parsed_version is not None and VERSION != parsed_version:
        print(
            f"Warning: checkpoint name suggests V{parsed_version:02d}, "
            f"but VERSION is forced to V{VERSION:02d}."
        )

    spec = find_spec_by_version(chosen_version)

    print(f"Using checkpoint: {MODEL_PATH}")
    print(
        f"Architecture: {spec.id} | depth={spec.depth} | widths={spec.widths} | "
        f"BN={spec.use_bn} | dropout={spec.dropout} | act={spec.activation}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model(s): softmax probabilities are averaged over the seeds
    models = []
    for state in load_state_dicts(MODEL_PATH, device):
        ckpt_has_bn = state_dict_has_bn(state)
        if ckpt_has_bn != spec.use_bn:
            raise ValueError(
                f"Architecture mismatch: checkpoint "
                f"{'has' if ckpt_has_bn else 'has no'} BatchNorm, "
                f"but {spec.id} "
                f"{'expects' if spec.use_bn else 'does not use'} BatchNorm."
            )
        model = build_model_from_spec(spec, IN_DIM, OUT_DIM).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    print(f"Models in checkpoint (seeds): {len(models)}")

    # Load data
    data = np.load(TEST_PATH, allow_pickle=True)

    X = torch.tensor(data["features"], dtype=torch.float32)

    if L2_NORMALIZE_FEATURES:
        X = nn.functional.normalize(X, p=2, dim=1)

    X = X.to(device)

    y = torch.tensor(data["labels"], dtype=torch.long).cpu().numpy()
    filenames = data["filenames"] if "filenames" in data.files else None

    print(f"Test samples: {len(X)}")
    print(f"Feature dim: {X.shape[1]}")

    # Predict
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i + BATCH_SIZE]
            probs = torch.stack([m(xb).softmax(dim=1) for m in models]).mean(dim=0)
            preds.extend(probs.argmax(dim=1).cpu().numpy())

    preds_np = np.asarray(preds)

    # Metrics
    acc = accuracy_score(y, preds_np)
    f1_macro = f1_score(y, preds_np, average="macro", zero_division=0)
    f1_per_class = f1_score(
        y,
        preds_np,
        average=None,
        labels=list(range(OUT_DIM)),
        zero_division=0,
    )

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1_macro:.4f}")
    print(
        "F1 per class: "
        + " ".join(
            f"{name}={score:.4f}"
            for name, score in zip(CLASS_NAMES, f1_per_class)
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y, preds_np, labels=list(range(OUT_DIM)))

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
    plt.title(f"Testing set | Acc={acc:.4f} | Macro-F1={f1_macro:.4f}")
    plt.tight_layout()

    ckpt_stem = os.path.splitext(os.path.basename(MODEL_PATH))[0].lower()
    cm_path = f"confusion_matrix_{ckpt_stem}.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Confusion matrix saved to: {cm_path}")

    # Save predictions
    if filenames is not None:
        csv_path = f"test_predictions_{ckpt_stem}.csv"

        pd.DataFrame({
            "filename": list(filenames),
            "true_id": y,
            "true_label": [CLASS_NAMES[i] for i in y],
            "predicted_id": preds_np,
            "predicted_label": [CLASS_NAMES[i] for i in preds_np],
        }).to_csv(csv_path, index=False)

        print(f"Predictions saved to: {csv_path}")


if __name__ == "__main__":
    main()
