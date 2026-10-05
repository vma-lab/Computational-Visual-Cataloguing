"""
Quantitative colour-name evaluation: predicted vs ground-truth coarse names.

PRED_JSON (Ouput_colour.json): predictions from 1_colour_patch.py via 3_add_colour_ann.py.
GT_JSON (instances_test.json): ground-truth colour names for scoring.
Both use the field name color_name; sources differ (pred file vs GT file).

Primary Macro F1: excluding the zero-support yellow class
(mean F1 over classes with non-zero ground-truth support only).
Macro F1: including zero-support yellow as F1=0
(mean F1 over all labels present in the eval set).
Balanced accuracy: over non-zero-support classes only.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =========================
# CONFIG
# =========================
PRED_JSON = "Ouput_colour.json"       # predicted colour names
GT_JSON = "instances_test.json"       # ground-truth colour names

OUTPUT_DIR = "colour_eval_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Same key name in both files; values come from different JSONs (pred vs GT).
PRED_COLOR_KEY = "color_name"
GT_COLOR_KEY = "color_name"
MATCH_KEY = "id"

CLASS_NAMES = [
    "black",
    "blue",
    "gray",
    "yellow",
    "red",
    "purple",
    "brown",
    "green",
    "orange",
    "pink",
    "white",
]


# =========================
# HELPER FUNCTIONS
# =========================
def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(label):
 
    if label is None:
        return None

    label = str(label).strip().lower()

    mapping = {
        "grey": "gray",
        "bluish": "blue",
        "yellowish": "yellow",
        "reddish": "red",
        "purplish": "purple",
        "brownish": "brown",
        "blackish": "black",
        "whitish": "white",
    }

    return mapping.get(label, label)


def annotations_to_dict(data, color_key, source_name):
    anns = data.get("annotations", [])

    out = {}
    duplicated_ids = []

    for ann in anns:
        ann_id = ann.get(MATCH_KEY)

        if ann_id is None:
            continue

        if ann_id in out:
            duplicated_ids.append(ann_id)

        out[ann_id] = {
            "id": ann_id,
            "color_name": normalize_label(ann.get(color_key)),
            "color_code": ann.get("color_code", ""),
            "image_id": ann.get("image_id", ""),
            "source": source_name,
        }

    if duplicated_ids:
        print(
            f"Warning: duplicated annotation IDs in {source_name}: "
            f"{len(duplicated_ids)}"
        )

    return out


def save_confusion_matrix_seaborn(
    y_true,
    y_pred,
    labels,
    class_names,
    output_path,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    plt.figure(figsize=(max(6, len(class_names) * 0.8), max(5, len(class_names) * 0.65)))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Colour Name Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return cm


# =========================
# LOAD DATA
# =========================
pred_data = load_json(PRED_JSON)
gt_data = load_json(GT_JSON)

pred_dict = annotations_to_dict(
    pred_data,
    PRED_COLOR_KEY,
    "prediction"
)

gt_dict = annotations_to_dict(
    gt_data,
    GT_COLOR_KEY,
    "groundtruth"
)

pred_ids = set(pred_dict.keys())
gt_ids = set(gt_dict.keys())

common_ids = sorted(pred_ids & gt_ids)
missing_in_pred = sorted(gt_ids - pred_ids)
missing_in_gt = sorted(pred_ids - gt_ids)

print("===== Matching Info =====")
print(f"Prediction annotations : {len(pred_ids)}")
print(f"Groundtruth annotations: {len(gt_ids)}")
print(f"Matched annotations    : {len(common_ids)}")
print(f"Missing in prediction  : {len(missing_in_pred)}")
print(f"Missing in groundtruth : {len(missing_in_gt)}")


# =========================
# BUILD EVAL TABLE
# =========================
rows = []

for ann_id in common_ids:
    pred_label = pred_dict[ann_id]["color_name"]
    gt_label = gt_dict[ann_id]["color_name"]

    if pred_label is None or gt_label is None:
        continue

    rows.append({
        "id": ann_id,
        "gt": gt_label,
        "pred": pred_label,
        "correct": gt_label == pred_label,
        "pred_color_code": pred_dict[ann_id].get("color_code", ""),
        "gt_color_code": gt_dict[ann_id].get("color_code", ""),
        "image_id": gt_dict[ann_id].get("image_id", ""),
    })

df = pd.DataFrame(rows)

if len(df) == 0:
    raise ValueError(
        "No valid matched annotations found. "
        "Please check annotation IDs and color_name keys."
    )

print(f"\nValid evaluated samples: {len(df)}")


# =========================
# LABEL ORDER
# =========================
existing_labels = sorted(set(df["gt"]) | set(df["pred"]))

labels = [c for c in CLASS_NAMES if c in existing_labels]
labels += [c for c in existing_labels if c not in labels]

print("\nLabels used:")
print(labels)


# =========================
# METRICS
# =========================
y_true = df["gt"].values
y_pred = df["pred"].values

accuracy = accuracy_score(y_true, y_pred)

# Include zero-support yellow (and any other predicted-only class): F1=0 in the average
macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=labels,
    average="macro",
    zero_division=0
)

# Exclude zero-support yellow: only classes with non-zero ground-truth support
labels_gt_supported = sorted(set(df["gt"]))

macro_f1_primary = f1_score(
    y_true,
    y_pred,
    labels=labels_gt_supported,
    average="macro",
    zero_division=0
)

# Balanced accuracy over non-zero-support classes only
balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    labels=labels,
    average="weighted",
    zero_division=0
)

print("\n===== Overall Metrics =====")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Primary Macro F1 (excluding zero-support yellow): {macro_f1_primary:.4f}")
print(f"Macro F1 (including zero-support yellow as F1=0): {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Balanced accuracy over non-zero-support classes: {balanced_acc:.4f}")


# =========================
# REPORT
# =========================
report_text = classification_report(
    y_true,
    y_pred,
    labels=labels,
    digits=4,
    zero_division=0
)

print("\n===== Classification Report =====")
print(report_text)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=labels,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()
report_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
report_df.to_csv(report_path)


# =========================
# CONFUSION MATRIX CSV
# =========================
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"GT_{x}" for x in labels],
    columns=[f"PRED_{x}" for x in labels]
)

cm_csv_path = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
cm_df.to_csv(cm_csv_path)

print("\n===== Confusion Matrix =====")
print(cm_df)


# =========================
# SEABORN CONFUSION MATRIX PNG
# =========================
cm_png_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix_seaborn_style.png"
)

save_confusion_matrix_seaborn(
    y_true=y_true,
    y_pred=y_pred,
    labels=labels,
    class_names=labels,
    output_path=cm_png_path,
)


# =========================
# WRONG PREDICTIONS
# =========================
wrong_df = df[df["correct"] == False].copy()

wrong_path = os.path.join(
    OUTPUT_DIR,
    "wrong_predictions.csv"
)

wrong_df.to_csv(wrong_path, index=False)

eval_table_path = os.path.join(
    OUTPUT_DIR,
    "all_eval_results.csv"
)

df.to_csv(eval_table_path, index=False)


# =========================
# SUMMARY (same content as terminal output)
# =========================
summary_path = os.path.join(
    OUTPUT_DIR,
    "summary.txt"
)

saved_files_block = (
    f"All eval results           : {eval_table_path}\n"
    f"Wrong predictions          : {wrong_path}\n"
    f"Classification report      : {report_path}\n"
    f"Confusion matrix CSV       : {cm_csv_path}\n"
    f"Confusion matrix PNG       : {cm_png_path}\n"
    f"Summary                    : {summary_path}\n"
)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("===== Matching Info =====\n")
    f.write(f"Prediction annotations : {len(pred_ids)}\n")
    f.write(f"Groundtruth annotations: {len(gt_ids)}\n")
    f.write(f"Matched annotations    : {len(common_ids)}\n")
    f.write(f"Missing in prediction  : {len(missing_in_pred)}\n")
    f.write(f"Missing in groundtruth : {len(missing_in_gt)}\n")
    f.write(f"\nValid evaluated samples: {len(df)}\n")
    f.write("\nLabels used:\n")
    f.write(f"{labels}\n")
    f.write("\n===== Overall Metrics =====\n")
    f.write(f"Accuracy   : {accuracy:.4f}\n")
    f.write(
        f"Primary Macro F1 (excluding zero-support yellow): {macro_f1_primary:.4f}\n"
    )
    f.write(
        f"Macro F1 (including zero-support yellow as F1=0): {macro_f1:.4f}\n"
    )
    f.write(f"Weighted F1: {weighted_f1:.4f}\n")
    f.write(
        f"Balanced accuracy over non-zero-support classes: {balanced_acc:.4f}\n"
    )
    f.write("\n===== Classification Report =====\n")
    f.write(report_text)
    if not report_text.endswith("\n"):
        f.write("\n")
    f.write("\n===== Confusion Matrix =====\n")
    f.write(cm_df.to_string())
    f.write("\n")
    f.write("\n===== Saved Files =====\n")
    f.write(saved_files_block)
    f.write("\nDone.\n")


print("\n===== Saved Files =====")
print(saved_files_block, end="")
print("\nDone.")
