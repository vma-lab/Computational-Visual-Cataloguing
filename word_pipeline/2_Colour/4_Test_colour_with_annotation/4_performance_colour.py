import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =========================
# CONFIG
# =========================
PRED_JSON = "Ouput_colour.json"
GT_JSON = "instances_test.json"

OUTPUT_DIR = "colour_eval_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    acc=None,
    f1_macro=None,
    normalize=False,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    if normalize:
        cm_display = cm.astype(float)
        row_sums = cm_display.sum(axis=1, keepdims=True)

        cm_display = np.divide(
            cm_display,
            row_sums,
            out=np.zeros_like(cm_display),
            where=row_sums != 0
        )

        fmt = ".2f"
        title_prefix = "Normalized Colour Name Confusion Matrix"
    else:
        cm_display = cm
        fmt = "d"
        title_prefix = "Colour Name Confusion Matrix"

    plt.figure(figsize=(max(6, len(class_names) * 0.8), max(5, len(class_names) * 0.65)))

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    extras = []

    if acc is not None:
        extras.append(f"acc={acc:.4f}")

    if f1_macro is not None:
        extras.append(f"f1={f1_macro:.4f}")

    title = title_prefix

    if extras:
        title += " | " + ", ".join(extras)

    plt.title(title)
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

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=labels,
    average="macro",
    zero_division=0
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
print(f"Macro F1   : {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")


# =========================
# REPORT
# =========================
report_text = classification_report(
    y_true,
    y_pred,
    labels=labels,
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
    acc=accuracy,
    f1_macro=macro_f1,
    normalize=False,
)

cm_norm_png_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix_normalized_seaborn_style.png"
)

save_confusion_matrix_seaborn(
    y_true=y_true,
    y_pred=y_pred,
    labels=labels,
    class_names=labels,
    output_path=cm_norm_png_path,
    acc=accuracy,
    f1_macro=macro_f1,
    normalize=True,
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
# SUMMARY
# =========================
summary_path = os.path.join(
    OUTPUT_DIR,
    "summary.txt"
)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Colour Name Evaluation Summary\n")
    f.write("==============================\n\n")
    f.write(f"Prediction JSON : {PRED_JSON}\n")
    f.write(f"Groundtruth JSON: {GT_JSON}\n\n")
    f.write(f"Prediction annotations : {len(pred_ids)}\n")
    f.write(f"Groundtruth annotations: {len(gt_ids)}\n")
    f.write(f"Matched annotations    : {len(common_ids)}\n")
    f.write(f"Valid evaluated samples: {len(df)}\n\n")
    f.write(f"Accuracy   : {accuracy:.4f}\n")
    f.write(f"Macro F1   : {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    f.write("Labels:\n")

    for label in labels:
        f.write(f"- {label}\n")


print("\n===== Saved Files =====")
print(f"All eval results           : {eval_table_path}")
print(f"Wrong predictions          : {wrong_path}")
print(f"Classification report      : {report_path}")
print(f"Confusion matrix CSV       : {cm_csv_path}")
print(f"Confusion matrix PNG       : {cm_png_path}")
print(f"Normalized confusion PNG   : {cm_norm_png_path}")
print(f"Summary                    : {summary_path}")

print("\nDone.")
