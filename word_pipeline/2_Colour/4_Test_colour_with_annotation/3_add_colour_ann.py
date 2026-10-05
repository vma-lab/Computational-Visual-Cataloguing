"""
Use instances_test.json only as a COCO scaffold (same ids/structure as GT).
Write PREDICTED color_name/color_code from ColoredJson_Output onto that
scaffold -> Ouput_colour.json, so pred and GT share format for
4_performance_colour.py.
"""

import os
import json
import copy


# ============================================================
# Paths
# ============================================================
TEST_JSON_PATH = "instances_test.json"  # COCO scaffold only
COLORED_JSON_FOLDER = "ColoredJson_Output"  # predicted colour from 1_colour_patch.py
OUTPUT_JSON_PATH = "Ouput_colour.json"  # prediction JSON for metrics


# ============================================================
# Options
# ============================================================
DEFAULT_COLOR_NAME = "unknown"
DEFAULT_COLOR_CODE = "unknown"


# ============================================================
# Helper functions
# ============================================================
def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collect_color_info_from_folder(colored_json_folder):
    """
    Collect PREDICTED color_name/color_code from ColoredJson_Output
    """
    if not os.path.exists(colored_json_folder):
        raise FileNotFoundError(
            f"Cannot find colored JSON folder: {colored_json_folder}"
        )

    json_files = [
        f for f in os.listdir(colored_json_folder)
        if f.lower().endswith(".json")
    ]

    color_info_by_ann_id = {}
    duplicated_ids = []

    for json_file in json_files:
        json_path = os.path.join(colored_json_folder, json_file)
        data = load_json(json_path)

        annotations = data.get("annotations", [])

        for ann in annotations:
            ann_id = ann.get("id")

            if ann_id is None:
                continue

            has_color_name = "color_name" in ann
            has_color_code = "color_code" in ann

            if not has_color_name and not has_color_code:
                continue

            if ann_id in color_info_by_ann_id:
                duplicated_ids.append(ann_id)

            color_info_by_ann_id[ann_id] = {
                "color_name": ann.get("color_name", DEFAULT_COLOR_NAME),
                "color_code": ann.get("color_code", DEFAULT_COLOR_CODE),
            }

    return color_info_by_ann_id, json_files, duplicated_ids


def update_test_json_with_color_info(test_data, color_info_by_ann_id):

    updated_data = copy.deepcopy(test_data)

    annotations = updated_data.get("annotations", [])

    matched = 0
    missing = 0
    overwritten = 0
    missing_ann_ids = []

    for ann in annotations:
        ann_id = ann.get("id")

        had_old_color_info = (
            "color_name" in ann or
            "color_code" in ann
        )

        if ann_id in color_info_by_ann_id:
            color_info = color_info_by_ann_id[ann_id]

            ann["color_name"] = color_info.get("color_name", DEFAULT_COLOR_NAME)
            ann["color_code"] = color_info.get("color_code", DEFAULT_COLOR_CODE)

            matched += 1

            if had_old_color_info:
                overwritten += 1

        else:
            ann["color_name"] = DEFAULT_COLOR_NAME
            ann["color_code"] = DEFAULT_COLOR_CODE

            missing += 1
            missing_ann_ids.append(ann_id)

    return updated_data, matched, missing, overwritten, missing_ann_ids


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 80)
    print("Merge predicted colour onto COCO scaffold -> Ouput_colour.json")
    print("Predicted source: ColoredJson_Output")
    print("=" * 80)

    if not os.path.exists(TEST_JSON_PATH):
        raise FileNotFoundError(
            f"Cannot find test JSON: {TEST_JSON_PATH}"
        )

    test_data = load_json(TEST_JSON_PATH)

    color_info_by_ann_id, json_files, duplicated_ids = collect_color_info_from_folder(
        COLORED_JSON_FOLDER
    )

    updated_data, matched, missing, _, missing_ann_ids = update_test_json_with_color_info(
        test_data=test_data,
        color_info_by_ann_id=color_info_by_ann_id,
    )

    save_json(updated_data, OUTPUT_JSON_PATH)

    total_test_annotations = len(test_data.get("annotations", []))
    total_color_annotations = len(color_info_by_ann_id)

    print(f"Scaffold JSON               : {TEST_JSON_PATH}")
    print(f"Predicted colour folder     : {COLORED_JSON_FOLDER}")
    print(f"Predicted JSON files        : {len(json_files)}")
    print(f"Predicted colour entries    : {total_color_annotations}")
    print(f"Unmatched (set to unknown)  : {missing}")
    print(f"Output prediction JSON      : {OUTPUT_JSON_PATH}")

    if duplicated_ids:
        print()
        print(f"Warning: duplicated annotation ids in ColoredJson_Output: {len(duplicated_ids)}")
        print("First duplicated ids:", duplicated_ids[:20])

    if missing_ann_ids:
        print()
        print("Warning: unmatched scaffold ids (color set to unknown).")
        print("First unmatched ids:", missing_ann_ids[:30])

    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
