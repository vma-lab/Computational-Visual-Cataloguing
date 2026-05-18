import os
import json
import copy


# ============================================================
# Paths
# ============================================================
TEST_JSON_PATH = "instances_test.json"
COLORED_JSON_FOLDER = "ColoredJson_Output"
OUTPUT_JSON_PATH = "Ouput_colour.json"


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
    Read all JSON files in ColoredJson_Output and build:

        color_info_by_ann_id[annotation_id] = {
            "color_name": color_name,
            "color_code": color_code
        }

    IMPORTANT:
    - Color information must come ONLY from ColoredJson_Output.
    - Do NOT use color_name/color_code already existing in instances_test.json.
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

            # Only collect color info from ColoredJson_Output
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
    """
    Update annotations in instances_test.json using color info from ColoredJson_Output.

    IMPORTANT:
    - If instances_test.json already has color_name/color_code, ignore them.
    - Always overwrite color_name/color_code using ColoredJson_Output if matched by annotation id.
    - If not matched, assign DEFAULT_COLOR_NAME and DEFAULT_COLOR_CODE.
    """

    updated_data = copy.deepcopy(test_data)

    annotations = updated_data.get("annotations", [])

    matched = 0
    missing = 0
    overwritten = 0
    missing_ann_ids = []

    for ann in annotations:
        ann_id = ann.get("id")

        # Check whether instances_test.json already had old color info
        had_old_color_info = (
            "color_name" in ann or
            "color_code" in ann
        )

        if ann_id in color_info_by_ann_id:
            color_info = color_info_by_ann_id[ann_id]

            # Always overwrite using ColoredJson_Output
            ann["color_name"] = color_info.get("color_name", DEFAULT_COLOR_NAME)
            ann["color_code"] = color_info.get("color_code", DEFAULT_COLOR_CODE)

            matched += 1

            if had_old_color_info:
                overwritten += 1

        else:
            # Do NOT keep old color info from instances_test.json
            # Replace it with default unknown
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
    print("Update instances_test.json with color_name and color_code")
    print("Source of color info: ColoredJson_Output ONLY")
    print("=" * 80)

    if not os.path.exists(TEST_JSON_PATH):
        raise FileNotFoundError(
            f"Cannot find test JSON: {TEST_JSON_PATH}"
        )

    test_data = load_json(TEST_JSON_PATH)

    color_info_by_ann_id, json_files, duplicated_ids = collect_color_info_from_folder(
        COLORED_JSON_FOLDER
    )

    updated_data, matched, missing, overwritten, missing_ann_ids = update_test_json_with_color_info(
        test_data=test_data,
        color_info_by_ann_id=color_info_by_ann_id,
    )

    save_json(updated_data, OUTPUT_JSON_PATH)

    total_test_annotations = len(test_data.get("annotations", []))
    total_color_annotations = len(color_info_by_ann_id)

    print(f"Input test JSON             : {TEST_JSON_PATH}")
    print(f"Colored JSON folder         : {COLORED_JSON_FOLDER}")
    print(f"Colored JSON files found    : {len(json_files)}")
    print(f"Color infos found           : {total_color_annotations}")
    print(f"Test annotations            : {total_test_annotations}")
    print(f"Matched annotations         : {matched}")
    print(f"Overwritten old color infos : {overwritten}")
    print(f"Missing color infos         : {missing}")
    print(f"Output JSON                 : {OUTPUT_JSON_PATH}")

    if duplicated_ids:
        print()
        print(f"Warning: duplicated annotation ids found in ColoredJson_Output: {len(duplicated_ids)}")
        print("First duplicated ids:", duplicated_ids[:20])

    if missing_ann_ids:
        print()
        print("Warning: some annotations in instances_test.json did not receive color info from ColoredJson_Output.")
        print("Their color_name/color_code were set to unknown.")
        print("First missing annotation ids:", missing_ann_ids[:30])

    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
