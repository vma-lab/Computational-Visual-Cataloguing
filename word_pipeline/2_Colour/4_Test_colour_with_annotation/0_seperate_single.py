import os
import json
import copy
from collections import defaultdict


# =========================
# Config
# =========================
INPUT_JSON = "instances_test.json"
OUTPUT_DIR = "jsons"

RESET_IMAGE_ID_TO_ZERO = True      # giống file HS01309379_0295.json: image id = 0
KEEP_ORIGINAL_ANNOTATION_ID = True # giữ nguyên annotation id trong test.json
INDENT = 2


# =========================
# Helper functions
# =========================
def load_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find input JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=INDENT)


def get_stem(file_name):
    """
    HS01309379_0295.jpg -> HS01309379_0295
    """
    return os.path.splitext(os.path.basename(file_name))[0]


def group_annotations_by_image_id(annotations):
    grouped = defaultdict(list)

    for ann in annotations:
        image_id = ann.get("image_id")
        grouped[image_id].append(ann)

    return grouped


def build_single_image_coco(
    full_data,
    image_info,
    image_annotations,
    reset_image_id_to_zero=True,
    keep_original_annotation_id=True
):
    """
    Build one COCO-style JSON file for a single image.
    """

    original_image_id = image_info["id"]

    # Copy image info
    new_image = copy.deepcopy(image_info)

    if reset_image_id_to_zero:
        new_image["id"] = 0
        new_image_id = 0
    else:
        new_image_id = original_image_id

    # Copy annotations of this image only
    new_annotations = []

    for new_idx, ann in enumerate(image_annotations, start=1):
        new_ann = copy.deepcopy(ann)

        # Reset image_id to match the single-image JSON
        new_ann["image_id"] = new_image_id

        # Optional: reset annotation id from 1..N
        if not keep_original_annotation_id:
            new_ann["id"] = new_idx

        new_annotations.append(new_ann)

    # Build output structure
    single_data = {
        "info": copy.deepcopy(
            full_data.get(
                "info",
                {
                    "description": "COCO Text Detection (Single Image)",
                    "version": "1.0",
                    "year": 2025,
                    "contributor": "Auto-generated",
                    "date_created": "2025-02-05",
                },
            )
        ),
        "licenses": copy.deepcopy(full_data.get("licenses", [])),
        "images": [new_image],
        "annotations": new_annotations,
        "categories": copy.deepcopy(full_data.get("categories", [])),
    }

    return single_data


# =========================
# Main
# =========================
def main():
    data = load_json(INPUT_JSON)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    if not images:
        raise ValueError("No images found in input JSON.")

    if not annotations:
        print("Warning: No annotations found in input JSON.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    annotations_by_image_id = group_annotations_by_image_id(annotations)

    total_images = len(images)
    total_annotations = len(annotations)

    print("=" * 80)
    print(f"Input JSON      : {INPUT_JSON}")
    print(f"Output folder   : {OUTPUT_DIR}")
    print(f"Total images    : {total_images}")
    print(f"Total annotations: {total_annotations}")
    print("=" * 80)

    written_files = 0

    for image_info in images:
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        image_annotations = annotations_by_image_id.get(image_id, [])

        single_data = build_single_image_coco(
            full_data=data,
            image_info=image_info,
            image_annotations=image_annotations,
            reset_image_id_to_zero=RESET_IMAGE_ID_TO_ZERO,
            keep_original_annotation_id=KEEP_ORIGINAL_ANNOTATION_ID,
        )

        output_name = get_stem(file_name) + ".json"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        save_json(single_data, output_path)

        written_files += 1

        print(
            f"[{written_files:03d}/{total_images:03d}] "
            f"Saved: {output_path} | "
            f"image_id={image_id} | "
            f"annotations={len(image_annotations)}"
        )

    print("=" * 80)
    print(f"Done. Created {written_files} JSON files in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
