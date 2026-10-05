"""Generate a YOLO dataset YAML from a folder containing train/val/test splits."""

from __future__ import annotations

import argparse
import os

import yaml


def generate_yaml(data_dir: str, output_yaml: str) -> None:
    sections = ["train", "val", "test"]
    data_dict = {"path": data_dir}

    class_file_path = os.path.join(data_dir, "classes.txt")
    if os.path.exists(class_file_path):
        with open(class_file_path, "r") as class_file:
            classes = [line.strip() for line in class_file if line.strip()]
        print(f"Read classes from {class_file_path}: {classes}")
    else:
        print(f"{class_file_path} not found; deriving classes from label files.")
        class_ids = set()
        for section in sections:
            label_path = os.path.join(data_dir, section, "labels")
            if not os.path.exists(label_path):
                continue
            for file in os.listdir(label_path):
                with open(os.path.join(label_path, file), "r") as f:
                    for line in f:
                        parts = line.split()
                        if parts:
                            class_ids.add(parts[0])
        classes = sorted(class_ids, key=int)
        print(f"Classes derived from label files: {classes}")

    data_dict["nc"] = len(classes)
    data_dict["names"] = classes

    for section in sections:
        image_path = os.path.abspath(os.path.join(data_dir, section, "images"))
        if os.path.exists(image_path):
            data_dict[section] = image_path.replace("\\", "/")

    with open(output_yaml, "w") as file:
        yaml.dump(data_dict, file, default_flow_style=False)

    print(f"YAML configuration written to {output_yaml}")
    print(data_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root of the YOLO dataset (contains train/, val/, test/, classes.txt).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output YAML path. Defaults to <data-dir>/dataset.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or os.path.join(args.data_dir, "dataset.yaml")
    generate_yaml(args.data_dir, output)


if __name__ == "__main__":
    main()
