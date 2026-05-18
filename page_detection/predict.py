"""Run a trained YOLOv11 page detector on a folder of notebook scans.

For each image, primary-page detections are written to a CSV of normalised
bounding boxes. The top-N detections can also be saved as cropped page images
(enabled by default) and the annotated detection visualisation can be saved.
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=UserWarning)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained YOLO weights (e.g. DLA_pages.pt).",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Directory of notebook scans (recursed) or a single image path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where crops, annotated images and detections.csv are written.",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument(
        "--max-det",
        type=int,
        default=3,
        help="Maximum detections per image passed to the detector.",
    )
    parser.add_argument(
        "--num-detections",
        type=int,
        default=1,
        help="Top-N detections per image to write to CSV and save as crops.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Extra pixels added around each crop's bounding box.",
    )
    parser.add_argument(
        "--no-save-crops",
        dest="save_crops",
        action="store_false",
        help="Disable saving cropped page images (crops are saved by default).",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Also save annotated visualisation images alongside crops.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device id (e.g. 'cuda:0') or 'cpu'. Default: auto-select.",
    )
    parser.set_defaults(save_crops=True)
    return parser.parse_args()


def gather_image_paths(source: str) -> list[str]:
    src_path = Path(source)
    if src_path.is_file():
        return [str(src_path)]

    paths: list[str] = []
    for root, _, files in os.walk(source):
        for file in files:
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(root, file))
    return paths


def save_crops(result, output_dir: str, num_detections: int, padding: int) -> None:
    if not result.boxes:
        return

    orig_img = result.orig_img
    img_h, img_w = orig_img.shape[:2]
    boxes = sorted(result.boxes, key=lambda b: float(b.conf[0]), reverse=True)

    for i, box in enumerate(boxes[:num_detections]):
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)
        cropped = orig_img[y1:y2, x1:x2]

        base = os.path.splitext(os.path.basename(result.path))[0]
        out_path = os.path.join(output_dir, f"{base}_crop_{i + 1}.jpg")
        cv2.imwrite(out_path, cropped)


def save_detections_csv(result, csv_writer, num_detections: int) -> None:
    if not result.boxes:
        return

    img_h, img_w = result.orig_img.shape[:2]
    boxes = sorted(result.boxes, key=lambda b: float(b.conf[0]), reverse=True)

    for box in boxes[:num_detections]:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cls_id = int(box.cls[0])
        x_c = ((x1 + x2) / 2) / img_w
        y_c = ((y1 + y2) / 2) / img_h
        w_n = (x2 - x1) / img_w
        h_n = (y2 - y1) / img_h
        csv_writer.writerow(
            [os.path.basename(result.path), cls_id, x_c, y_c, w_n, h_n]
        )


def save_annotated(result, output_dir: str) -> None:
    img = result.plot()
    parent = os.path.basename(os.path.dirname(result.path))
    name = os.path.basename(result.path)
    out_path = os.path.join(output_dir, f"{parent}_{name}" if parent else name)
    cv2.imwrite(out_path, img)


def main() -> None:
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    image_paths = gather_image_paths(args.source)
    if not image_paths:
        raise SystemExit(f"No images found under: {args.source}")
    print(f"Found {len(image_paths)} image(s) under {args.source}")

    model = YOLO(args.model).to(device)

    csv_path = os.path.join(args.output, "detections.csv")
    with open(csv_path, "w", newline="") as csv_f:
        csv_w = csv.writer(csv_f)
        csv_w.writerow(
            [
                "image_name",
                "class_id",
                "x_center_norm",
                "y_center_norm",
                "width_norm",
                "height_norm",
            ]
        )

        for image_path in image_paths:
            results = model.predict(
                source=image_path,
                save=False,
                save_txt=False,
                save_conf=False,
                save_crop=False,
                show_labels=True,
                show_conf=True,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=device,
                verbose=True,
            )

            for result in results:
                if not result.boxes:
                    continue
                save_detections_csv(result, csv_w, args.num_detections)
                if args.save_crops:
                    save_crops(result, args.output, args.num_detections, args.padding)
                if args.save_annotated:
                    save_annotated(result, args.output)

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
