"""Fine-tune a YOLOv11 detector for primary-page detection in notebook scans."""

from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the YOLO dataset YAML (produced by create_yaml.py).",
    )
    parser.add_argument(
        "--model",
        default="yolo11x.pt",
        help="YOLO weights to fine-tune from. Default: yolo11x.pt.",
    )
    parser.add_argument(
        "--project",
        default="./Results/",
        help="Project directory under which training runs are written.",
    )
    parser.add_argument(
        "--name",
        default="DLA_pages",
        help="Name of this training run (subfolder under --project).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device id (e.g. 'cuda:0') or 'cpu'. Default: auto-select.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--optimizer", default="AdamW")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = YOLO(args.model).to(device)
    model.info()

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        save=True,
        device=device,
        resume=False,
        workers=args.workers,
        save_period=args.save_period,
        cache=False,
        name=args.name,
        project=args.project,
        lr0=args.lr0,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        patience=args.patience,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=0.2,
        mixup=0.0,
        scale=0.2,
    )


if __name__ == "__main__":
    main()
