import os

from ultralytics import YOLO
import torch

# Set CUDA devices visible to PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# Clears CUDA memory
torch.cuda.empty_cache()


def train_yolo():
    # Build a YOLOv9c model from scratch
    # model = YOLO(" .yaml")

    # Build a YOLOv11x model from pretrained weight
    model = YOLO("models/yolo11x.pt").to(DEVICE)

    # Display model information (optional)
    model.info()

    # Train the model
    results = model.train(
        data=r'./YOLO/data/DLA/DLA.yaml',
        epochs=100,
        imgsz=1024,
        batch=2,
        save=True,
        device=DEVICE,
        resume=False,
        workers=0,
        save_period=10,
        cache=False,
        name="DLA_pages",
        project="./Results/",

        # Learning Rate
        lr0=0.0001,
        optimizer='AdamW',
        weight_decay=0.0005,

        # Freezing Layers
        # freeze=list(range(0, 4)),  # Freezes layers 0 to 5,

        # Early Stopping
        patience=10,

        # Recommended Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=0.2,
        mixup=0.0,
        scale=0.2
    )


if __name__ == '__main__':
    train_yolo()
