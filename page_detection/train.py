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

    # Build a YOLOv9c model from pretrained weight
    model = YOLO("models/yolo11x.pt").to(DEVICE)

    # Display model information (optional)
    model.info()

    # Train the model
    results = model.train(
        data=r'E:/Projects_PyCharm/YOLO/data/Drangsong/one_class/Drangsong.yaml',
        epochs=100,
        imgsz=2048,
        batch=2,
        save=True,
        device=DEVICE,
        resume=False,
        workers=0,
        save_period=10,
        cache=False,
        name="YOLOv11_Drangsong_phase2",
        project="./Results/",

        # Learning Rate
        lr0=0.0001,
        optimizer='Adam',

        # Freezing Layers
        # freeze=list(range(0, 4)),  # Freezes layers 0 to 5,

        # Early Stopping
        patience=10,

        # Recommended Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        scale=0.2
    )


if __name__ == '__main__':
    train_yolo()
