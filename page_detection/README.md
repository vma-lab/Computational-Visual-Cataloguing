# Page Detection (part of CVC)

This folder implements the **page-detection** stage of CVC (Computational Visual Cataloguing). Given a raw notebook scan (which may contain a partial opposing page and other background), it detects the primary content-bearing page(s) and outputs cropped page images for downstream processing. See CVC paper §4.1.

It is **one of three components** in the parent repository, alongside [`../word_pipeline/`](../word_pipeline) and [`../scriptsight/`](../scriptsight). See the top-level [`../README.md`](../README.md) for the overall CVC system and the paper citation.

## Method

A one-stage **YOLOv11** detector ([Ultralytics](https://docs.ultralytics.com/)) is fine-tuned from large-scale pre-training on a small set of annotated notebook images, with a single `page` class and images resized to 1024 px. On the validation split the model reaches precision = 0.985, recall = 1.0, mAP@50 = 0.990, mAP@50-95 = 0.983.

## Files

- `create_yaml.py`: generates the YOLO dataset YAML from a folder containing `train/`, `val/`, `test/` splits and a `classes.txt`.
- `train.py`: fine-tunes the YOLOv11x detector on the page-detection dataset.
- `predict.py`: runs inference on a folder of notebook scans, writing cropped primary pages plus a `detections.csv` summary.

All three scripts are driven by command-line arguments; run any of them with `-h` to see the full option list.

## Usage

```bash
# 1) Build the YOLO dataset YAML from your annotated split.
python create_yaml.py --data-dir /path/to/yolo_dataset_root

# 2) Fine-tune YOLOv11x on the page-detection task.
python train.py --data /path/to/yolo_dataset_root/dataset.yaml \
                --model yolo11x.pt \
                --project ./Results/ --name DLA_pages

# 3) Run inference on a folder of notebook scans and save crops + detections.csv.
python predict.py --model ./Results/DLA_pages/weights/best.pt \
                  --source /path/to/notebook_scans \
                  --output /path/to/cropped_pages
```

The crops produced by `predict.py` are the expected input for [`../word_pipeline/`](../word_pipeline) (place them under `word_pipeline/1_textdetection/data/DLA/img/`).

## Inputs and outputs

- **Input**: full notebook scans, including partial opposing pages and other background.
- **Output**: cropped primary-page images (one or more per scan), plus a CSV of normalised bounding boxes. These crops are the expected input for [`../word_pipeline/`](../word_pipeline).
