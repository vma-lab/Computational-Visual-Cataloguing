# Page Detection (part of CVC)

This folder implements the **page-detection** stage of CVC (Computational Visual Cataloguing). Given a raw notebook scan (which may contain a partial opposing page and other background), it detects the primary content-bearing page(s) and outputs cropped page images for downstream processing. See CVC paper §4.1.

It is **one of three components** in the parent repository, alongside [`../word_pipeline/`](../word_pipeline) and [`../scriptsight/`](../scriptsight). See the top-level [`../README.md`](../README.md) for the overall CVC system and the paper citation.

## Method

A one-stage **YOLOv11** detector ([Ultralytics](https://docs.ultralytics.com/)) is fine-tuned from large-scale pre-training on a small set of annotated notebook images, with a single `page` class and images resized to 1024 px. On the validation split the model reaches precision = 0.985, recall = 1.0, mAP@50 = 0.990, mAP@50-95 = 0.983.

## Files

- `create_yaml.py`: generates the YOLO dataset YAML from a folder containing `train/`, `val/`, `test/` splits and a `classes.txt`.
- `train.py`: fine-tunes the YOLOv11x detector on the page-detection dataset.
- `predict.py`: runs inference on a folder of notebook scans, writing cropped primary pages plus a `detections.csv` summary.

## Inputs and outputs

- **Input**: full notebook scans, including partial opposing pages and other background.
- **Output**: cropped primary-page images (one or more per scan), plus a CSV of normalised bounding boxes. These crops are the expected input for [`../word_pipeline/`](../word_pipeline).
