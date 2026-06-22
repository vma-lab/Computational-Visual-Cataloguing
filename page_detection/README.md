# Page Detection (part of CVC)

This folder implements the **page-detection** stage of CVC (Computational Visual Cataloguing). Given a raw notebook scan (which may contain a partial opposing page and other background), it detects the primary content-bearing page(s) so they can be cropped and passed to downstream processing.

It is **one of three components** in the parent repository, alongside [`../word_pipeline/`](../word_pipeline) and [`../scriptsight/`](../scriptsight). See the top-level [`../README.md`](../README.md) for the overall CVC system and the paper citation.

## Method

A one-stage **YOLOv11s** detector ([Ultralytics](https://docs.ultralytics.com/)) is fine-tuned from large-scale pre-training on a small set of annotated notebook images, with a single `page` class and images resized to 640 px. Evaluation follows a notebook-level protocol: the model is selected with 5-fold cross-validation on the training images and assessed on a held-out test set, with no notebook shared between training and test. Across the five folds it reaches mAP@50 = 0.9645 ± 0.0257 and mAP@50-95 = 0.8452 ± 0.0367 (mean ± standard deviation); on the held-out test set it reaches mAP@50 = 0.9926 and mAP@50-95 = 0.8263.

## Files

- `train_test_dla.py`: a single, self-contained script that runs the whole page-detection workflow — k-fold cross-validation on the training set, a final model trained on all training images for a CV-informed number of epochs, and evaluation on the held-out test set. It writes the YOLO dataset YAMLs and fold lists automatically and saves a `summary.json` with the cross-validation and test metrics and the final weights path.

All configuration is set in the `CONFIG` block at the top of the script; there are no command-line arguments.

## Usage

```bash
# 1) Arrange the dataset as two folders, each with images/ and labels/ (YOLO format):
#      <data>/train/images/*.png   <data>/train/labels/*.txt
#      <data>/test/images/*.png    <data>/test/labels/*.txt

# 2) Edit the CONFIG block at the top of train_test_dla.py
#    (TRAIN_DIR, TEST_DIR, MODEL, IMGSZ, K, BATCH, SEED, RESULTS, SPLITS_DIR).

# 3) Run the full pipeline: k-fold cross-validation -> final training -> test evaluation.
python train_test_dla.py
```

## Configuration and reproducibility

The defaults baked into the `CONFIG` block are:

- **Model**: `yolo11s.pt`; **image size**: 640; **batch**: 16.
- **Cross-validation**: `K = 5` folds (≈64 train / ≈16 val per fold); **random seed**: 42; single class (`page`).
- **Final model**: trained on all training images for a CV-informed budget (mean best epoch across folds, minimum 30), with early stopping disabled; the deployment weights are `last.pt`.
- **Augmentation**: horizontal flip 0.5, rotation ±10°, translate 0.1, scale 0.5, mosaic 1.0 (closed for the last 10 epochs) and HSV jitter; no mixup, shear, perspective or vertical flip.

## Inputs and outputs

- **Input**: full notebook scans (including partial opposing pages and other background) with YOLO-format `page` labels, arranged as `train/` and `test/` folders, each containing `images/` and `labels/`.
- **Output**:
  - trained detector weights at `Results_page_smodel/final_all_train/weights/last.pt`;
  - the per-fold runs, the final-training run and the test evaluation under `Results_page_smodel/`;
  - the generated fold lists and dataset YAMLs under `page_kfold_splits_smodel/`;
  - annotated test predictions for manual inspection;
  - a `summary.json` recording the cross-validation and test metrics and the final weights path.

## Inference (producing page crops)

Apply the final weights to new scans to obtain the primary-page crops that feed [`../word_pipeline/`](../word_pipeline). With Ultralytics this can be done directly:

```bash
yolo predict model=Results_page_smodel/final_all_train/weights/last.pt \
             source=/path/to/notebook_scans save_crop=True
```

Place the resulting page crops under `word_pipeline/1_textdetection/data/DLA/img/`.
