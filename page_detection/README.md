# Page Detection (part of CVC)

This folder implements the **page-detection** stage of CVC (Computational Visual Cataloguing). Given a raw notebook scan (which may contain a partial opposing page and other background), it detects the primary content-bearing page so that it can be cropped and passed to downstream processing.

It is **one of three components** in the parent repository, alongside [`../word_pipeline/`](../word_pipeline) and [`../scriptsight/`](../scriptsight). See the top-level [`../README.md`](../README.md) for the overall CVC system and the paper citation.

## Method

A one-stage **YOLOv11s** detector ([Ultralytics](https://docs.ultralytics.com/)) is fine-tuned from large-scale pre-training on a small set of annotated notebook images, with a single `page` class and images resized to 640 px. Each image contains one primary page, so only the highest-scoring detection per image is kept. Evaluation follows a notebook-level protocol: the training length is selected with 5-fold cross-validation on the training images, with all pages of a notebook in the same fold, and the final model is assessed once on a held-out test set, with no notebook shared between training and test. With the selected training length (10 epochs), the five folds reach mAP@50 = 0.9769 ± 0.0363 and mAP@50-95 = 0.8937 ± 0.1491 (mean ± standard deviation); on the held-out test set the final model reaches mAP@50 = 0.9950 and mAP@50-95 = 0.8594.

## Files

- `train_test_dla.py`: a single, self-contained script that runs the whole page-detection workflow — notebook-level k-fold cross-validation on the training set to select the number of training epochs, a final model trained on all training images for the selected number of epochs, and evaluation on the held-out test set. It writes the YOLO dataset YAMLs and fold lists automatically and saves a `summary.json` with the cross-validation and test metrics, the selected number of epochs and the final weights path.

All configuration is set in the `CONFIG` block at the top of the script; there are no command-line arguments.

## Usage

```bash
# 1) Arrange the dataset as two folders, each with images/ and labels/ (YOLO format):
#      <data>/train/images/*.png   <data>/train/labels/*.txt
#      <data>/test/images/*.png    <data>/test/labels/*.txt

# 2) Edit the CONFIG block at the top of train_test_dla.py
#    (TRAIN_DIR, TEST_DIR, MODEL, IMGSZ, K, BATCH, CANDIDATE_EPOCHS, SEED, RESULTS, SPLITS_DIR).

# 3) Run the full pipeline: k-fold cross-validation (epoch selection) -> final training -> test evaluation.
python train_test_dla.py
```

## Configuration and reproducibility

The defaults baked into the `CONFIG` block are:

- **Model**: `yolo11s.pt`; **image size**: 640; **batch**: 16.
- **Cross-validation**: `K = 5` notebook-level folds (all pages of a notebook in the same fold; ≈64 train / ≈16 val per fold); **random seed**: 42; single class (`page`). The script stops if a notebook appears in both the training and the test set.
- **Model selection**: each candidate training length in `CANDIDATE_EPOCHS = [10, 20, 30, 50]` is trained on every fold with early stopping disabled and scored at its last epoch; the candidate with the highest mean mAP@50-95 is selected.
- **Final model**: trained on all training images for the selected number of epochs, with early stopping disabled; the deployment weights are `last.pt`.
- **Detections**: one box per image (`max_det = 1`) in cross-validation, test evaluation and inference.
- **Augmentation**: horizontal flip 0.5, translate 0.05, scale 0.1 and HSV jitter; no rotation, mosaic, mixup, shear, perspective or vertical flip.

## Inputs and outputs

- **Input**: full notebook scans (including partial opposing pages and other background) with YOLO-format `page` labels, arranged as `train/` and `test/` folders, each containing `images/` and `labels/`. File names must start with the notebook ID followed by an underscore (e.g. `HS01309362_0003.png`; for `Rilke_nb_T15_Auszug_Page2.png` the notebook ID is `T15`), because the folds are built per notebook.
- **Output**:
  - trained detector weights at `Results_page_cv_select/final_all_train/weights/last.pt` (the exact path is recorded in `summary.json`);
  - the per-fold runs of every candidate, the final-training run and the test evaluation under `Results_page_cv_select/`;
  - the generated fold lists and dataset YAMLs under `page_kfold_splits_cv_select/`;
  - annotated test predictions for manual inspection (one box per image);
  - a `summary.json` recording the cross-validation and test metrics, the selected number of epochs and the final weights path.

## Inference (producing page crops)

Apply the final weights to new scans to obtain the primary-page crops that feed [`../word_pipeline/`](../word_pipeline). With Ultralytics this can be done directly; `max_det=1 conf=0.001` keeps exactly the highest-scoring box per scan, as in the evaluation:

```bash
yolo predict model=Results_page_cv_select/final_all_train/weights/last.pt \
             source=/path/to/notebook_scans max_det=1 conf=0.001 save_crop=True
```

Place the resulting page crops under `word_pipeline/1_textdetection/data/DLA/img/`.
