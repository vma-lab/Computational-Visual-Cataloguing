# CVC — Word Pipeline

This folder implements the **word pipeline** of CVC (Computational Visual Cataloguing). Given a content-bearing manuscript page, it localises every word and classifies its visual attributes (colour, orientation, writing implement), then writes the result into a structured, searchable catalogue.

It is **one of three components** in the parent repository:

- [`../page_detection/`](../page_detection) — detects primary content-bearing pages in raw notebook scans (CVC paper §4.1).
- **`./` (this folder)** — word detection + visual-attribute classification + visualisation (CVC paper §4.2–§4.4, §4.5, §6).
- [`../scriptsight/`](../scriptsight) — interactive GUI for exploring and filtering the catalogues produced by this pipeline (CVC paper §5).

See the top-level [`../README.md`](../README.md) for the overall CVC system and the paper citation.

This component is designed to learn robustly from densely written manuscript pages and limited human annotations, reflecting the natural challenges of historical handwritten collections (multilingual content, varied orientations and colours, diverse implements, irregular layouts). It was developed and evaluated on **Rainer Maria Rilke's notebooks**.

### Inputs to this pipeline

The word pipeline expects **cropped primary pages** as input (one content-bearing page per image). You can obtain these in two ways:

1. **Run `../page_detection/` first** and feed its cropped outputs into this folder, or
2. **Place pre-cropped pages directly** into `1_textdetection/data/DLA/img/` (see [§3. Download dataset](#3-download-dataset)). The released DLA-RMR dataset already contains the cropped primary pages.

### Internal stages

Concretely, this folder chains four sequential tasks plus a final visualisation stage:

```
1_textdetection  →  2_Colour  →  3_Orn (orientation)  →  4_Tool (writing tool)  →  5_visualization
```

The whole testing pipeline is driven by a **single script**:
[`Run_pipeline_testing_auto_paper.sh`](./Run_pipeline_testing_auto_paper.sh)

---

## Table of Contents

- [Project structure](#project-structure)
- [Quick start (testing)](#quick-start-testing)
- [1. Environment setup](#1-environment-setup)
- [2. Download checkpoints](#2-download-checkpoints)
- [3. Download dataset](#3-download-dataset)
- [4. Run the full testing pipeline](#4-run-the-full-testing-pipeline)
- [5. Visualization results](#5-visualization-results)
- [Training pipeline](#training-pipeline)
- [Pipeline internals](#pipeline-internals)
- [Configuration flags](#configuration-flags)
- [Maintainer](#maintainer)

---

## Project structure

This component follows a strict **5-stage pipeline** (one folder per stage). Tasks 3 (orientation) and 4 (writing tool) additionally share a **5-step internal layout** — `1_augmentation → 2_search_architecture → 3_Train_augmented_data → 4_Test_*_with_annotation → 5_Test_*_without_annotation` — so the research workflow (augment → architecture search → final training → evaluate on labelled test → inference on unlabelled data) is the same for both classifiers.

```
word_pipeline/
├── Run_pipeline_testing_auto_paper.sh         # ← One script runs the whole test pipeline
├── README.md
├── .gitignore
│
├── 1_textdetection/                           # ===== Stage 1: DBNet text detection (OCR-CLIP) =====
│   ├── setup_openmmlab_env.sh                 # one-shot installer (conda env `open-mmlab`, MMOCR 0.4.0)
│   ├── configs/
│   │   ├── _base_/                            # base configs (datasets, models, pipelines, runtime)
│   │   └── textdet/dbnet/config_DLA.py        # main DBNet+CLIP config for the DLA dataset
│   ├── datasets/                              # custom dataset/loader/pipeline/textdet_targets
│   ├── hooks/                                 # detailtext logger + lr_updater
│   ├── ocrclip/                               # OCR-CLIP modules (models, heads, losses, postprocessor, BPE vocab)
│   ├── optimizer/                             # Ranger2020 / Ranger21 optimizers
│   ├── pretrained/                            # CLIP RN50/RN101/ViT-B-16 weights
│   │   ├── RN50.pt                            # required backbone (~256 MB)
│   │   └── download_clip_models.sh
│   ├── data/DLA/                              # ← place dataset here (from UHHCloud)
│   │   ├── img/
│   │   ├── instances_test.json
│   │   └── instances_training.json
│   ├── work_dirs/dbnet/                       # ← place text-detection checkpoints here
│   │   ├── text_detection.pth                 #   final trained model (used at test time)
│   │   └── best_imgur5k_epoch_420.pth         #   pre-trained init for training (load_from)
│   ├── show/text_detection/                   # auto-generated detection outputs (jsons/ + imgs/)
│   ├── augmentation_textdetection_multiprocessing_V4.py
│   ├── dist_test.sh / dist_train.sh           # distributed launchers
│   ├── test_DLA.sh / train_DLA.sh             # entry points
│   ├── test.py / train_ocrclip.py             # underlying Python entries
│   └── __init__.py
│
├── 2_Colour/                                  # ===== Stage 2: Ink-colour extraction (GMM) =====
│   ├── 4_Test_colour_with_annotation/         # evaluation on labelled test set (with metrics)
│   │   ├── 0_seperate_single.py
│   │   ├── 1_colour_patch.py
│   │   ├── 2_vis_img.py
│   │   ├── 3_add_colour_ann.py
│   │   ├── 4_performance_colour.py            # accuracy / per-class metrics
│   │   ├── instances_test.json
│   │   └── run.sh
│   └── 5_Test_Colour_without_annotation/      # production inference (no ground truth required)
│       ├── V8_EnhancedBinarization_MultiAlgo_Gmm2Color_UpdatedJson_ProgressBar.py
│       ├── run.sh
│       ├── jsons/                             # ← detection JSONs (auto-copied by master script)
│       ├── img/  imgs/                        # ← matching images (auto-copied)
│       ├── ColoredJson_Output/                # JSON enriched with per-text colour
│       └── ColoredName_CropedPatch_Output/    # cropped text patches, filename encodes colour
│
├── 3_Orn/                                     # ===== Stage 3: Orientation classifier (hor / vrt / tlt) =====
│   ├── 1_augmentation/                        # step 1: build augmented training set
│   │   ├── Augment_patchForOrientation_v4.py
│   │   ├── aug.sh
│   │   ├── check_folder.py
│   │   └── text_patches_train_80/             # raw 80%-train patches (not in repo)
│   ├── 2_search_architecture/                 # step 2: K-fold architecture search (V01..V11 MLPs)
│   │   ├── 1_extract_features_kfold_V1.py     # CLIP-L/14 feature extraction
│   │   ├── 2_train_classifier_V11.py
│   │   └── train.sh
│   ├── 3_Train_augmented_data/                # step 3: train final classifier on all augmented data
│   │   ├── 1_extract_features_kfold_V1.py
│   │   ├── 2_train_classifier_V3.py           # produces Orn_classifier_v09_*.pt
│   │   ├── train.sh
│   │   └── text_patches_augmented_betterv3_80/  (augmented patches, not in repo)
│   ├── 4_Test_orn_with_annotation/            # step 4: evaluate on 20% labelled test (with metrics)
│   │   ├── 1_extract_features_kfold_V1.py
│   │   ├── 2_test_classifier_V3.py
│   │   └── test.sh
│   └── 5_Test_Orn_without_annotation/         # step 5: production inference for the master pipeline
│       ├── 1_extract_features_testing_MultiBatchThread_Fastest_V4.py
│       ├── 2_test_classifier_without_evaluation_MultiBatchThread_Fastest_V4.py
│       ├── test.sh
│       ├── Orn_classifier_v09_trained_on_all_augmented_training_data.pt  # ← place ORN checkpoint here
│       ├── ColoredJson_Output2Input/          (auto-copied from stage 2)
│       ├── ColoredName_CropedPatch_Output2Input/
│       ├── ColoredOrnedJson_Output/           # JSON + orientation label
│       ├── ColoredOrnedName_Output/           # renamed patches: *_hor / *_vrt / *_tlt
│       └── features_clipL14.npz               # extracted features (regenerated each run)
│
├── 4_Tool/                                    # ===== Stage 4: Writing-tool classifier (pencil / ink) =====
│   ├── 1_augmentation/                        # step 1: augmentation
│   │   ├── 1_augmentation.py
│   │   ├── aug.sh
│   │   └── text_patches_train_80/  (not in repo)
│   ├── 2_search_architecture/                 # step 2: K-fold architecture search (V01..V16 MLPs)
│   │   ├── 1_extract_features_kfold_V1.py
│   │   ├── 2_train_classifier_writing_tool_V11.py
│   │   └── train.sh
│   ├── 3_Train_augmented_data/                # step 3: train final classifier on all augmented data
│   │   ├── 1_extract_features_kfold_V1.py
│   │   ├── 2_train_classifier_V1.py           # produces writing_tool_classifier_V16_*.pt
│   │   ├── train.sh
│   │   └── augmented_patches_tool_v3/  (augmented patches, not in repo)
│   ├── 4_Test_tool_with_annotation/           # step 4: evaluate on 20% labelled test (with metrics)
│   │   ├── 1_extract_features_kfold_V1.py
│   │   ├── 2_test_classifier_V3.py
│   │   └── test.sh
│   └── 5_Test_tool_without_annotation/        # step 5: production inference for the master pipeline
│       ├── 1_extract_features_testing_MultiBatchThread_Fastest_V4.py
│       ├── 2_test_classifier_without_evaluation_MultiBatchThread_Fastest_V4.py
│       ├── test.sh
│       ├── writing_tool_classifier_V16_trained_on_all_training_data.pt   # ← place TOOL checkpoint here
│       ├── ColoredOrnedJson_Output2Input/     (auto-copied from stage 3)
│       ├── ColoredOrnedName_Output2Input/
│       ├── ColoredOrnedTooledJson_Output/     # final per-image JSON: colour + orientation + tool
│       └── features_clipL14.npz
│
└── 5_visualization/                           # ===== Stage 5: Side-by-side prediction vs. GT renders =====
    ├── 0_combine_json.py                      # merge per-image JSONs into Prediction_full.json
    ├── 1_visualize_adaptcolour_predict_v1.py  # render predictions onto images
    ├── 1_visualize_prediction.sh
    ├── 2_visualize_GT.py                      # render ground-truth onto images
    ├── 2_visualize_groundtruth.sh
    ├── 3_output_compare_pred_top_gt_bottom.py # stitch prediction (top) and GT (bottom)
    ├── 3_output_compare_pred_top_gt_bottom.sh
    ├── 4_visualize_all.sh                     # entry point that chains all three steps
    ├── img/                                   # ← matching images (auto-copied)
    ├── ColoredOrnedTooledJson_Output/         # ← per-image JSONs (auto-copied from stage 4)
    ├── GT_updated.json                        # ← ground-truth JSON (auto-copied from dataset)
    ├── Prediction_full.json                   # merged predictions
    ├── output_visualized_predict_full/        # rendered predictions
    ├── output_visualized_GT_updated/          # rendered ground truth
    └── output_compare_pred_top_gt_bottom/     # *** FINAL QUALITATIVE RESULT ***
```

### Naming conventions used across tasks

| Prefix in folder name | Meaning |
|---|---|
| `1_augmentation/` | Data augmentation script that turns the 80%-train split into augmented patches |
| `2_search_architecture/` | K-fold sweep over MLP architectures (V01 … V20) to pick the best head |
| `3_Train_augmented_data/` | Final training of the chosen architecture on **all** augmented data → produces the released checkpoint |
| `4_Test_*_with_annotation/` | Evaluation on the 20%-held-out labelled set (reports accuracy, confusion matrix, …) |
| `5_Test_*_without_annotation/` | Production inference used by `Run_pipeline_testing_auto_paper.sh` (no labels required, outputs JSONs/patches consumed by the next stage) |

---

## Quick start (testing)

```bash
# 1) Environment (only first time)
cd 1_textdetection
bash setup_openmmlab_env.sh
conda activate open-mmlab
cd ..

# 2) Download checkpoints + dataset (see sections below)

# 3) Run end-to-end testing
bash Run_pipeline_testing_auto_paper.sh

# 4) Open results
xdg-open 5_visualization/output_compare_pred_top_gt_bottom/
```

---

## 1. Environment setup

The project uses **MMOCR 0.4.0 + MMDet 2.19.1 + MMCV 1.3.17 + PyTorch 1.8.0 (CUDA 11.1)** on **Python 3.7**. A one-shot installer is provided:

```bash
cd 1_textdetection
bash setup_openmmlab_env.sh
```

This creates a conda env called **`open-mmlab`**. Every sub-script in the pipeline activates this env, so you do **not** need to activate it manually before running the master script.

---

## 2. Download checkpoints

All trained checkpoints are hosted on UHHCloud:

➡ **[checkpoint_model (UHHCloud)](https://cloud.uni-hamburg.de/s/XoZrLXBeL9gGqKi)**

After download, place the files exactly as below:

| File | Destination |
|---|---|
| `text_detection.pth` | `1_textdetection/work_dirs/dbnet/text_detection.pth` |
| `best_imgur5k_epoch_420.pth` (used as `load_from` pre-trained weights) | `1_textdetection/work_dirs/dbnet/best_imgur5k_epoch_420.pth` |
| `Orn_classifier_v09_trained_on_all_augmented_training_data.pt` | `3_Orn/5_Test_Orn_without_annotation/` |
| `writing_tool_classifier_V16_trained_on_all_training_data.pt` | `4_Tool/5_Test_tool_without_annotation/` |

> The CLIP RN50 backbone (`pretrained/RN50.pt`, 256 MB) used by DBNet can be re-downloaded via `1_textdetection/pretrained/download_clip_models.sh` if missing.

The master script automatically searches in both the `5_Test_*_without_annotation/` folders and the `3_Train_augmented_data/` folders for the ORN / TOOL classifiers, so either location works.

---

## 3. Download dataset

➡ **[Dataset_DLA (UHHCloud)](https://cloud.uni-hamburg.de/s/6fqNCGsEMq87bg7)**

Contents:

```
img/                          # all page images
instances_test.json           # COCO-style test annotation
instances_training.json       # COCO-style training annotation
```

Place them under:

```
1_textdetection/data/DLA/
├── img/
├── instances_test.json
└── instances_training.json
```

That is the **only** location you need to fill in for testing — the master script propagates images and JSONs to every downstream stage automatically.

---

## 4. Run the full testing pipeline

From the project root:

```bash
bash Run_pipeline_testing_auto_paper.sh
```

The script executes the following 16 stages:

| Stage | Action |
|---:|---|
| 0 | Sanity-check inputs (dataset, checkpoints, folders) |
| 1 | Run **text detection** (`1_textdetection/test_DLA.sh`) |
| 2 | Locate detection outputs (`show/text_detection/{jsons,imgs}`) |
| 3-4 | Copy detection JSONs + matching images into `2_Colour/` |
| 5 | Run **colour extraction** (GMM-based, V8 script) |
| 6 | Verify colour outputs |
| 7 | Copy colour outputs into `3_Orn/` |
| 8 | Auto-locate and stage the **orientation classifier** checkpoint |
| 9 | Run **ORN test** (`3_Orn/5_Test_Orn_without_annotation/test.sh`) |
| 10 | Copy ORN outputs + CLIP-L/14 features into `4_Tool/` |
| 11 | Auto-locate and stage the **writing-tool classifier** checkpoint |
| 12 | Run **TOOL test** |
| 13-15 | Copy TOOL outputs + images + GT JSON into `5_visualization/` |
| 16 | Run **visualization** (`5_visualization/4_visualize_all.sh`) |

Final logs end with:

```
DONE.
Final comparison output:
  5_visualization/output_compare_pred_top_gt_bottom
```

---

## 5. Visualization results

After the pipeline finishes you will find:

- `5_visualization/Prediction_full.json` — merged predictions
- `5_visualization/output_visualized_predict_full/` — pages with predicted polygons + labels
- `5_visualization/output_visualized_GT_updated/` — pages with ground-truth polygons + labels
- `5_visualization/output_compare_pred_top_gt_bottom/` — side-by-side (prediction on top, GT below) — **this is the main qualitative result**

---

## Training pipeline

For training you also need the **augmented dataset** (the augmented text patches used by the ORN and TOOL classifiers).

➡ **[Augmented_DLA (UHHCloud)](https://cloud.uni-hamburg.de/s/SwNdEaXeki9wdsk)**

After download you should have:

```
Augmented_DLA/
├── 1_TextDetection/
│   ├── img/
│   ├── instances_training.json     # augmented training annotation for DBNet
│   └── instances_test.json
├── 3_Orn/                          # augmented orientation patches (≈40k images)
└── 4_Tool/                         # augmented writing-tool patches (≈38k images)
```

Map each part into the project as below.

| Source (from UHHCloud) | Destination in repo |
|---|---|
| `Augmented_DLA/1_TextDetection/img/` + `instances_*.json` | `1_textdetection/data/DLA/` |
| `Augmented_DLA/3_Orn/`  (patch files) | `3_Orn/3_Train_augmented_data/text_patches_augmented_betterv3_80/` |
| `Augmented_DLA/4_Tool/` (patch files) | `4_Tool/3_Train_augmented_data/augmented_patches_tool_v3/` |

### 1) Text detection

```
1_textdetection/data/DLA/   # img/ + instances_training.json + instances_test.json
```

```bash
cd 1_textdetection
bash train_DLA.sh
```

### 2) Orientation classifier

Place augmented patches under:

```
3_Orn/3_Train_augmented_data/text_patches_augmented_betterv3_80/
```

Then:

```bash
cd 3_Orn/3_Train_augmented_data
bash train.sh
```

(`train.sh` runs `2_train_classifier_V3.py`. Uncomment the feature-extraction step `python 1_extract_features_kfold_V1.py` the first time, to produce `orientation_features_clipL14_all.npz`.)

### 3) Writing-tool classifier

Place augmented patches under:

```
4_Tool/3_Train_augmented_data/augmented_patches_tool_v3/
```

Then:

```bash
cd 4_Tool/3_Train_augmented_data
bash train.sh
```

(`train.sh` runs `2_train_classifier_V1.py`; first-time also run `1_extract_features_kfold_V1.py` to produce `writing_tool_features_clipL14_all.npz`.)

---

## Pipeline internals

Data flow between stages (handled automatically by the master script):

```
[DLA images + instances_test.json]
        │
        ▼
1_textdetection (DBNet)  ──►  show/text_detection/{jsons,imgs}
        │
        ▼
2_Colour/5_Test_Colour_without_annotation
        ColoredJson_Output/                (per-image JSON enriched with colour)
        ColoredName_CropedPatch_Output/    (cropped text patches)
        │
        ▼
3_Orn/5_Test_Orn_without_annotation
        ColoredOrnedJson_Output/           (JSON + orientation label)
        ColoredOrnedName_Output/           (renamed patches: …_hor / _vrt / _tlt)
        features_clipL14.npz               (CLIP-L/14 features)
        │
        ▼
4_Tool/5_Test_tool_without_annotation
        ColoredOrnedTooledJson_Output/     (final JSON: colour + orientation + tool)
        │
        ▼
5_visualization
        Prediction_full.json
        output_compare_pred_top_gt_bottom/ (final result)
```

---

## Configuration flags

The master script exposes two env-vars to control how it deals with previous runs:

| Variable | Values | Default | Effect |
|---|---|---|---|
| `OUTPUT_CONFLICT_MODE` | `archive` \| `delete` | `delete` | When an output already exists, `archive` renames it to `*_auto_V1`, `*_auto_V2`, … instead of deleting it. |
| `CLEAN_AUTO_VERSIONS` | `true` \| `false` | `true` | Before running, delete any previously archived `*_auto_Vx` files/folders. |

Examples:

```bash
# Keep previous outputs, don't clean archives
OUTPUT_CONFLICT_MODE=archive CLEAN_AUTO_VERSIONS=false bash Run_pipeline_testing_auto_paper.sh

# Force a clean run (default)
bash Run_pipeline_testing_auto_paper.sh
```

You can also override which scripts/checkpoints the pipeline picks up:

```bash
COLOUR_SCRIPT=/abs/path/to/colour.py \
ORN_MODEL_SRC=/abs/path/to/orn_model.pt \
TOOL_MODEL_SRC=/abs/path/to/tool_model.pt \
bash Run_pipeline_testing_auto_paper.sh
```

---

## Maintainer

This component (the CVC word pipeline) is maintained by **Quang Vinh Dang** (Visual Manuscript Analysis Lab, Centre for the Study of Manuscript Cultures, Universität Hamburg), under supervision from **Dr. Hussein Mohammed**.

For the full author list of CVC and the paper citation, see the top-level [`../README.md`](../README.md).

