# Word Pipeline (part of CVC)

This folder contains the **word pipeline** of CVC (Computational Visual Cataloguing): word detection, colour estimation, orientation and writing-implement classification, and visualisation. It is one of three components of the parent repository, together with [`../page_detection/`](../page_detection) and [`../scriptsight/`](../scriptsight). See [`../README.md`](../README.md) for the overall system and citation.

```
1_textdetection → 2_Colour → 3_Orn → 4_Tool → 5_visualization
```

## Downloads (UHHCloud)

| Resource | Link |
|---|---|
| Checkpoints | [checkpoint_model](https://cloud.uni-hamburg.de/s/GQjJdxzxe9zdSjJ) |
| Dataset (DLA-RMR) | [Dataset_DLA](https://cloud.uni-hamburg.de/s/AymfqDj3H5Zz3Qx) |
| Augmented data (training only) | [Augmented_DLA.zip](https://cloud.uni-hamburg.de/s/bwr99TCNMdLdJGj) |
| Reference output: colour prediction | [output_visualized_predict_Colour](https://cloud.uni-hamburg.de/s/8XDR92bixgGtdoF) |
| Reference output: full prediction | [output_visualized_predict_full](https://cloud.uni-hamburg.de/s/ZsQakA5Mnd2Cmem) |

## 1. Environment

```bash
cd 1_textdetection
bash setup_openmmlab_env.sh      # creates conda env `open-mmlab`
```

Scripts use GPU 0 by default; set `CUDA_VISIBLE_DEVICES` to change it.

## 2. Place checkpoints and dataset

| File | Destination |
|---|---|
| `text_detection.pth` | `1_textdetection/work_dirs/dbnet/` |
| `best_imgur5k_epoch_420.pth` (training only) | `1_textdetection/work_dirs/dbnet/` |
| `Orn_classifier_v16_trained_on_all_augmented_training_data.pt` | `3_Orn/5_Test_Orn_without_annotation/` and `3_Orn/4_Test_orn_with_annotation/` |
| `writing_tool_classifier_V09_trained_on_all_training_data.pt` | `4_Tool/5_Test_tool_without_annotation/` and `4_Tool/4_Test_tool_with_annotation/` |
| `Dataset_DLA/` (`img/`, `instances_test.json`, `instances_training.json`) | `1_textdetection/data/DLA/` |

The `.meta.json` file next to each orientation checkpoint location is part of the repository and must stay with the `.pt` file. If `1_textdetection/pretrained/RN50.pt` is missing, run `1_textdetection/pretrained/download_clip_models.sh`.

## 3. Run the testing pipeline

```bash
bash Run_pipeline_testing_auto_paper.sh
```

Outputs are written to `5_visualization/`; the main qualitative result is `5_visualization/output_compare_pred_top_gt_bottom/`.

## 4. Evaluation results

| Task | Result files |
|---|---|
| Orientation | `3_Orn/2_search_architecture/orientation_mlp_arch_search_V01-V20_rankscore_report.txt`<br>`3_Orn/4_Test_orn_with_annotation/test_predictions_*.csv` |
| Writing implement | `4_Tool/2_search_architecture/writing_tool_mlp_arch_search_V01-V20_rankscore_report.txt`<br>`4_Tool/4_Test_tool_with_annotation/test_predictions_*.csv` |
| Colour | `2_Colour/4_Test_colour_with_annotation/colour_eval_result/` |

To re-run the evaluations:

```bash
# orientation / writing implement
cp -r /path/to/Dataset_DLA/img 3_Orn/0_Crop_Patches/imgs
(cd 3_Orn/0_Crop_Patches && bash run.sh)
cp -r 3_Orn/0_Crop_Patches/text_patches_test_20 3_Orn/4_Test_orn_with_annotation/
cp -r 3_Orn/0_Crop_Patches/text_patches_test_20 4_Tool/4_Test_tool_with_annotation/
(cd 3_Orn/4_Test_orn_with_annotation && bash test.sh)
(cd 4_Tool/4_Test_tool_with_annotation && bash test.sh)

# colour
cp -r /path/to/Dataset_DLA/img 2_Colour/4_Test_colour_with_annotation/img
(cd 2_Colour/4_Test_colour_with_annotation && bash run.sh)
```

## 5. Training

Unzip `Augmented_DLA.zip` and place:

| Source | Destination |
|---|---|
| `Augmented_DLA/1_TextDetection/` (`img/`, `instances_*.json`) | `1_textdetection/data/DLA/` |
| `Augmented_DLA/3_Orn/` | `3_Orn/1_augmentation/text_patches_augmented_betterv3_80/` |
| `Augmented_DLA/4_Tool/` | `4_Tool/1_augmentation/augmented_patches_tool_v3/` |

Then link the patches into steps 2 and 3:

```bash
for s in 2_search_architecture 3_Train_augmented_data; do
  ln -sfn ../1_augmentation/text_patches_augmented_betterv3_80 3_Orn/$s/text_patches_augmented_betterv3_80
  ln -sfn ../1_augmentation/augmented_patches_tool_v3          4_Tool/$s/augmented_patches_tool_v3
done
```

Text detection:

```bash
(cd 1_textdetection && bash train_DLA.sh)
```

Orientation (architecture search, final training, test):

```bash
(cd 3_Orn/2_search_architecture && bash train.sh)
(cd 3_Orn/3_Train_augmented_data && bash train.sh)
(cd 3_Orn/4_Test_orn_with_annotation && bash test.sh)
```

Writing implement (architecture search, final training, test):

```bash
(cd 4_Tool/2_search_architecture && bash train.sh)
cp 4_Tool/2_search_architecture/writing_tool_features_clipL14_all.npz 4_Tool/3_Train_augmented_data/
(cd 4_Tool/3_Train_augmented_data && bash train.sh)
(cd 4_Tool/4_Test_tool_with_annotation && bash test.sh)
```

Each `3_Train_augmented_data/train.sh` copies the trained checkpoint into the corresponding `4_Test_*` and `5_Test_*` folders.

## Developer

Developed by **Quang Vinh Dang** (Visual Manuscript Analysis Lab, Centre for the Study of Manuscript Cultures, Universität Hamburg), under supervision from **Dr. Hussein Mohammed**.
