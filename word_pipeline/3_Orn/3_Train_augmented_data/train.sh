export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

set +u
source activate open-mmlab
set -u
set -e

python 1_extract_features_kfold_V1.py

python 2_train_classifier_V3.py

VER=$(printf '%02d' "$(sed -n 's/^BEST_VERSION *= *\([0-9]*\).*/\1/p' 2_train_classifier_V3.py)")
STEM="Orn_classifier_v${VER}_trained_on_all_augmented_training_data"

cp "${STEM}.pt" "${STEM}.meta.json" ../4_Test_orn_with_annotation/

cp "${STEM}.pt" "${STEM}.meta.json" ../5_Test_Orn_without_annotation/
