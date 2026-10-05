export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

set +u
source activate open-mmlab
set -u
set -e

#python 1_extract_features_kfold_V1.py

python 2_train_classifier_V1.py

VER=$(printf '%02d' "$(sed -n 's/^BEST_VERSION *= *\([0-9]*\).*/\1/p' 2_train_classifier_V1.py)")
STEM="writing_tool_classifier_V${VER}_trained_on_all_training_data"

cp "${STEM}.pt" "${STEM}.meta.json" ../4_Test_tool_with_annotation/
cp "${STEM}.pt" "${STEM}.meta.json" ../5_Test_tool_without_annotation/
