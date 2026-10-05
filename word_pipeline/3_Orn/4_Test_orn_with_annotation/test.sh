export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

set +u
source activate open-mmlab
set -u

python 1_extract_features_kfold_V1.py

python 2_test_classifier_V3.py
