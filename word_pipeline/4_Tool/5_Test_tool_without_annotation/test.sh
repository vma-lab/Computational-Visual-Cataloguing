export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

set +u
source activate open-mmlab
set -u
set -e

python 1_extract_features_testing_MultiBatchThread_Fastest_V4.py

python 2_test_classifier_without_evaluation_MultiBatchThread_Fastest_V4.py
