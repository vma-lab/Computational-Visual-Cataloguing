export CUDA_VISIBLE_DEVICES=1

source activate open-mmlab

python 1_extract_features_kfold_V1.py

python 2_test_classifier_V3.py
