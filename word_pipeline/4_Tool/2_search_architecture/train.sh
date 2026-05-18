export CUDA_VISIBLE_DEVICES=0

source activate open-mmlab

#bash aug.sh

#python 1_extract_features_kfold_V1.py

python 2_train_classifier_writing_tool_V11.py
