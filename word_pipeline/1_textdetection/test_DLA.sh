export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

source activate open-mmlab

bash dist_test.sh \
  configs/textdet/dbnet/config_DLA.py \
  work_dirs/dbnet/text_detection.pth 1 \
  --eval hmean-iou \
  --show-dir show/text_detection \
  --show-with-score-map
