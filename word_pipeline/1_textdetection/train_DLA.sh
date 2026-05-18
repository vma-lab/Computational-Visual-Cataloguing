export CUDA_VISIBLE_DEVICES=0
source activate open-mmlab

bash dist_train.sh \
configs/textdet/dbnet/config_DLA.py \
work_dirs/dbnet \
1 \


