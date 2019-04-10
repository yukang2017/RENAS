#!/usr/bin/env sh
#export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

echo ${1}
NETWORK='nasnet'
NUM_EPOCH=200
LR=0.05
SECOND_LR=0.0001
BATCH_SIZE=256
GPU='0,1,2,3'
LR_FACTOR=0.1
SECOND_LR_FACTOR=1
FIRST_EPOCH=180
LOAD_EPOCH=180
LR_SCHEDULER='poly'
LR_SCHEDULER_S2='step'
MAX_NUM_UPDATE=900000
WARMUP='--warmup'
WARMUP_LINEAR='--warmup-linear'
WARMUP_EPOCH=1
WARMUP_LR=0.1
WARMUP_END_LR=0.1
USE_AUX_HEAD='--use-aux'


USE_PYTHON_ITER=0
USE_GPU_AUGMENTER=0
USE_COLOR_AUG=0
USE_LIGHTING_AUG=0

DATA_DIR= 'path_to_dataset'

python -u train_model.py \
 --data-dir ${DATA_DIR} \
 --model-prefix ${HADOOP_MODEL_SAVE} \
 --batch-size ${BATCH_SIZE}  \
 --aug-level=3 \
 --num-epoch ${FIRST_EPOCH}  \
 --lr ${LR}  \
 --lr-factor ${LR_FACTOR} \
 --lr-step-epochs ${LR_STEP_EPOCH} \
 --gpus ${GPU} \
 ${USE_AUX_HEAD} \
 --max-num-update ${MAX_NUM_UPDATE} \
 --lr-scheduler ${LR_SCHEDULER}
