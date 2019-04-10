#!/usr/bin/env sh

DATA_PATH = 'path_to_validation_rec'
MODEL_PATH ='path_to_model'

python -u eval.py \
 --data_path ${DATA_PATH} \
 --model_path ${MODEL_PATH}
