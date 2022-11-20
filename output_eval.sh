#!/usr/bin/env bash

data_root=../condqa_old/data
model_root=../condqa_old/model

echo "evaluating..."
python train.py \
    --mode=inference --epoch=1 --data_root=${data_root} --model_root=${model_root}   
