#!/usr/bin/env bash

data_root=../condqa_files/data
model_root=../condqa_files/model

echo "evaluating..."
python main.py \
    --mode=inference --epoch=1 --data_root=${data_root} --model_root=${model_root} \
    --tqdm --inference_data=dev_data --model_size large #--only_need_to_predict_conditions
