#!/usr/bin/env bash

# these should be consistent with the setting in YAML config files
data_root=../condqa_old/data
model_root=../condqa_old/model
logdir=new_log_11152022.txt

for i in {0..0..1}
do
    # python -m torch.distributed.run \
    #     --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=6005 train_answer.py \
    #     --mode=train --epoch=${i} --data_root=${data_root} --model_root=${model_root} \
    #     --logdir=${logdir} --accumulation_step=4

    python train_answer.py \
        --mode=inference --epoch=${i+1} --data_root=${data_root} --model_root=${model_root} \
        --logdir=${logdir}
        
done

