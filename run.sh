#!/usr/bin/env bash

data_root=../condqa_files/data
model_root=../condqa_files/model

# rm "${model_root}/result.txt"
logdir=1215_longformer_large #globalattention.txt

for i in {0..50..1}
do
    echo "training epoch ${i}..."
    python -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=6005 main.py \
        --mode=train --epoch=${i} --data_root=${data_root} --model_root=${model_root} \
        --logdir=${logdir} --accumulation_step=1 --train_condition --batch_size=8 \
        --warmup_epoch_num=5 --total_epoch_num=50 --tqdm --repeat=2 --model_size large #--contrastive_learning --contrastive_mode=hpt \

    echo "evaluating..."
    python main.py \
        --mode=inference --epoch=${i} --data_root=${data_root} --model_root=${model_root} \
        --logdir=${logdir} --tqdm --inference_data=dev_data --model_size large
        
    ((num=$i %5))
    if [ "$num" == 0 ];then
        python main.py \
            --mode=inference --epoch=${i} --data_root=${data_root} --model_root=${model_root} \
            --logdir=${logdir} --tqdm --inference_data=train_data  --model_size large
    fi
done

