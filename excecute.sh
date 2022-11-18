#!/usr/bin/env bash

# these should be consistent with the setting in YAML config files
data_root=../condqa_old/data
model_root=../condqa_old/model


logdir=cl_hpt.txt
for i in {0..99..1}
do
    echo "training..."
    python -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=6005 train_answer.py \
        --mode=train --epoch=${i} --data_root=${data_root} --model_root=${model_root} \
        --logdir=${logdir} --accumulation_step=4 --train_condition --contrastive_learning \
        --warmup_epoch_num=10 --total_epoch_num=100 --contrastive_mode=hpt

    echo "evaluating..."
    python train_answer.py \
        --mode=inference --epoch=${i} --data_root=${data_root} --model_root=${model_root} \
        --logdir=${logdir}
        
done

rm "${model_root}/model_current.pt"
cp "${model_root}/model_best.pt" "${model_root}/model_best_hptcl_nocondition.pt"
rm "${model_root}/model_best.pt"
rm "${model_root}/result.txt"


logdir=cl_simcse.txt
for i in {0..99..1}
do
    echo "training..."
    python -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=6005 train_answer.py \
        --mode=train --epoch=${i} --data_root=${data_root} --model_root=${model_root} \
        --logdir=${logdir} --accumulation_step=4 --train_condition --contrastive_learning \
        --warmup_epoch_num=10 --total_epoch_num=100 --contrastive_mode=simcse

    echo "evaluating..."
    python train_answer.py \
        --mode=inference --epoch=${i} --data_root=${data_root} --model_root=${model_root} \
        --logdir=${logdir}
        
done

rm "${model_root}/model_current.pt"
cp "${model_root}/model_best.pt" "${model_root}/model_best_simcse_nocondition.pt"
rm "${model_root}/model_best.pt"
rm "${model_root}/result.txt"


# 接下来应该做哪些实验
# 首先我觉得最有效的应该是contrastive learning. 先把图画出来然后做实验，学习mzy的文章
# 其次attention机制估计不会很好写，但是那篇文章一定要好好看看 还有socialformer
# 实验效果要拉个excel写上去
# pretraining最后写，我觉得至少要跑三天，之后怎么样再说
