import os
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()

for i in range(0, 50):
    os.system(f'python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --node_rank=0  --master_port=6005 train_answer.py    --mode=train   --epoch={i}')
    os.system(f'python train_answer.py    --mode=inference    --epoch={i+1}')

# for i in range(19, 20):
#     os.system(f'python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --node_rank=0  --master_port=6005 train_condition.py    --mode=train   --epoch={i}')
#     os.system(f'python train_condition.py    --mode=inference    --epoch={i+1}')
