#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 main_v2.py --beta=0.7 --total_iters=5000 --H=4 --ft_lr=0.001 --update_lr=0.01 --meta_lr=0.01 --k_spt=5 --random_seed=123 --num_tasks=5 --dataset=cifar10 --n_way=1
