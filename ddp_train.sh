
#! /bin/bash

NUM_GPUS_PER_WORKER=4
MASTER_PORT=29502

train_options=" \
        --savepath blip_reward
       --batch_size 4 \
       --gradient_accumulation_steps 4 \
       --epochs 10 \
       --distributed True \
       --gpu_num ${NUM_GPUS_PER_WORKER} \
       --gpu_id '2,3,4,5' \
       --clear_visualizer \
       --fix_rate 0.5 \
       --lr 1e-05 \
       --lr-decay-style cosine \
       --warmup 0.0 \
       --valid_per_epoch 4 \
"

run_cmd="torchrun
        --nnodes=1
        --nproc_per_node=${NUM_GPUS_PER_WORKER}
        --master_port=${MASTER_PORT}
        ./train.py ${train_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x