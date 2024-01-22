#! /bin/bash

NUM_GPUS_PER_WORKER=2
MASTER_PORT=29500


run_cmd="torchrun
        --nnodes=1
        --nproc_per_node=${NUM_GPUS_PER_WORKER}
        --master_port=${MASTER_PORT}
        ./train.py"

echo ${run_cmd}
eval ${run_cmd}
set +x