#!/bin/bash
set -ex

MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --rdzv_id=pytorchddp --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} "

cat /home/dongyuanliang/.netrc
export ENCODECMAE_RUN='encodecmae_base'
/2214/conda_envs/musicgen/bin/torchrun $DISTRIBUTED_ARGS trainer.py 2>&1 | tee -a train_log_${MUSICGEN_RUN}_${NODE_RANK}.txt
tail -f /dev/null
