#!/bin/bash
set -ex
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_LOGS="+dynamo"

MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --rdzv_id=pytorchddp --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} "

# be sure to set ~/.netrc for wandb
cat /2214/wandb.netrc > /root/.netrc
export ENCODECMAE_RUN='encodecmae_large_100w'
export TORCH_HOME='/2214/torch'
cd /2214/dongyuanliang/encodecmae_pl
/2214/conda_envs/encodecmae/bin/torchrun $DISTRIBUTED_ARGS trainer.py 2>&1 | tee train_log_${ENCODECMAE_RUN}_${NODE_RANK}.txt
tail -f /dev/null
