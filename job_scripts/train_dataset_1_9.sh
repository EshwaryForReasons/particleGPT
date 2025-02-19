shifter --image=docker:eshym/docker_img_particlegpt:v2 /bin/bash

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29507
export OMP_NUM_THREADS=8

echo $MASTER_ADDR
echo $MASTER_PORT
echo $OMP_NUM_THREADS
echo $SLURM_JOB_NUM_NODES

torchrun --nproc_per_node=4 --nnodes=$SLURM_JOB_NUM_NODES --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py config/dataset_1_9.json