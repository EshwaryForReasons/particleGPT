#!/bin/bash

#SBATCH -A m3443
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 32

export SLURM_CPU_BIND="cores"
cd $SCRATCH/particleGPT
srun -n 1 -G 4 -c 32 bash job_scripts/script_train_5_10_2.sh