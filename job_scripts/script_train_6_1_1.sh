cd $SCRATCH/particleGPT
nvidia-smi
torchrun --standalone --nproc_per_node=4 train.py config/model_6_1_1.json