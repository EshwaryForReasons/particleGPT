cd $SCRATCH/particleGPT
torchrun --standalone --nproc_per_node=4 train.py config/model_5_9_8.json