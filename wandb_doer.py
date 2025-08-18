import sys
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image

import wandb

import pUtil
import analysis as anal
import configurator

script_dir = Path(__file__).resolve().parent

# learning rate decay scheduler
def get_lr(conft, it):
    if conft.lr_scheduler == 'cosine_annealing_with_warmup':
        # 1) linear warmup for warmup_iters steps
        if it < conft.warmup_iters:
            return conft.learning_rate * it / conft.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it >= conft.lr_decay_iters:
            return conft.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - conft.warmup_iters) / (conft.lr_decay_iters - conft.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return conft.min_lr + coeff * (conft.learning_rate - conft.min_lr)
    elif conft.lr_scheduler == 'cosine_with_warmup':
        # 1) linear warmup for warmup_iters steps
        if it < conft.warmup_iters:
            return conft.learning_rate * it / conft.warmup_iters
        # 2) in between, use cosine decay down to min learning rate
        decay_ratio = (it - conft.warmup_iters) / (conft.lr_decay_iters - conft.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return conft.min_lr + coeff * (conft.learning_rate - conft.min_lr)
    elif conft.lr_scheduler == 'cosine_annealing_with_warm_restarts':
        # 1) linear warmup for warmup_iters steps
        if it < conft.warmup_iters:
            return conft.learning_rate * (it / conft.warmup_iters)
        # Adjust iteration to account for warmup
        it -= conft.warmup_iters
        # 2) Find current cycle and position in the cycle
        cycle = 0
        curr_cycle_len = conft.lr_decay_iters
        iter_in_cycle = it
        while iter_in_cycle >= curr_cycle_len:
            iter_in_cycle -= curr_cycle_len
            cycle += 1
            curr_cycle_len = int(curr_cycle_len * conft.cycle_steps_mult)
        # 3) Decay the base learning rate for the current cycle
        curr_base_lr = conft.learning_rate * (conft.base_lr_decay_mult ** cycle)
        # 4) Normalized progress within the cycle
        t = iter_in_cycle / curr_cycle_len
        # 5) Cosine annealing
        lr = conft.min_lr + 0.5 * (curr_base_lr - conft.min_lr) * (1 + math.cos(math.pi * t))
        return lr
    elif conft.lr_scheduler == 'constant':
        return conft.learning_rate
    raise ValueError(f"Unknown lr_scheduler {conft.lr_scheduler}")

def send_to_wandb(model_name):
    config_filepath = pUtil.get_model_config_filepath(model_name)
    with open(config_filepath, 'r') as f:
        config = json.load(f)
        _, conft, _ = configurator.perform_configuration(config_filepath)

    # 1) Initialize the project.
    wandb.init(project="particleGPT", name=f"run-{model_name}", config=config)

    # 2) Log training metrics.
    training_run_data = anal.tables.get_training_run_data(model_name)
    for i, row in training_run_data.running_df.iterrows():
        wandb.log({"iter": row['iter'], "train_loss": row['train_loss'], "val_loss": row['val_loss'], "lr": get_lr(conft, row['iter'])})

    # 3) Log the distributions as images.
    # I have found this is the easiest way to do this. Simple and effective.
    latest_sampling_dir = pUtil.get_latest_sampling_dir(model_name)
    for image_filepath in latest_sampling_dir.glob("*.png"):
        image = Image.open(image_filepath)
        img_array = np.array(image)
        wandb.log({image_filepath.stem: wandb.Image(img_array)})
    
    wandb.finish()
    
    print(f"Logged data for model {model_name} to wandb.")
    
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: python wandb_doer.py config/model_conf_file.json")
        sys.exit(1)
    
    config_filepath = script_dir / sys.argv[1]
    generic, training, sampling = configurator.perform_configuration(config_filepath)
    send_to_wandb(generic.model_name)