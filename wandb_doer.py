import numpy as np
import json
from pathlib import Path
import math
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

model_name = 'model_10M_9_5'
generated_samples_dir = script_dir.parent / 'generated_samples' / model_name / 'sampling_0'

config_filepath = pUtil.get_model_config_filepath(model_name)
with open(config_filepath, 'r') as f:
    config = json.load(f)
    _, conft, _ = configurator.perform_configuration(config_filepath)

# 1) Initialize the project.
wandb.init(
    project="particleGPT",
    name=f"run-{model_name}",
    config=config
)

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

# # 3. Log a physics distribution
# dist = np.random.normal(0, 1, 1000)  # example distribution
# fig, ax = plt.subplots()
# ax.hist(dist, bins=50)
# wandb.log({"physics_distribution": wandb.Image(fig)})

# # 4. Save raw distribution as artifact
# np.save("distribution.npy", dist)
# wandb.save("distribution.npy")  # Upload file to W&B


# ------------------------------------------
# Log distributions as histogram
# ------------------------------------------

# # 3) Log distributions.
# verbose_columns = ["pdgid", "e", "px", "py", "pz", "pt", "eta", "theta", "phi"]

# def get_leading_column_data(model_name, column_name):
#     relevant_column_pos = verbose_columns.index(column_name)
    
#     real_verbose_data = data_manager.load_verbose_dataset(pUtil.get_model_preparation_dir(model_name) / 'real_verbose_test_particles.csv', pad_token = np.nan)
#     all_instances_of_this_column_real = []
#     for event in real_verbose_data:
#         secondaries = event[1:]
#         # Find index of particle with the highest energy
#         leading_particle_idx = np.nanargmax(secondaries[:, 1])
#         leading_particle = secondaries[leading_particle_idx]
#         all_instances_of_this_column_real.append(leading_particle[relevant_column_pos])
    
#     sampled_verbose_data = data_manager.load_verbose_dataset(pUtil.get_latest_sampling_dir(model_name) / 'untokenized_samples_verbose.csv', pad_token = np.nan)
#     all_instances_of_this_column_sampled = []
#     for event in sampled_verbose_data:
#         secondaries = event[1:]
#         # Find index of particle with the highest energy
#         leading_particle_idx = np.nanargmax(secondaries[:, 1])
#         leading_particle = secondaries[leading_particle_idx]
#         all_instances_of_this_column_sampled.append(leading_particle[relevant_column_pos])
    
#     return all_instances_of_this_column_real, all_instances_of_this_column_sampled

# def get_all_column_data(model_name, column_name):
#     relevant_column_pos = verbose_columns.index(column_name)
    
#     real_verbose_data = data_manager.load_verbose_dataset(pUtil.get_model_preparation_dir(model_name) / 'real_verbose_test_particles.csv', pad_token = np.nan)
#     all_instances_of_this_column_real = []
#     for event in real_verbose_data:
#         secondaries = event[1:]
#         for particle in secondaries:
#             if not np.isnan(particle[relevant_column_pos]):
#                 all_instances_of_this_column_real.append(particle[relevant_column_pos])
    
#     sampled_verbose_data = data_manager.load_verbose_dataset(pUtil.get_latest_sampling_dir(model_name) / 'untokenized_samples_verbose.csv', pad_token = np.nan)
#     all_instances_of_this_column_sampled = []
#     for event in sampled_verbose_data:
#         secondaries = event[1:]
#         for particle in secondaries:
#             if not np.isnan(particle[relevant_column_pos]):
#                 all_instances_of_this_column_sampled.append(particle[relevant_column_pos])
    
#     return all_instances_of_this_column_real, all_instances_of_this_column_sampled

# continuous_dist_columns = ["e", "px", "py", "pz", "pt", "eta", "theta", "phi"]

# # Leading distributions
# for column_name in continuous_dist_columns:
#     real_data, sampled_data = get_leading_column_data(model_name, column_name)
#     bin_settings = anal.plotting.get_common_data(model_name)
#     range = (bin_settings[column_name]['min'], bin_settings[column_name]['max'])
#     n_bins = bin_settings[column_name]['bins']
#     wandb.log({f"{column_name}_leading_real": wandb.Histogram(np_histogram=np.histogram(real_data, bins=n_bins, range=range))})
#     wandb.log({f"{column_name}_leading_sampled": wandb.Histogram(np_histogram=np.histogram(sampled_data, bins=n_bins, range=range))})
    
# # All distributions
# for column_name in continuous_dist_columns:
#     real_data, sampled_data = get_all_column_data(model_name, column_name)
#     bin_settings = anal.plotting.get_common_data(model_name)
#     range = (bin_settings[column_name]['min'], bin_settings[column_name]['max'])
#     n_bins = bin_settings[column_name]['bins']
#     wandb.log({f"{column_name}_all_real": wandb.Histogram(np_histogram=np.histogram(real_data, bins=n_bins, range=range))})
#     wandb.log({f"{column_name}_all_sampled": wandb.Histogram(np_histogram=np.histogram(sampled_data, bins=n_bins, range=range))})

# ------------------------------------------
# Log distributions as table
# ------------------------------------------

# def log_distributions(input_eta, sampled_eta, name="eta"):
#     # Create a combined table for interactive histograms
#     rows = [[v, "input"] for v in input_eta] + [[v, "sampled"] for v in sampled_eta]
#     table = wandb.Table(data=rows, columns=[name, "source"])
    
#     wandb.log({
#         f"{name}_histogram": wandb.Histogram(input_eta),
#         f"{name}_sampled_histogram": wandb.Histogram(sampled_eta),
#         f"{name}_table": table
#     })
    
# # Leading distributions
# for column_name in continuous_dist_columns:
#     real_data, sampled_data = get_leading_column_data(model_name, column_name)
#     bin_settings = anal.plotting.get_common_data(model_name)
#     table = wandb.Table(data=[[v, "input"] for v in real_data] + [[v, "sampled"] for v in sampled_data], columns=[column_name, "source"])
#     wandb.log({f"{column_name}_leading_table": table})
    
# # Leading distributions
# for column_name in continuous_dist_columns:
#     real_data, sampled_data = get_all_column_data(model_name, column_name)
#     bin_settings = anal.plotting.get_common_data(model_name)
#     table = wandb.Table(data=[[v, "input"] for v in real_data] + [[v, "sampled"] for v in sampled_data], columns=[column_name, "source"])
#     wandb.log({f"{column_name}_all_table": table})
    
