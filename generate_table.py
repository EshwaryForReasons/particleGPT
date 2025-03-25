import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import glob
import json

import configurator
import pUtil

script_dir = os.path.dirname(os.path.abspath(__file__))
latest_sampling_dir = pUtil.get_latest_sampling_dir(configurator.output_dir_name)

dictionary_file = os.path.join('data', configurator.dataset, 'dictionary.json')
meta_file = os.path.join('data', configurator.dataset, 'outputs', 'meta.pkl')

if not os.path.exists(meta_file):
    print(f'meta.pkl does not exist for {configurator.dataset}')
    exit()

# Get dataset information from the meta file
with open(meta_file, 'rb') as f:
    print("Reading meta file for dataset information.")
    meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    total_events = meta['total_events']
    num_train_events = meta['num_train_events']
    num_train_tokens = meta['num_train_tokens']
    num_val_events = meta['num_val_events']
    num_val_tokens = meta['num_val_tokens']
    print("Finished reading meta file for dataset information.")

# Get training information from the training log file
train_model_dir = pUtil.get_training_dir(configurator.output_dir_name)
training_log_files = glob.glob(os.path.join(train_model_dir, "*.jsonl"))

# Since we only test and save checkpoint every eval_interval, we need to keep track of checkpoint data separately for saved train_loss and val_loss
train_data_dict = {}
checkpoint_data_dict = {}
with open(training_log_files[0], 'r') as data_file:
    for line in data_file:
        jdata = json.loads(line)
        if jdata['message'] == "Training progress" and 'iter' in jdata:
            train_data_dict[jdata['iter']] = (jdata['train_loss'], jdata['val_loss'])
        elif jdata['message'] == "Training progress: checking checkpoint conditions":
            checkpoint_data_dict[jdata['step']] = (jdata['train_loss'], jdata['val_loss'])

# This is for the running data
sorted_iterations = sorted(train_data_dict.keys())
train_loss = [train_data_dict[i][0] for i in sorted_iterations]
val_loss = [train_data_dict[i][1] for i in sorted_iterations]

# This is for the saved (checkpointed) data
saved_sorted_iterations = sorted(checkpoint_data_dict.keys())
saved_train_loss = [checkpoint_data_dict[i][0] for i in saved_sorted_iterations]
saved_val_loss = [checkpoint_data_dict[i][1] for i in saved_sorted_iterations]

min_saved_train_loss = min(saved_train_loss)
min_saved_val_loss = min(saved_val_loss)

epochs_trained = 3
training_time = 10
table_data = {
    "Dataset": f'{configurator.output_dir_name}',
    "Vocab size": f'{vocab_size:,}',
    "Train tokens": f'{num_train_tokens:,} tokens',
    "Val tokens": f'{num_val_tokens:,} tokens',
    "Train events": f'{num_train_events:,} events',
    "Val events": f'{num_val_events:,} events',
    "Batch size": configurator.batch_size,
    "Block size": configurator.block_size,
    "Learning rate": configurator.learning_rate,
    "Min learning rate": configurator.min_lr,
    "Training time": training_time,
    "Epochs trained": epochs_trained,
    "Training iters": len(train_data_dict),
    "Best val loss": f'{min_saved_val_loss:.5f}',
    "Best train loss": f'{min_saved_train_loss:.5f}'
}

for key, value in table_data.items():
    print(f'{key}; {value}')

# Uncomment to create a table.png using matplotlib. I do not need this right now, so I am not using it.

# # Creating the table
# rows = list(table_data.keys())
# values = list(table_data.values())
# table_data_list = list(zip(rows, values))

# # Create figure and axis
# fig, ax = plt.subplots(figsize=(6, len(rows) * 0.5))

# table = ax.table(cellText=table_data_list, colLabels=["Metric", "Value"], loc="center")
# table.scale(1, 1.5)

# # Hide axes and border
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# plt.box(on=None)

# plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0)
# plt.savefig("table.png", bbox_inches="tight", dpi=300)
# plt.close()