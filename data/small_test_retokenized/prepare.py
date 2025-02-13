
import os
import numpy as np
import pandas as pd
import pickle

import tokenizer
import dictionary

script_dir = os.path.dirname(os.path.abspath(__file__))

input_data_filename = os.path.join(script_dir, 'data.csv')
preprocessed_data_filename = os.path.join(script_dir, 'outputs/temp_preprocessed.csv')
tokenized_data_filename = os.path.join(script_dir, 'outputs/temp_tokenized.csv')
humanized_dictionary_filename = os.path.join(script_dir, 'outputs/humanized_dictionary.txt')
train_bin_filename = os.path.join(script_dir, 'outputs/train.bin')
val_bin_filename = os.path.join(script_dir, 'outputs/val.bin')
meta_filename = os.path.join(script_dir, 'outputs/meta.pkl')

# Ensure the outputs directory exists
os.makedirs(os.path.dirname(os.path.join(script_dir, preprocessed_data_filename)), exist_ok=True)

dictionary.output_humanized_dictionary(humanized_dictionary_filename)
tokenizer.preprocess_data(input_data_filename, preprocessed_data_filename)
tokenizer.tokenize_data(preprocessed_data_filename, tokenized_data_filename)

# create the train and test splits
input_file_pd = pd.read_csv(os.path.join(script_dir, tokenized_data_filename), sep=' ', dtype=np.uint16)
num_events = len(input_file_pd) + 1
num_train_events = int(num_events * 0.9)
num_val_events = num_events - num_train_events

print(f"Vocab size: {dictionary.get_vocab_size():,} tokens")
print(f"Total events: {num_events:,} events")
print(f"Train has: {num_train_events} events")
print(f"Val has: {num_val_events} events")

train_np = input_file_pd[:num_train_events].to_numpy().flatten()
val_np = input_file_pd[num_train_events - 1:].to_numpy().flatten()
train_np.tofile(train_bin_filename)
val_np.tofile(val_bin_filename)

meta = {
    'vocab_size': dictionary.get_vocab_size(),
}
with open(os.path.join(os.path.dirname(__file__), meta_filename), 'wb') as f:
    pickle.dump(meta, f)