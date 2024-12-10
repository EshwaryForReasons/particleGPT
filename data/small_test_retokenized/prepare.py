
import os
import numpy as np
import pandas as pd
import pickle

from tokenizer import preprocess_data
from tokenizer import tokenize_data
from dictionary import get_vocab_size
from dictionary import output_humanized_dictionary

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

output_humanized_dictionary(humanized_dictionary_filename)

preprocess_data(input_data_filename, preprocessed_data_filename)
tokenize_data(preprocessed_data_filename, tokenized_data_filename)

# create the train and test splits
input_file_pd = pd.read_csv(os.path.join(script_dir, tokenized_data_filename))
num_events = len(input_file_pd) + 1
num_train_events = int(num_events * 0.9)
num_val_events = num_events - num_train_events

print(f"Vocab size: {get_vocab_size():,} tokens")
print(f"Total events: {num_events:,} tokens")
print(f"Train has: {num_train_events} tokens")
print(f"Val has: {num_val_events} tokens")

np_train_data = np.array([])
np_val_data = np.array([])
with open(os.path.join(script_dir, tokenized_data_filename), 'r') as f:
    line_num = 0
    for line in f:
        generations = line.split()
        for gen in generations:
            if line_num < num_train_events:
                np_train_data = np.append(np_train_data, int(gen))
            else:
                np_val_data = np.append(np_val_data, int(gen))
        line_num += 1

np_train_data = np.array(np_train_data, dtype=np.uint16)
np_val_data = np.array(np_val_data, dtype=np.uint16)

np_train_data.tofile(train_bin_filename)
np_val_data.tofile(val_bin_filename)

meta = {
    'vocab_size': get_vocab_size(),
}
with open(os.path.join(os.path.dirname(__file__), meta_filename), 'wb') as f:
    pickle.dump(meta, f)