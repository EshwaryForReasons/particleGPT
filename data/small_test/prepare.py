
import os
import numpy as np
import pandas as pd
import pickle

from tokenizer import preprocess_data
from tokenizer import tokenize_data
from dictionary import get_vocab_size

input_data_filename = 'data.csv'
preprocessed_data_filename = './outputs/temp_preprocessed.csv'
tokenized_data_filename = './outputs/temp_tokenized.csv'

# Ensure the outputs directory exists
os.makedirs(os.path.dirname(preprocessed_data_filename), exist_ok=True)

preprocess_data(input_data_filename, preprocessed_data_filename)
tokenize_data(preprocessed_data_filename, tokenized_data_filename)

# create the train and test splits
input_file_pd = pd.read_csv(tokenized_data_filename)
num_events = len(input_file_pd) + 1
num_train_events = int(num_events * 0.9)
num_val_events = num_events - num_train_events

print(f"Vocab size: {get_vocab_size():,} tokens")
print(f"Total events: {num_events:,} tokens")
print(f"Train has: {num_train_events} tokens")
print(f"Val has: {num_val_events} tokens")

np_train_data = np.array([])
np_val_data = np.array([])
with open(tokenized_data_filename, 'r') as f:
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

np_train_data.tofile(os.path.join(os.path.dirname(__file__), 'outputs', 'train.bin'))
np_val_data.tofile(os.path.join(os.path.dirname(__file__), 'outputs', 'val.bin'))

meta = {
    'vocab_size': get_vocab_size(),
}
with open(os.path.join(os.path.dirname(__file__), './outputs/meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)