import os
import numpy as np
import pandas as pd
import pickle
import sys

import data.tokenizer as tokenizer
import data.dictionary as dictionary

import pTokenizerModule as pTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configurator

script_dir = os.path.dirname(os.path.abspath(__file__))

def prepare_training():
    # Ensure the outputs directory exists
    input_data_filename = os.path.join(script_dir, configurator.dataset, 'data.csv')
    dictionary_filename = os.path.join(script_dir, configurator.dataset, 'dictionary.json')
    tokenized_data_filename = os.path.join(script_dir, configurator.dataset, 'tokenized_data.csv')
    humanized_dictionary_filename = os.path.join(script_dir, configurator.dataset, 'humanized_dictionary.txt')
    train_bin_filename = os.path.join(script_dir, configurator.dataset, 'outputs', 'train.bin')
    val_bin_filename = os.path.join(script_dir, configurator.dataset, 'outputs', 'val.bin')
    meta_filename = os.path.join(script_dir, configurator.dataset, 'outputs', 'meta.pkl')
    os.makedirs(os.path.dirname(train_bin_filename), exist_ok=True)

    # Only prepare if we haven't already prepared the data
    if os.path.exists(meta_filename):
        with open(meta_filename, 'rb') as f:
            meta = pickle.load(f)
            if meta['already_prepared']:
                print("Data already prepared")
                return

    dictionary.update_dictionary_particle_list(input_data_filename, dictionary_filename)
    dictionary.output_humanized_dictionary(humanized_dictionary_filename)
    
    pTokenizer.tokenize_data(dictionary_filename, input_data_filename, tokenized_data_filename)

    # create the train and test splits
    # dtype=np.uint16
    input_file_pd = pd.read_csv(tokenized_data_filename, sep=' ')
    num_events = len(input_file_pd) + 1
    num_train_events = int(num_events * 0.9)
    num_val_events = num_events - num_train_events
    
    train_np = input_file_pd[:num_train_events].to_numpy().flatten()
    val_np = input_file_pd[num_train_events:].to_numpy().flatten()
    train_np.tofile(train_bin_filename)
    val_np.tofile(val_bin_filename)

    print("-----------------------")
    print("Data information:")
    print(f"Vocab size: {dictionary.get_vocab_size():,} tokens")
    print(f"Total events: {num_events:,} events")
    print(f"Train has: {num_train_events:,} events")
    print(f"Train has: {len(train_np):,} tokens")
    print(f"Val has: {num_val_events:,} events")
    print(f"Val has: {len(val_np):,} tokens")
    print("-----------------------")

    with open(meta_filename, 'wb') as f:
        meta = {
            'vocab_size': dictionary.get_vocab_size(),
            'total_events': num_events,
            'num_train_events': num_train_events,
            'num_train_tokens': len(train_np),
            'num_val_events': num_val_events,
            'num_val_tokens': len(val_np),
            'already_prepared': True
        }
        pickle.dump(meta, f)
        
def prepare_sampling():
    tokenized_data_filename = os.path.join(script_dir, configurator.dataset, 'tokenized_data.csv')
    sampling_lead_filename = os.path.join(script_dir, configurator.dataset, 'outputs', 'temp_sampling_lead.csv')

    with open(tokenized_data_filename, 'r') as in_file, open(sampling_lead_filename, 'w') as out_file:
        for event in in_file:
            first_event = event.split(' 4 ')[0]
            first_event += ' 4\n'
            out_file.write(first_event)
            
prepare_training()