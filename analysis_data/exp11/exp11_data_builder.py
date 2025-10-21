import pickle
import shutil
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent

input_preparation_dir                       = script_dir / 'data' / 'preparation_10M_9'
input_meta_filepath                         = input_preparation_dir / 'meta.pkl'
input_dictionary_filepath                   = input_preparation_dir / 'dictionary.json'
input_humanized_dictionary_filepath         = input_preparation_dir / 'humanized_dictionary.txt'
input_real_verbose_test_particles_filepath  = input_preparation_dir / 'real_verbose_test_particles.csv'
input_train_dataset_filepath                = input_preparation_dir / 'train.bin'
input_val_dataset_filepath                  = input_preparation_dir / 'val.bin'
input_test_tokenized_dataset_filepath       = input_preparation_dir / 'test_tokenized.bin'
input_test_real_dataset_filepath            = input_preparation_dir / 'test_real.bin'

def create_preparation(training_dataset_size, preparation_name):
    print(f'Building dataset with {training_dataset_size} training events.')
    
    output_preparation_dir                          = script_dir / 'data' / f'preparation_exp11_{preparation_name}'
    output_preparation_dir.mkdir(parents=True, exist_ok=True)
    output_meta_filepath                         = output_preparation_dir / 'meta.pkl'
    output_dictionary_filepath                   = output_preparation_dir / 'dictionary.json'
    output_humanized_dictionary_filepath         = output_preparation_dir / 'humanized_dictionary.txt'
    output_real_verbose_test_particles_filepath  = output_preparation_dir / 'real_verbose_test_particles.csv'
    output_train_dataset_filepath                = output_preparation_dir / 'train.bin'
    output_val_dataset_filepath                  = output_preparation_dir / 'val.bin'
    output_test_tokenized_dataset_filepath       = output_preparation_dir / 'test_tokenized.bin'
    output_test_real_dataset_filepath            = output_preparation_dir / 'test_real.bin'
    
    # Copy the validation dataset
    common_validation_dataset.tofile(output_val_dataset_filepath)
    # Create and copy the test datasets
    input_test_tokenized_dataset = np.memmap(input_test_tokenized_dataset_filepath, dtype=np.uint16, mode='r')
    input_test_tokenized_dataset.tofile(output_test_tokenized_dataset_filepath)
    input_test_real_dataset = np.memmap(input_test_real_dataset_filepath, dtype=np.uint16, mode='r')
    input_test_real_dataset.tofile(output_test_real_dataset_filepath)
    # Create and copy the training dataset
    training_data = input_train_dataset[:training_dataset_size]
    training_data.tofile(output_train_dataset_filepath)
    
    # Update the meta file for this preparation to be accurate to its information
    with open(output_meta_filepath, 'wb') as f:
        new_meta = {
            'vocab_size': meta['vocab_size'],
            'total_events': training_data.shape[0] + common_validation_dataset.shape[0] + input_test_tokenized_dataset.shape[0],
            'num_train_events': training_data.shape[0],
            'num_train_tokens': training_data.shape[0] * meta['max_sequence_length'],
            'num_val_events': common_validation_dataset.shape[0],
            'num_val_tokens': common_validation_dataset.shape[0] * meta['max_sequence_length'],
            'num_test_events': input_test_tokenized_dataset.shape[0],
            'num_test_tokens': input_test_tokenized_dataset.shape[0] * meta['max_sequence_length'],
            'num_particles_per_event': meta['num_particles_per_event'],
            'max_sequence_length': meta['max_sequence_length'],
            'already_prepared': True
        }
        pickle.dump(new_meta, f)
        
    # Copy the dictionary, humanized dictionary, and real verbose test particles files
    shutil.copy(input_dictionary_filepath, output_dictionary_filepath)
    shutil.copy(input_humanized_dictionary_filepath, output_humanized_dictionary_filepath)
    shutil.copy(input_real_verbose_test_particles_filepath, output_real_verbose_test_particles_filepath)
        
def validate_preparation(training_dataset_size, preparation_name):
    print(f'Validating preparation for {preparation_name} with dataset size {training_dataset_size}.')
    
    output_preparation_dir                          = script_dir / 'data' / preparation_name
    output_preparation_dir.mkdir(parents=True, exist_ok=True)
    output_train_dataset_filepath                = output_preparation_dir / 'train.bin'
    output_val_dataset_filepath                  = output_preparation_dir / 'val.bin'
    
    train_dataset = np.memmap(output_train_dataset_filepath, dtype=np.uint16, mode='r')
    train_dataset = train_dataset.reshape(-1, meta['max_sequence_length'])
    val_dataset = np.memmap(output_val_dataset_filepath, dtype=np.uint16, mode='r')
    val_dataset = val_dataset.reshape(-1, meta['max_sequence_length'])
    
    if train_dataset.shape[0] != training_dataset_size:
        raise ValueError(f'Training dataset size mismatch: expected {training_dataset_size}, got {train_dataset.shape[0]}')
    if val_dataset.shape[0] != 100_000:
        raise ValueError(f'Validation dataset size mismatch: expected 100000, got {val_dataset.shape[0]}')

if __name__ == '__main__':
    with open(input_meta_filepath, 'rb') as f:
        meta = pickle.load(f)

    training_dataset_sizes = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
    preparation_name_suffixes = ['100k', '500k', '1M', '2M', '5M']

    # Create the validation dataset all models will use
    input_validation_dataset = np.memmap(input_val_dataset_filepath, dtype=np.uint16, mode='r')
    input_validation_dataset = input_validation_dataset.reshape(-1, meta['max_sequence_length'])
    common_validation_dataset = input_validation_dataset[:100_000]

    input_train_dataset = np.memmap(input_train_dataset_filepath, dtype=np.uint16, mode='r')
    input_train_dataset = input_train_dataset.reshape(-1, meta['max_sequence_length'])

    for training_dataset_size, preparation_name in zip(training_dataset_sizes, preparation_name_suffixes):
        create_preparation(training_dataset_size, preparation_name)
        
    for training_dataset_size, preparation_name in zip(training_dataset_sizes, preparation_name_suffixes):
        validate_preparation(training_dataset_size, f'preparation_exp11_{preparation_name}')