import pickle
import sys
from pathlib import Path

import dictionary
import pTokenizerModule as pTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
import configurator

script_dir = Path(__file__).resolve().parent

def prepare_dataset():
    # Ensure the outputs directory exists
    input_data_filename = Path(script_dir, 'data', configurator.dataset, 'data.csv')
    dictionary_filename = Path(script_dir, 'data', configurator.dataset, 'dictionary.json')
    tokenized_data_filename = Path(script_dir, 'data', configurator.dataset, 'tokenized_data.csv')
    humanized_dictionary_filename = Path(script_dir, 'data', configurator.dataset, 'humanized_dictionary.txt')
    train_bin_filename = Path(script_dir, 'data', configurator.dataset, 'outputs', 'train.bin')
    val_bin_filename = Path(script_dir, 'data', configurator.dataset, 'outputs', 'val.bin')
    test_bin_filename = Path(script_dir, 'data', configurator.dataset, 'outputs', 'test.bin')
    real_leading_test_particles_data_filename = Path(script_dir, 'data', configurator.dataset, 'outputs', 'real_leading_test_particles.csv')
    meta_filename = Path(script_dir, 'data', configurator.dataset, 'outputs', 'meta.pkl')
    Path(train_bin_filename).parent.mkdir(parents=True, exist_ok=True)

    # Only prepare if we haven't already prepared the data
    if meta_filename.exists():
        with open(meta_filename, 'rb') as f:
            meta = pickle.load(f)
            if meta['already_prepared']:
                print("Data already prepared")
                return

    dictionary.update_dictionary_particle_list(input_data_filename, dictionary_filename)
    dictionary.output_humanized_dictionary(humanized_dictionary_filename)
    pTokenizer.load_dictionary(dictionary_filename.as_posix())
    pTokenizer.load_raw_data(input_data_filename.as_posix())
    pTokenizer.tokenize_data(tokenized_data_filename.as_posix())

    # create the train, val, and test splits
    num_events_total = pTokenizer.get_num_tokenized_events()
    # The test split should be 10% or 100k events, whichever is smaller
    num_test_events = min(int(num_events_total * 0.1), 100000)
    num_events_unreserved = num_events_total - num_test_events
    # We use a 9:1 train:val split for the remaining events
    num_train_events = int(num_events_unreserved * 0.9)
    num_val_events = num_events_unreserved - num_train_events
    
    pTokenizer.output_split_bins(num_train_events, num_val_events, num_test_events, train_bin_filename.as_posix(), val_bin_filename.as_posix(), test_bin_filename.as_posix())
    pTokenizer.output_real_leading_particle_information(real_leading_test_particles_data_filename.as_posix(), num_test_events)
    
    print("----------------------------------------")
    print("Data information:")
    print(f"Vocab size: {dictionary.get_vocab_size():,} tokens")
    print(f"Total events: {num_events_total:,} events")
    print(f"Train has: {num_train_events:,} events")
    print(f"Train has: {pTokenizer.get_num_train_tokens():,} tokens")
    print(f"Val has: {num_val_events:,} events")
    print(f"Val has: {pTokenizer.get_num_val_tokens():,} tokens")
    print(f"Test has: {num_test_events:,} events")
    print(f"Test has: {pTokenizer.get_num_test_tokens():,} tokens")
    print(f"Max sequence length: {pTokenizer.get_max_sequence_length()} tokens")
    print("----------------------------------------")

    with open(meta_filename, 'wb') as f:
        meta = {
            'vocab_size': dictionary.get_vocab_size(),
            'total_events': num_events_total,
            'num_train_events': num_train_events,
            'num_train_tokens': pTokenizer.get_num_train_tokens(),
            'num_val_events': num_val_events,
            'num_val_tokens': pTokenizer.get_num_val_tokens(),
            'num_test_events': num_test_events,
            'num_test_tokens': pTokenizer.get_num_test_tokens(),
            'max_sequence_length': pTokenizer.get_max_sequence_length(),
            'already_prepared': True
        }
        pickle.dump(meta, f)