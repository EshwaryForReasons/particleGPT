import pickle
import csv
import numpy as np
from pathlib import Path
from file_read_backwards import FileReadBackwards
import time
from dataclasses import dataclass
import sys

from dictionary import Dictionary
import pUtil
import tokenizer

script_dir = Path(__file__).resolve().parent
dictionary = None

# A raw particle will always be pdgid, e, px, py, pz.
NUM_FEATURES_IN_RAW_PARTICLE = 5

@dataclass
class DatasetInfo:
    num_events_total: int       = -1
    num_train_events: int       = -1
    num_val_events: int         = -1
    num_test_events: int        = -1
    max_sequence_length: int    = -1
    max_num_particles: int      = -1

def get_dataset_info():
    """
    Returns a DatasetInfo object with the current dataset information.
    Helper because this information is required a lot.
    """
    
    assert dictionary != None, "Dictionary must be initialized before preparing the dataset."
    tokenized_data_filename = script_dir / 'data' / dictionary.preparation_name / 'tokenized_data.csv'
    
    num_events_total = pUtil.count_rows(tokenized_data_filename)
    # create the train, val, and test splits
    # The test split should be 10% or 100k events, whichever is smaller
    num_test_events = min(int(num_events_total * 0.1), 100000)
    num_events_unreserved = num_events_total - num_test_events
    # We use a 9:1 train:val split for the remaining events
    num_train_events = int(num_events_unreserved * 0.9)
    num_val_events = num_events_unreserved - num_train_events
    
    max_sequence_length=pUtil.count_columns(tokenized_data_filename)
    max_num_particles = int((max_sequence_length - 2) / dictionary.num_tokens_per_particle)
    
    return DatasetInfo(
        num_events_total=num_events_total,
        num_train_events=num_train_events,
        num_val_events=num_val_events,
        num_test_events=num_test_events,
        max_sequence_length=max_sequence_length,
        max_num_particles=max_num_particles
    )

def bin_data_fast():
    """
    Much faster than bin_data_in_place, but requires a lot of memory (depending on the dataset).
    This should be used by default whenever memory is not a concern.
    """
    tokenized_data_filename = script_dir / 'data' / dictionary.preparation_name / 'tokenized_data.csv'
    train_bin_filename = script_dir / 'data' / dictionary.preparation_name / 'train.bin'
    val_bin_filename = script_dir / 'data' / dictionary.preparation_name / 'val.bin'
    test_tokenized_bin_filename = script_dir / 'data' / dictionary.preparation_name / 'test_tokenized.bin'
    
    dataset_info = get_dataset_info()
    print(f'Starting fast bin over {dataset_info.num_train_events + dataset_info.num_val_events + dataset_info.num_test_events} events.')
    
    # Step 1: Read the entire data file into memory as a 2D array
    data = np.loadtxt(tokenized_data_filename, dtype=np.uint16)

    # Step 2: Calculate the number of rows for each split
    train_data = data[:dataset_info.num_train_events]
    val_data = data[dataset_info.num_train_events:dataset_info.num_train_events + dataset_info.num_val_events]
    test_data = data[dataset_info.num_train_events + dataset_info.num_val_events:]

    # Step 3: Write each section to the corresponding binary file
    with open(train_bin_filename, 'wb') as train_out, open(val_bin_filename, 'wb') as val_out, open(test_tokenized_bin_filename, 'wb') as test_tokenized_out:
        train_data.tofile(train_out)
        val_data.tofile(val_out)
        test_data.tofile(test_tokenized_out)
    
    print('Finished fast binning.')

def bin_data_in_place():
    """
    Bins in place so doesn't require much memory, but is much slower than bin_data_fast.
    This should only be used for large datasets where memory is a concern.
    """
    tokenized_data_filename = script_dir / 'data' / dictionary.preparation_name / 'tokenized_data.csv'
    train_bin_filename = script_dir / 'data' / dictionary.preparation_name / 'train.bin'
    val_bin_filename = script_dir / 'data' / dictionary.preparation_name / 'val.bin'
    test_tokenized_bin_filename = script_dir / 'data' / dictionary.preparation_name / 'test_tokenized.bin'
    
    dataset_info = get_dataset_info()
    
    print(f'Starting in-place bin over {dataset_info.num_train_events + dataset_info.num_val_events + dataset_info.num_test_events} events.')
    with open(tokenized_data_filename, 'r') as f, open(train_bin_filename, 'wb') as train_out, open(val_bin_filename, 'wb') as val_out, open(test_tokenized_bin_filename, 'wb') as test_tokenized_out:
        for i, line in enumerate(f):
            arr = np.fromstring(line, sep=' ', dtype=np.uint16)
            
            if i <= dataset_info.num_train_events:
                arr.tofile(train_out)
            elif i <= dataset_info.num_train_events + dataset_info.num_val_events:
                arr.tofile(val_out)
            else:
                arr.tofile(test_tokenized_out)
    print('Finished in-place binning.')

def bin_data():
    input_data_filename = dictionary.dataset_filepath
    test_real_bin_filename = script_dir / 'data' / dictionary.preparation_name / 'test_real.bin'
    
    bin_data_fast()
    
    dataset_info = get_dataset_info()
    with FileReadBackwards(input_data_filename) as in_file:
        accumulated_data = []
        while True:
            line = in_file.readline()
            arr = []
            particles = line.strip().split(";")
            
            b_ignore_this_event = False
            for particle in particles:
                particle = [float(x) for x in particle.split()]
                
                px = particle[2]
                py = particle[3]
                pz = particle[4]
                
                r = np.sqrt(px * px + py * py + pz * pz)
                theta = np.arccos(pz / r)
                eta = -np.log(np.tan(theta / 2))
                
                if np.abs(eta) > 4:
                    b_ignore_this_event = True
            
                arr.extend(particle)
            
            if not b_ignore_this_event:
                accumulated_data.append(arr)
            
            # This is a risky way of doing this, but it works
            if len(accumulated_data) >= dataset_info.num_test_events:
                break
        
        # Pad data because np.array requires uniform length
        padded_accumulated_data = [row + [0] * ((dataset_info.max_num_particles * 5) - len(row)) for row in accumulated_data]
        padded_accumulated_data_np = np.array(padded_accumulated_data)
        padded_accumulated_data_np.flatten().tofile(test_real_bin_filename)
            
def generate_verbose_particle_information():
    # Output will be num_particles, pdgid, e, px, py, pz, pt, eta, theta, phi
    
    test_real_bin_filename = script_dir / 'data' / dictionary.preparation_name / 'test_real.bin'
    real_verbose_test_particles_filename = script_dir / 'data' / dictionary.preparation_name / 'real_verbose_test_particles.csv'
    
    dataset_info = get_dataset_info()
    real_bin_data = np.memmap(test_real_bin_filename, dtype=float, mode='r', shape=(dataset_info.num_test_events, dataset_info.max_num_particles, NUM_FEATURES_IN_RAW_PARTICLE))
    
    NUM_FEATURES_PER_PARTICLE_VERBOSE = 9
    verbose_data = np.full(shape=(real_bin_data.shape[0], real_bin_data.shape[1], NUM_FEATURES_PER_PARTICLE_VERBOSE), fill_value=np.nan, dtype=np.float64)
    for idx_e, event in enumerate(real_bin_data):
        for idx_p, particle in enumerate(event):
            if particle[0] == 0:
                verbose_data[idx_e, idx_p] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                continue
            
            pdgid, e, px, py, pz = particle
            r = np.sqrt(px * px + py * py + pz * pz)
            r = np.sqrt(px * px + py * py + pz * pz)
            pt = np.sqrt(px * px + py * py)
            theta = np.arccos(pz / r) if r != 0 else 0
            phi = np.arctan2(py, px)
            eta = -np.log(np.tan(theta / 2))
            
            verbose_data[idx_e, idx_p] = [pdgid, e, px, py, pz, pt, eta, theta, phi]
    
    with open(real_verbose_test_particles_filename, 'w') as out_file:
        for event in verbose_data:
            for particle in event:
                pdgid, e, px, py, pz, pt, eta, theta, phi = particle
                if np.isnan(pdgid):
                    continue
                out_file.write(f'{int(pdgid)} {e:.5f} {px:.5f} {py:.5f} {pz:.5f} {pt:.5f} {eta:.5f} {theta:.5f} {phi:.5f};')
            out_file.write('\n')

def prepare_dataset():
    assert dictionary != None, "Dictionary must be initialized before preparing the dataset."
    # Ensure the outputs directory exists
    input_data_filepath            = dictionary.dataset_filepath
    meta_filepath                  = script_dir / 'data' / dictionary.preparation_name / 'meta.pkl'
    dictionary_filepath            = script_dir / 'data' / dictionary.preparation_name / 'dictionary.json'
    humanized_dictionary_filepath  = script_dir / 'data' / dictionary.preparation_name / 'humanized_dictionary.txt'
    temp_data_dir                  = script_dir / 'data' / dictionary.preparation_name / 'temp'
    Path(meta_filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_data_dir).mkdir(parents=True, exist_ok=True)
    
    # Only prepare if we haven't already prepared the data
    if meta_filepath.exists():
        meta = None
        with open(meta_filepath, 'rb') as f:
            meta = pickle.load(f)
            if meta['already_prepared']:
                print("Data already prepared")
                return

    dictionary.update_dictionary_particle_list(input_data_filepath, dictionary_filepath)
    dictionary.output_humanized_dictionary(humanized_dictionary_filepath)
    tokenizer.batch_tokenize_data()
    tokenizer.concatenate_tokenized_data()

    bin_data()
    generate_verbose_particle_information()
    
    dataset_info = get_dataset_info()
    
    print("----------------------------------------")
    print("Data information:")
    print(f"Vocab size: {dictionary.vocab_size:,} tokens")
    print(f"Total events: {dataset_info.num_events_total:,} events")
    print(f"Train has: {dataset_info.num_train_events:,} events")
    print(f"Train has: {(dataset_info.num_train_events * dataset_info.max_sequence_length):,} tokens")
    print(f"Val has: {dataset_info.num_val_events:,} events")
    print(f"Val has: {(dataset_info.num_val_events * dataset_info.max_sequence_length):,} tokens")
    print(f"Test has: {dataset_info.num_test_events:,} events")
    print(f"Test has: {(dataset_info.num_test_events * dataset_info.max_sequence_length):,} tokens")
    print(f"Particles per event: {int((dataset_info.max_sequence_length - 2) / dictionary.num_tokens_per_particle)} particles")
    print(f"Max sequence length: {dataset_info.max_sequence_length} tokens")
    print("----------------------------------------")

    with open(meta_filepath, 'wb') as f:
        meta = {
            'vocab_size': dictionary.vocab_size ,
            'total_events': dataset_info.num_events_total,
            'num_train_events': dataset_info.num_train_events,
            'num_train_tokens': dataset_info.num_train_events * dataset_info.max_sequence_length,
            'num_val_events': dataset_info.num_val_events,
            'num_val_tokens': dataset_info.num_val_events * dataset_info.max_sequence_length,
            'num_test_events': dataset_info.num_test_events,
            'num_test_tokens': dataset_info.num_test_events * dataset_info.max_sequence_length,
            'num_particles_per_event': int((dataset_info.max_sequence_length - 2) / dictionary.num_tokens_per_particle),
            'max_sequence_length': dataset_info.max_sequence_length,
            'already_prepared': True
        }
        pickle.dump(meta, f)
        
if __name__ == "__main__":
    dictionary_filepath = sys.argv[1]
    dictionary = Dictionary(dictionary_filepath)
    prepare_dataset()