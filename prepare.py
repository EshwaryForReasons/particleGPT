import pickle
import csv
import numpy as np
from pathlib import Path
from file_read_backwards import FileReadBackwards
import time

from dictionary import Dictionary
import pTokenizerModule as pTokenizer
import configurator as conf
import pUtil

script_dir = Path(__file__).resolve().parent
dictionary = Dictionary(script_dir / 'data' / conf.generic.preparation_name / 'dictionary.json')

num_events_total = -1
num_train_events = -1
num_val_events = -1
num_test_events = -1
max_sequence_length = -1

num_tokens_per_particle = -1
num_features_per_particle = -1

# A raw particle will always be pdgid, e, px, py, pz.
num_features_in_raw_particle = 5

# Much faster, but required a lot of memory.
def fast_bin(num_train_events, num_val_events, num_test_events):
    tokenized_data_filename = script_dir / 'data' / conf.generic.preparation_name / 'tokenized_data.csv'
    train_bin_filename = script_dir / 'data' / conf.generic.preparation_name / 'train.bin'
    val_bin_filename = script_dir / 'data' / conf.generic.preparation_name / 'val.bin'
    test_tokenized_bin_filename = script_dir / 'data' / conf.generic.preparation_name / 'test_tokenized.bin'
    
    print(f'Starting fast bin over {num_train_events + num_val_events + num_test_events} events.')
    
    # Step 1: Read the entire data file into memory as a 2D array
    data = np.loadtxt(tokenized_data_filename, dtype=np.uint16)

    # Step 2: Calculate the number of rows for each split
    train_data = data[:num_train_events]
    val_data = data[num_train_events:num_train_events + num_val_events]
    test_data = data[num_train_events + num_val_events:]

    # Step 3: Write each section to the corresponding binary file
    with open(train_bin_filename, 'wb') as train_out, open(val_bin_filename, 'wb') as val_out, open(test_tokenized_bin_filename, 'wb') as test_tokenized_out:
        train_data.tofile(train_out)
        val_data.tofile(val_out)
        test_data.tofile(test_tokenized_out)
    
    print('Finished fast binning.')

# Bins in place so doesn't require much memory, but is much slower.
def in_place_bin(num_train_events, num_val_events, num_test_events):
    tokenized_data_filename = script_dir / 'data' / conf.generic.preparation_name / 'tokenized_data.csv'
    train_bin_filename = script_dir / 'data' / conf.generic.preparation_name / 'train.bin'
    val_bin_filename = script_dir / 'data' / conf.generic.preparation_name / 'val.bin'
    test_tokenized_bin_filename = script_dir / 'data' / conf.generic.preparation_name / 'test_tokenized.bin'
    
    print(f'Starting in-place bin over {num_train_events + num_val_events + num_test_events} events.')
    with open(tokenized_data_filename, 'r') as f, open(train_bin_filename, 'wb') as train_out, open(val_bin_filename, 'wb') as val_out, open(test_tokenized_bin_filename, 'wb') as test_tokenized_out:
        for i, line in enumerate(f):
            arr = np.fromstring(line, sep=' ', dtype=np.uint16)
            
            if i <= num_train_events:
                arr.tofile(train_out)
            elif i <= num_train_events + num_val_events:
                arr.tofile(val_out)
            else:
                arr.tofile(test_tokenized_out)
    print('Finished in-place binning.')

def bin_data():
    global num_events_total
    global num_train_events
    global num_val_events
    global num_test_events
    global num_tokens_per_particle
    global num_features_per_particle
    global num_features_in_raw_particle
    
    input_data_filename = script_dir / 'data' / conf.generic.dataset
    tokenized_data_filename = script_dir / 'data' / conf.generic.preparation_name / 'tokenized_data.csv'
    test_real_bin_filename = script_dir / 'data' / conf.generic.preparation_name / 'test_real.bin'
    
    num_events_total = pUtil.count_rows(tokenized_data_filename)
    # create the train, val, and test splits
    # The test split should be 10% or 100k events, whichever is smaller
    num_test_events = min(int(num_events_total * 0.1), 100000)
    num_events_unreserved = num_events_total - num_test_events
    # We use a 9:1 train:val split for the remaining events
    num_train_events = int(num_events_unreserved * 0.9)
    num_val_events = num_events_unreserved - num_train_events
    
    fast_bin(num_train_events, num_val_events, num_test_events)
    
    with FileReadBackwards(input_data_filename) as f, open(test_real_bin_filename, 'wb') as test_real_out:
        accumulated_data = []
        while True:
            line = f.readline()
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
            if len(accumulated_data) >= num_test_events:
                break
        
        max_sequence_length = pUtil.count_columns(tokenized_data_filename)
        max_num_particles = int((max_sequence_length - 2) / num_tokens_per_particle)
        
        # Pad data because np.array requires uniform length
        padded_accumulated_data = [row + [0] * ((max_num_particles * 5) - len(row)) for row in accumulated_data]
        padded_accumulated_data_np = np.array(padded_accumulated_data)
        padded_accumulated_data_np.flatten().tofile(test_real_bin_filename)

def generate_leading_particle_information():
    global num_features_in_raw_particle

    # Output will be num_particles, pdgid, e, px, py, pz, pt, eta, theta, phi
    tokenized_data_filename = script_dir / 'data' / conf.generic.preparation_name / 'tokenized_data.csv'
    test_real_bin_filename = script_dir / 'data' / conf.generic.preparation_name / 'test_real.bin'
    real_leading_test_particles_filename = script_dir / 'data' / conf.generic.preparation_name / 'real_leading_test_particles.csv'
    
    num_events_total = pUtil.count_rows(tokenized_data_filename)
    # create the train, val, and test splits
    # The test split should be 10% or 100k events, whichever is smaller
    num_test_events = min(int(num_events_total * 0.1), 100000)
    num_events_unreserved = num_events_total - num_test_events
    # We use a 9:1 train:val split for the remaining events
    num_train_events = int(num_events_unreserved * 0.9)
    num_val_events = num_events_unreserved - num_train_events
    
    max_sequence_length = pUtil.count_columns(tokenized_data_filename)
    max_num_particles = int((max_sequence_length - 2) / num_tokens_per_particle)
    
    real_bin_data = np.memmap(test_real_bin_filename, dtype=float, mode='r', shape=(num_test_events, max_num_particles * num_features_in_raw_particle))
    with open(real_leading_test_particles_filename, 'w') as out_file:
        for event in real_bin_data:
            particles = event.reshape((max_num_particles, num_features_in_raw_particle))
            secondaries = particles[1:]
            # Find index of particle with the highest energy
            leading_particle_idx = np.argmax(secondaries[:, 1])
            leading_particle = secondaries[leading_particle_idx]
            secondaries = [s for s in secondaries if s[0] != 0]
            
            pdgid = leading_particle[0]
            e = leading_particle[1]
            px = leading_particle[2]
            py = leading_particle[3]
            pz = leading_particle[4]
            
            r = np.sqrt(px * px + py * py + pz * pz)
            pt = np.sqrt(px * px + py * py)
            theta = np.arccos(pz / r)
            phi = np.arctan2(py, px)
            eta = -np.log(np.tan(theta / 2))
            
            out_file.write(f'{len(secondaries)} {int(pdgid)} {e} {px} {py} {pz} {pt} {eta:.5f} {theta:.5f} {phi:.5f}\n')

def prepare_dataset():
    global max_sequence_length
    global num_tokens_per_particle
    global num_features_per_particle
    
    if dictionary.scheme == '':
        num_features_per_particle = 4
        num_tokens_per_particle = num_features_per_particle
    if dictionary.scheme == 'standard':
        num_features_per_particle = 5
        num_tokens_per_particle = num_features_per_particle + 2
    elif dictionary.scheme == 'no_eta':
        num_features_per_particle = 4
        num_tokens_per_particle = num_features_per_particle + 2
    elif dictionary.scheme == 'no_particle_boundaries':
        num_features_per_particle = 5
        num_tokens_per_particle = num_features_per_particle
    elif dictionary.scheme == 'paddingv2':
        num_features_per_particle = 5
        num_tokens_per_particle = num_features_per_particle + 2
    elif dictionary.scheme == 'neo_no_particle_boundaries':
        num_features_per_particle = 5
        num_tokens_per_particle = num_features_per_particle
    elif dictionary.scheme == 'neov2':
        num_features_per_particle = 4
        num_tokens_per_particle = num_features_per_particle
    
    # Ensure the outputs directory exists
    input_data_filename            = script_dir / 'data' / conf.generic.dataset
    meta_filename                  = script_dir / 'data' / conf.generic.preparation_name / 'meta.pkl'
    dictionary_filename            = script_dir / 'data' / conf.generic.preparation_name / 'dictionary.json'
    tokenized_data_filename        = script_dir / 'data' / conf.generic.preparation_name / 'tokenized_data.csv'
    humanized_dictionary_filename  = script_dir / 'data' / conf.generic.preparation_name / 'humanized_dictionary.txt'
    temp_data_dir                  = script_dir / 'data' / conf.generic.preparation_name / 'temp'
    Path(meta_filename).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_data_dir).mkdir(parents=True, exist_ok=True)
    
    # Only prepare if we haven't already prepared the data
    if meta_filename.exists():
        meta = None
        with open(meta_filename, 'rb') as f:
            meta = pickle.load(f)
            if meta['already_prepared']:
                print("Data already prepared")
                return

    # dictionary.update_dictionary_particle_list(input_data_filename, dictionary_filename)
    # dictionary.output_humanized_dictionary(humanized_dictionary_filename)
    # pTokenizer.tokenize_data(dictionary_filename.as_posix(), input_data_filename.as_posix(), temp_data_dir.as_posix())

    # # The tokenizer generates a bunch of files which need to be concatenated
    # print('Started concatenating tokenized files.')
    # tokenized_csv_files = sorted(
    #     [
    #         Path(temp_data_dir, f.name)
    #         for f in temp_data_dir.iterdir()
    #         if f.name.startswith("tokenized_batch_") and f.name.endswith(".csv")
    #     ],
    #     key=lambda x: int(x.stem.split('_')[-1])
    # )
    # pUtil.concat_csv_files(tokenized_csv_files, tokenized_data_filename)
    # print('Finished concatenating tokenized files.')
    
    # Then we go through and bin the concatenated file
    bin_data()
    
    # Generate leading particle information here since this is needed per dataset, not per config
    generate_leading_particle_information()
    
    max_sequence_length = pUtil.count_columns(tokenized_data_filename)
    
    print("----------------------------------------")
    print("Data information:")
    print(f"Vocab size: {dictionary.vocab_size:,} tokens")
    print(f"Total events: {num_events_total:,} events")
    print(f"Train has: {num_train_events:,} events")
    print(f"Train has: {(num_train_events * max_sequence_length):,} tokens")
    print(f"Val has: {num_val_events:,} events")
    print(f"Val has: {(num_val_events * max_sequence_length):,} tokens")
    print(f"Test has: {num_test_events:,} events")
    print(f"Test has: {(num_test_events * max_sequence_length):,} tokens")
    print(f"Particles per event: {int((max_sequence_length - 2) / num_tokens_per_particle)} particles")
    print(f"Max sequence length: {max_sequence_length} tokens")
    print("----------------------------------------")

    with open(meta_filename, 'wb') as f:
        meta = {
            'vocab_size': dictionary.vocab_size ,
            'total_events': num_events_total,
            'num_train_events': num_train_events,
            'num_train_tokens': num_train_events * max_sequence_length,
            'num_val_events': num_val_events,
            'num_val_tokens': num_val_events * max_sequence_length,
            'num_test_events': num_test_events,
            'num_test_tokens': num_test_events * max_sequence_length,
            'num_particles_per_event': int((max_sequence_length - 2) / num_tokens_per_particle),
            'max_sequence_length': max_sequence_length,
            'already_prepared': True
        }
        pickle.dump(meta, f)
        
if __name__ == "__main__":
    prepare_dataset()