import pickle
import sys
import csv
import numpy as np
from pathlib import Path
from file_read_backwards import FileReadBackwards

import dictionary
import pTokenizerModule as pTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
import configurator

script_dir = Path(__file__).resolve().parent

num_events_total = -1
num_train_events = -1
num_val_events = -1
num_test_events = -1
max_sequence_length = -1

def count_lines(csv_filename):
    # This takes about a minute to run on 100M events. Fastest way I can think of.
    with open(csv_filename, 'rb') as f:
        return sum(buf.count(b'\n') for buf in iter(lambda: f.read(1024 * 1024), b''))
    
def determine_max_sequence_length(csv_filename):
    # Since the csv is uniform, simply t first line can provide the max sequence length
    with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        first_line = next(reader)
        return len(first_line[0].split())

def concat_tokenized_batches():
    # All the tokenized batches will be uniform so we only need to concatenate them.
    temp_data_dir = script_dir / 'data' / configurator.dataset / 'temp'
    tokenized_data_filename = script_dir / 'data' / configurator.dataset / 'tokenized_data.csv'
    
    tokenized_csv_files = sorted(
        [f for f in temp_data_dir.iterdir() if f.name.startswith("tokenized_batch_") and f.name.endswith(".csv")],
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    with open(tokenized_data_filename, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.writer(outfile)

        for file in tokenized_csv_files:
            file_path = Path(temp_data_dir, file)
            print(f"Processing {file_path}...")

            with open(file_path, "r", encoding="utf-8") as infile:
                reader = csv.reader(infile)
                writer.writerows(reader)

def bin_data():
    global num_events_total
    global num_train_events
    global num_val_events
    global num_test_events
    
    temp_data_dir = script_dir / 'data' / configurator.dataset / 'temp'
    input_data_filename = script_dir / 'data' / configurator.dataset / 'data.csv'
    tokenized_data_filename = script_dir / 'data' / configurator.dataset / 'tokenized_data.csv'
    train_bin_filename = script_dir / 'data' / configurator.dataset / 'outputs' / 'train.bin'
    val_bin_filename = script_dir / 'data' / configurator.dataset / 'outputs' / 'val.bin'
    test_tokenized_bin_filename = script_dir / 'data' / configurator.dataset / 'outputs' / 'test_tokenized.bin'
    test_real_bin_filename = script_dir / 'data' / configurator.dataset / 'outputs' / 'test_real.bin'
    
    num_events_total = count_lines(tokenized_data_filename)
    # create the train, val, and test splits
    # The test split should be 10% or 100k events, whichever is smaller
    num_test_events = min(int(num_events_total * 0.1), 100000)
    num_events_unreserved = num_events_total - num_test_events
    # We use a 9:1 train:val split for the remaining events
    num_train_events = int(num_events_unreserved * 0.9)
    num_val_events = num_events_unreserved - num_train_events
    
    with open(tokenized_data_filename, 'r') as f, open(train_bin_filename, 'wb') as train_out, open(val_bin_filename, 'wb') as val_out, open(test_tokenized_bin_filename, 'wb') as test_tokenized_out:
        for i, line in enumerate(f):
            values = [int(x) for x in line.strip().split()]
            arr = np.array(values, dtype=np.uint16)
            
            if i <= num_train_events:
                arr.tofile(train_out)
            elif i <= num_train_events + num_val_events:
                arr.tofile(val_out)
            else:
                arr.tofile(test_tokenized_out)
    
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
            
            # This is a risky way of being this, but it works
            if len(accumulated_data) >= num_test_events:
                break
        
        max_sequence_length = determine_max_sequence_length(Path(temp_data_dir, "tokenized_batch_0.csv"))
        max_num_particles = int((max_sequence_length - 2) / 7)
        
        # Pad data because np.array requires uniform length
        padded_accumulated_data = [row + [0] * ((max_num_particles * 5) - len(row)) for row in accumulated_data]
        padded_accumulated_data_np = np.array(padded_accumulated_data)
        padded_accumulated_data_np.flatten().tofile(test_real_bin_filename)

def generate_leading_particle_information():
    # Output will be num_particles, pdgid, e, px, py, pz, eta, theta, phi
    temp_data_dir = script_dir / 'data' / configurator.dataset / 'temp'
    tokenized_data_filename = script_dir / 'data' / configurator.dataset / 'tokenized_data.csv'
    test_real_bin_filename = script_dir / 'data' / configurator.dataset / 'outputs' / 'test_real.bin'
    real_leading_test_particles_filename = Path(script_dir, 'data', configurator.dataset, 'outputs', 'real_leading_test_particles.csv')
    
    num_events_total = count_lines(tokenized_data_filename)
    # create the train, val, and test splits
    # The test split should be 10% or 100k events, whichever is smaller
    num_test_events = min(int(num_events_total * 0.1), 100000)
    num_events_unreserved = num_events_total - num_test_events
    # We use a 9:1 train:val split for the remaining events
    num_train_events = int(num_events_unreserved * 0.9)
    num_val_events = num_events_unreserved - num_train_events
    
    max_sequence_length = determine_max_sequence_length(Path(temp_data_dir, "tokenized_batch_0.csv"))
    max_num_particles = int((max_sequence_length - 2) / 7)
    
    real_bin_data = np.memmap(test_real_bin_filename, dtype=float, mode='r', shape=(num_test_events, max_num_particles * 5))
    
    with open(real_leading_test_particles_filename, 'w') as out_file:
        for event in real_bin_data:
            particles = event.reshape((max_num_particles, 5))
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
            theta = np.arccos(pz / r)
            phi = np.arctan2(py, px)
            eta = -np.log(np.tan(theta / 2))
            
            out_file.write(f'{len(secondaries)} {int(pdgid)} {e} {px} {py} {pz} {eta:.5f} {theta:.5f} {phi:.5f}\n')

def prepare_dataset():
    global max_sequence_length
    
    # Ensure the outputs directory exists
    input_data_filename = Path(script_dir, 'data', configurator.dataset, 'data.csv')
    dictionary_filename = Path(script_dir, 'data', configurator.dataset, 'dictionary.json')
    temp_data_dir = script_dir / 'data' / configurator.dataset / 'temp'
    temp_data_dir_as_filename = Path(script_dir, 'data', configurator.dataset, 'temp', 'something.csv')
    humanized_dictionary_filename = Path(script_dir, 'data', configurator.dataset, 'humanized_dictionary.txt')
    meta_filename = Path(script_dir, 'data', configurator.dataset, 'outputs', 'meta.pkl')
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

    dictionary.update_dictionary_particle_list(input_data_filename, dictionary_filename)
    dictionary.output_humanized_dictionary(humanized_dictionary_filename)
    pTokenizer.tokenize_data(dictionary_filename.as_posix(), input_data_filename.as_posix(), temp_data_dir_as_filename.as_posix())
    
    # The tokenizer generates a bunch of files which need to be concatenated
    concat_tokenized_batches()
    
    # Then we go through and bin the concatenated file
    bin_data()
    
    # Generate leading particle information here since this is needed per dataset, not per config
    generate_leading_particle_information()
    
    max_sequence_length = determine_max_sequence_length(Path(temp_data_dir, "tokenized_batch_0.csv"))
    
    print("----------------------------------------")
    print("Data information:")
    print(f"Vocab size: {dictionary.get_vocab_size():,} tokens")
    print(f"Total events: {num_events_total:,} events")
    print(f"Train has: {num_train_events:,} events")
    print(f"Train has: {(num_train_events * max_sequence_length):,} tokens")
    print(f"Val has: {num_val_events:,} events")
    print(f"Val has: {(num_val_events * max_sequence_length):,} tokens")
    print(f"Test has: {num_test_events:,} events")
    print(f"Test has: {(num_test_events * max_sequence_length):,} tokens")
    print(f"Particles per event: {int((max_sequence_length - 2) / 7)} particles")
    print(f"Max sequence length: {max_sequence_length} tokens")
    print("----------------------------------------")

    with open(meta_filename, 'wb') as f:
        meta = {
            'vocab_size': dictionary.get_vocab_size(),
            'total_events': num_events_total,
            'num_train_events': num_train_events,
            'num_train_tokens': num_train_events * max_sequence_length,
            'num_val_events': num_val_events,
            'num_val_tokens': num_val_events * max_sequence_length,
            'num_test_events': num_test_events,
            'num_test_tokens': num_test_events * max_sequence_length,
            'num_particles_per_event': int((max_sequence_length - 2) / 7),
            'max_sequence_length': max_sequence_length,
            'already_prepared': True
        }
        pickle.dump(meta, f)
        
if __name__ == "__main__":
    prepare_dataset()