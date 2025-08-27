
import math
import csv
import numpy as np
from numba import njit, float64
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from dictionary import Dictionary
import configurator as conf

script_dir = Path(__file__).resolve().parent
dictionary = Dictionary(script_dir / 'data' / conf.generic.preparation_name / 'dictionary.json')

NUM_FEATURES_PER_PARTICLE_RAW = 5

def analyze_dataset(dataset_filepath, delimiter = ';'):
    num_events = 0
    num_particles_max = 0

    with open(dataset_filepath, 'r', buffering=1024*1024) as file:
        for line in tqdm(file, desc="Analyzing dataset", unit=" events", unit_scale=True):
            line = line.rstrip('\n')
            if not line:
                continue
            num_events += 1
            col_count = 1 + line.count(delimiter)
            if col_count > num_particles_max:
                num_particles_max = col_count

    return num_events, num_particles_max

@njit("int64(float64, float64[:])", cache=True, nogil=True)
def custom_searchsorted(value, bins):
    if value < bins[0]:
        return 0
    elif value >= bins[-1]:
        return len(bins) - 1
    return np.searchsorted(bins, value, side='right')

def tokenize_event(event: list[float]) -> list[int]:
    """
    Tokenizes a single event.
    """
    tokenized_event = [dictionary.event_start_token]
    num_particles = len(event) // NUM_FEATURES_PER_PARTICLE_RAW
    
    for particle_idx in range(num_particles):
        pdgid  = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 0]
        energy = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 1]
        px     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 2]
        py     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 3]
        pz     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 4]
        
        r     = math.sqrt(px ** 2 + py ** 2 + pz ** 2)
        pt    = math.sqrt(px ** 2 + py ** 2)
        theta = math.acos(pz / r) if r != 0 else 0.0
        phi   = math.atan2(py, px)
        eta   = -math.log(math.tan(theta / 2)) if theta != 0 else 0.0
        
        # Eta constrain
        if abs(eta) > 4:
            return []
        
        particle_index = int(dictionary.pdgids_to_index[pdgid])
        for schema in dictionary.tokenization_schema:
            if schema == "pdgid":
                tokenized_event.append(particle_index + dictionary.PDGID_OFFSET)
            elif schema == "pt":
                tokenized_event.append(custom_searchsorted(pt, dictionary.pt_bins) + dictionary.PT_OFFSET)
            elif schema == "px":
                tokenized_event.append(custom_searchsorted(px, dictionary.px_bins) + dictionary.PX_OFFSET)
            elif schema == "py":
                tokenized_event.append(custom_searchsorted(py, dictionary.py_bins) + dictionary.PY_OFFSET)
            elif schema == "pz":
                tokenized_event.append(custom_searchsorted(pz, dictionary.pz_bins) + dictionary.PZ_OFFSET)
            elif schema == "energy":
                tokenized_event.append(custom_searchsorted(energy, dictionary.e_bins) + dictionary.ENERGY_OFFSET)
            elif schema == "eta":
                tokenized_event.append(custom_searchsorted(eta, dictionary.eta_bins) + dictionary.ETA_OFFSET)
            elif schema == "theta":
                tokenized_event.append(custom_searchsorted(theta, dictionary.theta_bins) + dictionary.THETA_OFFSET)
            elif schema == "phi":
                tokenized_event.append(custom_searchsorted(phi, dictionary.phi_bins) + dictionary.PHI_OFFSET)
            elif schema == "particle_start":
                tokenized_event.append(dictionary.particle_start_token)
            elif schema == "particle_end":
                tokenized_event.append(dictionary.particle_end_token)
            else:
                raise RuntimeError(f"pTokenizer: Unknown tokenization schema: {schema}")
            
    tokenized_event.append(dictionary.event_end_token)
    return tokenized_event

def tokenize_events_in_range(input_data_filepath, output_data_filepath, start_event_idx, end_event_idx, num_particles_max):
    """
    Tokenizes events in the specified range from the input file and saves them to the output file.
    
    The events are padded to a max_sequence_length calculated using num_particles_max * num_tokens_per_particle + 2 for the
    event start and end tokens.
    """
    tokenized_events = []
    
    # Tokenize all events in range
    with open(input_data_filepath, 'r') as in_file:
        for idx, event_str in enumerate(in_file):
            if idx < start_event_idx:
                continue
            if idx >= end_event_idx:
                break
            
            event_str = list(map(float, (feature_str for particle_str in event_str.strip().split(';') for feature_str in particle_str.split())))
            tokenized_event = tokenize_event(event_str)
            if len(tokenized_event) > 0:
                tokenized_events.append(tokenized_event)
    
    # Pad to max_sequence_len
    padding_sequence = dictionary.get_padding_sequence()
    max_sequence_len = num_particles_max * dictionary.num_tokens_per_particle + 2

    padded_array = []
    for event_idx, tokenized_event in enumerate(tokenized_events):
        len_pad_required = max_sequence_len - len(tokenized_event)
        num_padding_sequences_required = len_pad_required // len(padding_sequence)
        padded_event = tokenized_event + padding_sequence * num_padding_sequences_required
        padded_array.append(padded_event)
    padded_array = np.array(padded_array, dtype=np.int32)
        
    # Output data to temp file, pending concatenation
    with open(output_data_filepath, 'w') as out_file:
        np.savetxt(out_file, padded_array, delimiter=' ', fmt='%d')

def tokenize_data_worker(thread_idx, input_data_filepath, temp_data_dir, start_event_idx, end_event_idx, num_particles_max):
    """
    Worker function for tokenizing data in parallel.
    """
    this_thread_out_file = temp_data_dir / f'tokenized_batch_{thread_idx}.csv'
    tokenize_events_in_range(input_data_filepath, this_thread_out_file, start_event_idx, end_event_idx, num_particles_max)
    
def batch_tokenize_data():
    """
    Handles the tokenization of the dataset in parallel using multiple processes.
    This function reads the dataset, analyzes it to determine the optimal number of threads,
    and then tokenizes the data in chunks, saving each chunk to a temporary file.
    
    These temporary files should be concatenated into a single output file using concatenate_tokenized_data.
    """
    # Load the data
    input_data_filepath         = script_dir / 'data' / conf.generic.dataset
    temp_data_dir               = script_dir / 'data' / conf.generic.preparation_name / 'temp'
    temp_data_dir.mkdir(parents=True, exist_ok=True)
    
    num_events, num_particles_max = analyze_dataset(input_data_filepath)
    MAX_EVENTS_PER_THREAD = 100_000
    
    # Generate chunks
    ranges = []
    for thread_id, start_idx in enumerate(range(0, num_events, MAX_EVENTS_PER_THREAD)):
        end_idx = min(start_idx + MAX_EVENTS_PER_THREAD, num_events)
        ranges.append((thread_id, start_idx, end_idx))

    print(f"Launching {len(ranges)} threads to process {num_events} events")

    with ProcessPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [
            executor.submit(tokenize_data_worker, thread_idx, input_data_filepath, temp_data_dir, start, end, num_particles_max)
            for thread_idx, start, end in ranges
        ]
        for f in futures:
            f.result()

def concatenate_tokenized_data():
    """
    Concatenates all tokenized data files in the temp directory into a single output file.
    """
    tokenized_data_filepath     = script_dir / 'data' / conf.generic.preparation_name / 'tokenized_data.csv'
    temp_data_dir               = script_dir / 'data' / conf.generic.preparation_name / 'temp'
    
    tokenized_csv_files = sorted(
        [
            Path(temp_data_dir, f.name)
            for f in temp_data_dir.iterdir()
            if f.name.startswith("tokenized_batch_") and f.name.endswith(".csv")
        ],
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    b_log_outputs = False
    with open(tokenized_data_filepath, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.writer(outfile)

        for file_path in tokenized_csv_files:
            if b_log_outputs:
                print(f"Processing {file_path}...")

            with open(file_path, "r", encoding="utf-8") as infile:
                reader = csv.reader(infile)
                writer.writerows(reader)

# Main can also do everything in case we only want to tokenize the data but not prepare
# it for training.
if __name__ == "__main__":
    batch_tokenize_data()
    concatenate_tokenized_data()