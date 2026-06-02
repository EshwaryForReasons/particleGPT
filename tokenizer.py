import shutil
import sys
import os
import math
import csv
import numpy as np
from numba import njit, float64
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from dictionary import Dictionary
import configurator as conf

script_dir = Path(__file__).resolve().parent

NUM_FEATURES_PER_PARTICLE_RAW = 5
N_WORKERS = 256
IO_BUFFER = 16 * 1024 * 1024  # 16 MB
COPY_BUFFER = 256 * 1024 * 1024  # 256 MB

@dataclass
class Paths:
    # Input raw data
    input_data_filepath: Path = None
    # Output tokenized data
    tokenized_data_filepath: Path = None
    # Directory to store temp tokenized files before concatenation
    temp_data_dir: Path = None
    

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

def make_byte_ranges(filepath, n_ranges):
    file_size = os.path.getsize(filepath)
    chunk_size = file_size // n_ranges

    ranges = []
    for i in range(n_ranges):
        start = i * chunk_size
        end = file_size if i == n_ranges - 1 else (i + 1) * chunk_size
        ranges.append((i, start, end))

    return ranges

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
        
        particle_index = int(dictionary.pdgids_to_index[int(pdgid)])
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

def tokenize_byte_range_worker(
    worker_idx,
    input_data_filepath,
    temp_data_dir,
    start_byte,
    end_byte,
    num_particles_max,
):
    output_data_filepath = temp_data_dir / f"tokenized_batch_{worker_idx}.csv"

    padding_sequence = dictionary.get_padding_sequence()
    max_sequence_len = num_particles_max * dictionary.num_tokens_per_particle + 2

    num_written = 0

    with open(input_data_filepath, "rb", buffering=IO_BUFFER) as in_file, \
         open(output_data_filepath, "w", buffering=IO_BUFFER) as out_file:

        if start_byte == 0:
            in_file.seek(0)
        else:
            # Check whether start_byte is already at the beginning of a line.
            in_file.seek(start_byte - 1)
            prev_byte = in_file.read(1)

            # If previous byte is not newline, we landed in the middle of an event,
            # so discard the partial line.
            if prev_byte != b"\n":
                in_file.readline()

        while in_file.tell() < end_byte:
            line = in_file.readline()
            if not line:
                break

            line = line.decode("utf-8").strip()
            if not line:
                continue

            event = list(map(
                float,
                (
                    feature_str
                    for particle_str in line.split(";")
                    for feature_str in particle_str.split()
                )
            ))

            tokenized_event = tokenize_event(event)

            if len(tokenized_event) == 0:
                continue

            len_pad_required = max_sequence_len - len(tokenized_event)

            if len_pad_required < 0:
                raise RuntimeError(
                    f"Tokenized event length {len(tokenized_event)} exceeds "
                    f"max_sequence_len {max_sequence_len}"
                )

            num_padding_sequences_required = len_pad_required // len(padding_sequence)

            padded_event = (
                tokenized_event
                + padding_sequence * num_padding_sequences_required
            )

            out_file.write(" ".join(map(str, padded_event)))
            out_file.write("\n")

            num_written += 1

    return worker_idx, num_written

def batch_tokenize_data(in_paths: Paths):
    """
    Handles tokenization using byte-range parallelism.

    Each worker seeks to a different region of the file, discards one possible
    partial first line, and then processes complete event lines.
    """
    in_paths.temp_data_dir.mkdir(parents=True, exist_ok=True)

    num_events, num_particles_max = analyze_dataset(in_paths.input_data_filepath)

    if dictionary.particle_count_override is not None:
        assert dictionary.particle_count_override >= num_particles_max, (
            "The particle_count_override in the dictionary must be >= maximum "
            "number of particles found in the dataset."
        )
        num_particles_max = dictionary.particle_count_override

    ranges = make_byte_ranges(in_paths.input_data_filepath, N_WORKERS)

    print(f"Found {num_events:,} events")
    print(f"Max particles/event: {num_particles_max:,}")
    print(f"Launching {N_WORKERS} workers")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [
            executor.submit(
                tokenize_byte_range_worker,
                worker_idx,
                in_paths.input_data_filepath,
                in_paths.temp_data_dir,
                start_byte,
                end_byte,
                num_particles_max,
            )
            for worker_idx, start_byte, end_byte in ranges
        ]

        total_written = 0

        for f in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing"):
            worker_idx, num_written = f.result()
            total_written += num_written

    print(f"Tokenized {total_written:,} valid events")

def concatenate_tokenized_data(in_paths: Paths):
    tokenized_csv_files = sorted(
        [
            f
            for f in in_paths.temp_data_dir.iterdir()
            if f.name.startswith("tokenized_batch_") and f.name.endswith(".csv")
        ],
        key=lambda x: int(x.stem.split("_")[-1])
    )

    with open(in_paths.tokenized_data_filepath, "wb", buffering=0) as outfile:
        for file_path in tqdm(tokenized_csv_files, desc="Concatenating"):
            with open(file_path, "rb", buffering=0) as infile:
                shutil.copyfileobj(infile, outfile, length=COPY_BUFFER)

# Main can also do everything in case we only want to tokenize the data but not prepare
# it for training.
if __name__ == "__main__":
    dictionary_filepath = sys.argv[1]
    dictionary = Dictionary(dictionary_filepath)
    
    relevant_paths = Paths(
        input_data_filepath     = script_dir / 'data' / 'raw' / dictionary.dataset_name,
        tokenized_data_filepath = script_dir / 'data' / 'tokenized' / dictionary.tokenization_name / 'tokenized_data.csv',
        temp_data_dir           = script_dir / 'data' / 'tokenized' / dictionary.tokenization_name / 'temp'
    )
    humanized_dictionary_filepath = script_dir / 'data' / 'tokenized' / dictionary.tokenization_name / 'humanized_dictionary.txt'
    
    dictionary.update_dictionary_particle_list(relevant_paths.input_data_filepath, dictionary_filepath)
    dictionary.output_humanized_dictionary(humanized_dictionary_filepath)
    
    batch_tokenize_data(relevant_paths)
    concatenate_tokenized_data(relevant_paths)