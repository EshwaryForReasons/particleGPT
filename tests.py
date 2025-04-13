import time
import json
import cupy as cp
import numpy as np
from numba import jit, njit
from numba.typed import List
from pathlib import Path

import dictionary
import pTokenizerModule as pTokenizer

script_dir = Path(__file__).resolve().parent

dictionary_filename = script_dir / 'data' / 'dataset_5_2' / 'dictionary.json'
sampling_storage_dir = script_dir / 'generated_samples' / 'dataset_5_2_1' / 'sampling_4'
generated_samples_filename = sampling_storage_dir / 'generated_samples.csv'
filtered_samples_filename = sampling_storage_dir / 'filtered_samples.csv'

def convert_tokenized_into_generated_samples():
    tokenized_data = []
    print("Starting data loading.")
    with open(generated_samples_filename) as gen_samples_file:
        for event in gen_samples_file:
            event = [int(x) for x in event.strip().split()]
            index = event.index(2)
            event = event[:index + 1]
            tokenized_data.append(event)
    print("Finished data loading.")
    
    print("Starting data output.")
    with open(generated_samples_filename, 'w') as gen_samples_file:
        for event in tokenized_data:
            event = [str(x) for x in event]
            event = ' '.join(event)
            gen_samples_file.write(event + '\n')
    print("Finished data output.")

def cpp_filter_version():
    start_time = time.perf_counter()
    pTokenizer.filter_data(dictionary_filename.as_posix(), generated_samples_filename.as_posix(), filtered_samples_filename.as_posix())
    end_time = time.perf_counter()
    delta_time = end_time - start_time
    print(f"Time taken for filtering: {delta_time:.6f} seconds (C++)")
    return delta_time

def python_filter_version():
    start_time = time.perf_counter()
    
    # Load data
    tokenized_data = []
    with open(generated_samples_filename) as gen_samples_file:
        for event in gen_samples_file:
            event = [int(x) for x in event.strip().split()]
            tokenized_data.append(event)
            
    filtered_data = []
    
    # Ensure valid borders
    tokenized_data = [e for e in tokenized_data if e[0] == 1 and e[-1] == 2]
    
    # Remove special tokens
    tokenized_data = [[x for x in e if x not in [0, 1, 2, 3, 4]] for e in tokenized_data]
        
    # Ensure events are well formed
    tokenized_data = [e for e in tokenized_data if len(e) > 5 and len(e) % 5 == 0]
    
    # Ensure valid token ranges
    pdgid_offset_min = dictionary.PDGID_OFFSET
    pdgid_offset_max = dictionary.PDGID_OFFSET + len(dictionary.particles_index)
    energy_offset_min = dictionary.ENERGY_OFFSET
    energy_offset_max = dictionary.ENERGY_OFFSET + len(dictionary.e_bins)
    eta_offset_min = dictionary.ETA_OFFSET
    eta_offset_max = dictionary.ETA_OFFSET + len(dictionary.eta_bins)
    theta_offset_min = dictionary.THETA_OFFSET
    theta_offset_max = dictionary.THETA_OFFSET + len(dictionary.theta_bins)
    phi_offset_min = dictionary.PHI_OFFSET
    phi_offset_max = dictionary.PHI_OFFSET + len(dictionary.phi_bins)
    
    for event in tokenized_data:
        b_keep_event = True
        for i, token in enumerate(event):
            token_type_id = i % 5
            if token_type_id == 0:
                if token < pdgid_offset_min or token >= pdgid_offset_max:
                    b_keep_event = False
                    break
            elif token_type_id == 1:
                if token < energy_offset_min or token >= energy_offset_max:
                    b_keep_event = False
                    break
            elif token_type_id == 2:
                if token < eta_offset_min or token >= eta_offset_max:
                    b_keep_event = False
                    break
            elif token_type_id == 3:
                if token < theta_offset_min or token >= theta_offset_max:
                    b_keep_event = False
                    break
            elif token_type_id == 4:
                if token < phi_offset_min or token >= phi_offset_max:
                    b_keep_event = False
                    break
        
        if b_keep_event:
            filtered_data.append(event)
    
    # Output data
    with open(filtered_samples_filename, 'w') as filtered_file:
        for event in filtered_data:
            event = [str(x) for x in event]
            event = ' '.join(event)
            filtered_file.write(event + '\n')
    
    end_time = time.perf_counter()
    delta_time = end_time - start_time
    print(f"Time taken for filtering: {delta_time:.6f} seconds (Python)")
    return delta_time

# ------------------------------------------
# Numba optimized version
# ------------------------------------------

@njit
def validate_event(event, event_len):
    if event_len <= 5 or event_len % 5 != 0:
        return False

    for i in range(event_len):
        token = event[i]
        token_type_id = i % 5
        if token_type_id == 0:
            if token < dictionary.PDGID_OFFSET or token >= dictionary.PDGID_OFFSET + 68:
                return False
        elif token_type_id == 1:
            if token < dictionary.ENERGY_OFFSET or token >= dictionary.ENERGY_OFFSET + 140:
                return False
        elif token_type_id == 2:
            if token < dictionary.ETA_OFFSET or token >= dictionary.ETA_OFFSET + 81:
                return False
        elif token_type_id == 3:
            if token < dictionary.THETA_OFFSET or token >= dictionary.THETA_OFFSET + 126:
                return False
        elif token_type_id == 4:
            if token < dictionary.PHI_OFFSET or token >= dictionary.PHI_OFFSET + 126:
                return False
    return True

@njit
def numba_fast_filter(data, max_event_len):
    n_events = data.shape[0]
    filtered = np.full((n_events, max_event_len), -1, dtype=np.int32)
    count = 0

    for i in range(n_events):
        row = data[i]
        
        # Find where token 2 occurs
        end_idx = -1
        for j in range(len(row)):
            if row[j] == 2:
                end_idx = j
                break
        if end_idx == -1 or row[0] != 1:
            continue

        # Remove special tokens [0-4] and build cleaned list
        cleaned = np.empty(end_idx + 1, dtype=np.int32)
        k = 0
        for j in range(end_idx + 1):
            if 0 <= row[j] <= 4:
                continue
            cleaned[k] = row[j]
            k += 1

        # Validate
        if validate_event(cleaned, k):
            filtered[count, :k] = cleaned[:k]
            count += 1

    return filtered[:count]

def load_tokenized_data():
    with open(generated_samples_filename, 'r') as f:
        lines = f.readlines()
    tokenized = []
    max_len = 0
    for line in lines:
        tokens = list(map(int, line.strip().split()))
        if 2 in tokens:
            idx = tokens.index(2)
            tokens = tokens[:idx+1]
        else:
            continue
        tokenized.append(tokens)
        max_len = max(max_len, len(tokens))
    return tokenized, max_len

def output_data(filtered_data):
    with open(filtered_samples_filename, 'w') as f:
        for row in filtered_data:
            cleaned = row[row != -1]
            f.write(' '.join(map(str, cleaned)) + '\n')

def numba_optimized_pipeline():
    start_time = time.perf_counter()
    
    tokenized_data, max_len = load_tokenized_data()
    padded = np.full((len(tokenized_data), max_len), -1, dtype=np.int32)
    for i, row in enumerate(tokenized_data):
        padded[i, :len(row)] = row

    filtered = numba_fast_filter(padded, max_len)
    output_data(filtered)

    end_time = time.perf_counter()
    print(f"Optimized Numba version took {end_time - start_time:.6f} seconds")

# ------------------------------------------
# cupy version
# ------------------------------------------

def cupy_filter_version():
    start_time = time.perf_counter()
    
    PAD_TOKEN = -1

    # Step 1: Load and pad data
    tokenized_data = []
    with open(generated_samples_filename) as gen_samples_file:
        for event in gen_samples_file:
            tokens = [int(x) for x in event.strip().split()]
            tokenized_data.append(tokens)

    # Pad data to max length
    max_len = max(len(e) for e in tokenized_data)
    num_events = len(tokenized_data)

    padded_np = cp.full((num_events, max_len), PAD_TOKEN, dtype=cp.int32)
    lengths = cp.zeros(num_events, dtype=cp.int32)

    for i, event in enumerate(tokenized_data):
        padded_np[i, :len(event)] = cp.array(event, dtype=cp.int32)
        lengths[i] = len(event)

    # Step 2: Ensure valid borders (start==1 and end==2)
    start_valid = padded_np[:, 0] == 1
    end_indices = lengths - 1
    end_tokens = padded_np[cp.arange(num_events), end_indices]
    end_valid = end_tokens == 2
    mask_valid_borders = start_valid & end_valid

    # Step 3: Remove special tokens [0,1,2,3,4]
    special_tokens = cp.array([0, 1, 2, 3, 4])
    mask_not_special = ~cp.isin(padded_np, special_tokens)
    cleaned = cp.where(mask_not_special, padded_np, PAD_TOKEN)

    # Step 4: Recompute lengths after removing special tokens
    clean_lengths = (cleaned != PAD_TOKEN).sum(axis=1)

    # Step 5: Ensure well-formed events (len > 5 and len % 5 == 0)
    mask_well_formed = (clean_lengths > 5) & (clean_lengths % 5 == 0)

    # Step 6: Token-type specific range checks
    def type_mask(seq_len, token_type):
        indices = cp.arange(seq_len)
        return indices % 5 == token_type

    def check_token_range(data, token_type, min_val, max_val):
        positions = cp.arange(data.shape[1])
        mask = (positions % 5 == token_type)[None, :]
        tokens = cp.where(mask, data, 0)
        valid = ((tokens >= min_val) & (tokens < max_val)) | (data == PAD_TOKEN)
        return valid

    mask0 = check_token_range(cleaned, 0, dictionary.PDGID_OFFSET, dictionary.PDGID_OFFSET + len(dictionary.particles_index))
    mask1 = check_token_range(cleaned, 1, dictionary.ENERGY_OFFSET, dictionary.ENERGY_OFFSET + len(dictionary.e_bins))
    mask2 = check_token_range(cleaned, 2, dictionary.ETA_OFFSET, dictionary.ETA_OFFSET + len(dictionary.eta_bins))
    mask3 = check_token_range(cleaned, 3, dictionary.THETA_OFFSET, dictionary.THETA_OFFSET + len(dictionary.theta_bins))
    mask4 = check_token_range(cleaned, 4, dictionary.PHI_OFFSET, dictionary.PHI_OFFSET + len(dictionary.phi_bins))

    # Combine all masks
    all_valid_mask = cp.all(mask0 & mask1 & mask2 & mask3 & mask4, axis=1)

    # Final mask
    final_mask = mask_valid_borders & mask_well_formed & all_valid_mask

    # Filtered results
    filtered = cleaned[final_mask]
    filtered_lengths = clean_lengths[final_mask]

    # Step 7: Output
    with open(filtered_samples_filename, 'w') as f:
        filtered_cpu = filtered.get()
        filtered_lengths_cpu = filtered_lengths.get()
        for row, L in zip(filtered_cpu, filtered_lengths_cpu):
            tokens = [str(x) for x in row[:L] if x != PAD_TOKEN]
            f.write(" ".join(tokens) + "\n")
    
    end_time = time.perf_counter()
    delta_time = end_time - start_time
    print(f"Time taken for filtering: {delta_time:.6f} seconds (CuPy)")
    return delta_time

cupy_filter_version()