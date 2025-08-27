
import math
import csv
import numpy as np
from numba import njit, float64
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal

from particle import Particle
from particle import PDGID
import vector

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

def determine_bin_index(i, event, type_str: str, type_offset: int) -> int:
    try:
        type_pos = dictionary.tokenization_schema.index(type_str)
        return event[i + type_pos] - type_offset
    except ValueError:
        return None

def get_bin_median(bins, bin_idx):
    return (bins[bin_idx - 1] + bins[bin_idx]) / 2

def quantile_detokenize(tokens, bin_edges, token_min=0):
    indices = tokens - token_min
    # Clip just in case
    indices = np.clip(indices, 0, len(bin_edges) - 2)
    # Use bin centers (can replace with per-bin means for more precision)
    return 0.5 * (bin_edges[indices] + bin_edges[indices + 1])

def untokenize_token_type(type_str: str, bins, bin_idx: int):
    tokenization_function = 'linear'
    if 'tokenization' in dictionary.dictionary_data[type_str + '_bin_data']:
        tokenization_function = dictionary.dictionary_data[type_str + '_bin_data']['tokenization']
    
    if tokenization_function == 'linear':
        return get_bin_median(bins, bin_idx)
    elif tokenization_function == 'gaussian':
        """
        Decode tokenized values back to approximate original values
        using bin centers.

        Parameters:
            tokens (np.ndarray): Array of token indices (0-based).
            bin_edges (np.ndarray): Bin edges used during digitization.

        Returns:
            np.ndarray: Decoded values.
        """
        # Bin centers: len = len(bin_edges) + 1 - 1 = len(bin_edges)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # Clip tokens to valid range
        tokens = np.clip(bin_idx, 0, len(bin_centers) - 1)
        return bin_centers[tokens]
    elif tokenization_function == 'quantile':
        quantile_edges_filepath = dictionary.dictionary_filename.parent / f'pt_quantile_edges.npy'
        
        quantile_edges = np.load(quantile_edges_filepath)
        return quantile_detokenize(bin_idx, quantile_edges)

def untokenize_data(input_data_filepath, output_data_filepath):
    untokenized_data = []
    with open(input_data_filepath) as input_file:
        for event_str in input_file:
            event = list(map(int, event_str.strip().split()))
            untokenized_event = untokenize_event(event)
            untokenized_data.append(untokenized_event)
    
    with open(output_data_filepath, 'w') as output_file:
        for event in untokenized_data:
            event_np = np.array(event, dtype=np.float64)
            event_np.resize(int(len(event_np) / NUM_FEATURES_PER_PARTICLE_RAW), NUM_FEATURES_PER_PARTICLE_RAW)
            for particle in event_np:
                pdgid, energy, px, py, pz = particle
                output_file.write(f"{int(pdgid)} {energy:.5f} {px:.5f} {py:.5f} {pz:.5f};")
            output_file.write('\n')

def untokenize_event(event: list[int]) -> list[float]:
    """
    Untokenize a single event.
    """
    
    untokenized_event = []
    for i in range(0, len(event), len(dictionary.tokenization_schema)):
        pdgid_idx = determine_bin_index(i, event, 'pdgid', dictionary.PDGID_OFFSET)
        energy_bin_idx = determine_bin_index(i, event, 'energy', dictionary.ENERGY_OFFSET)
        pt_bin_idx = determine_bin_index(i, event, 'pt', dictionary.PT_OFFSET)
        eta_bin_idx = determine_bin_index(i, event, 'eta', dictionary.ETA_OFFSET)
        theta_bin_idx = determine_bin_index(i, event, 'theta', dictionary.THETA_OFFSET)
        phi_bin_idx = determine_bin_index(i, event, 'phi', dictionary.PHI_OFFSET)
        px_bin_idx = determine_bin_index(i, event, 'px', dictionary.PX_OFFSET)
        py_bin_idx = determine_bin_index(i, event, 'py', dictionary.PY_OFFSET)
        pz_bin_idx = determine_bin_index(i, event, 'pz', dictionary.PZ_OFFSET)

        # We can reasonably assume pdgid exists in the tokenization since it is needed to have a proper particle.
        pdgid = dictionary.pdgids[str(pdgid_idx)]
        
        energy = None
        pt = None
        eta = None
        theta = None
        phi = None
        px = None
        py = None
        pz = None
        
        if energy_bin_idx is not None:
            energy = untokenize_token_type('e', dictionary.e_bins, energy_bin_idx)
        if pt_bin_idx is not None:
            pt = untokenize_token_type('pt', dictionary.pt_bins, pt_bin_idx)
        if eta_bin_idx is not None:
            eta = untokenize_token_type('eta', dictionary.eta_bins, eta_bin_idx)
        if theta_bin_idx is not None:
            theta = untokenize_token_type('theta', dictionary.theta_bins, theta_bin_idx)
        if phi_bin_idx is not None:
            phi = untokenize_token_type('phi', dictionary.phi_bins, phi_bin_idx)
        if px_bin_idx is not None:
            px = untokenize_token_type('px', dictionary.px_bins, px_bin_idx)
        if py_bin_idx is not None:
            py = untokenize_token_type('py', dictionary.py_bins, py_bin_idx)
        if pz_bin_idx is not None:
            pz = untokenize_token_type('pz', dictionary.pz_bins, pz_bin_idx)

        particle = Particle.from_pdgid(pdgid)
        
        if pt_bin_idx is not None and eta_bin_idx is not None and phi_bin_idx is not None:
            particle_vector = vector.obj(mass=particle.mass, pt=pt, eta=eta, phi=phi)
        elif energy_bin_idx is not None and theta_bin_idx is not None and phi_bin_idx is not None:
            particle_vector = vector.obj(energy=energy, theta=theta, phi=phi)
        elif energy_bin_idx is not None and px_bin_idx is not None and py_bin_idx is not None and pz_bin_idx is not None:
            particle_vector = vector.obj(energy=energy, px=px, py=py, pz=pz)
        else:
            raise ValueError("Not enough information to create a vector for the particle.")
        
        untokenized_event += [
            int(pdgid),
            particle_vector.energy,
            particle_vector.px,
            particle_vector.py,
            particle_vector.pz
        ]
    
    return untokenized_event

# Main can also do everything in case we only want to untokenize the data.
if __name__ == "__main__":
    untokenize_data(script_dir / 'generated_samples' / 'model_5_10_2' / 'sampling_0' / 'filtered_samples.csv', script_dir / 'generated_samples' / 'model_5_10_2' / 'sampling_0' / 'untokenized_samples.csv')