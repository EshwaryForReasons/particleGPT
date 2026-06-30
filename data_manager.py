
import json
import numpy as np
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from particle import Particle

import paths
import particleGPT.quickmaths as quickmaths

NUM_FEATURES_PER_PARTICLE_RAW = 5



def _get_metadata_filename(data_filepath: Path):
    """For the sake of consistency"""
    return f'{data_filepath.name}.json'

def _load_geant4_dataset_csv(dataset_filepath, flattened_events=False, pad_token=np.nan):
    """
    Ultra-fast loader for Geant4 MC dataset files.
    a Geant4 dataset refers to data in the format "pdgid e px py pz; pdgid e px py pz; ...".
    Each event is a line in the file, and each particle in the event is separated by a semicolon.
    """
    
    if not dataset_filepath.exists():
        return np.empty((0, 0, 5), dtype=np.float64)

    all_data = []
    lengths = []

    with open(dataset_filepath, 'r') as f:
        for line in f:
            if ';' not in line:
                continue

            line = line.replace(';', ' ').strip()
            floats = np.fromstring(line, sep=' ')
            if floats.size == 0:
                continue

            reshaped = floats.reshape(-1, NUM_FEATURES_PER_PARTICLE_RAW)
            all_data.append(reshaped)
            lengths.append(reshaped.shape[0])

    if not all_data:
        return np.empty((0, 0, NUM_FEATURES_PER_PARTICLE_RAW), dtype=np.float64)

    max_particles = max(lengths)
    n_events = len(all_data)

    # Preallocate final array and copy using slice assignment
    result = np.full((n_events, max_particles, NUM_FEATURES_PER_PARTICLE_RAW), pad_token, dtype=np.float64)
    for i, (ev, length) in enumerate(zip(all_data, lengths)):
        result[i, :length, :] = ev

    if flattened_events:
        return result.reshape(n_events, -1)
    return result

def load_geant4_dataset(dataset_filepath: Path, flattened_events: bool = False, pad_token=np.nan):
    metadata_filepath = dataset_filepath.parent / _get_metadata_filename(dataset_filepath)
    if not metadata_filepath.exists():
        # @TODO: Metadata should not be required if the format is a .csv, but should be supported for faster loading
        raise RuntimeError("Metadata not found! Support for running this without accompanying metadata is planned for the future!")
    
    with metadata_filepath.open('r') as f:
        metadata = json.load(f)
    
    num_events = metadata['num_events']
    num_particles_max = metadata['num_particles_max']
    
    match dataset_filepath.suffix:
        case ".csv":
            data = _load_geant4_dataset_csv(dataset_filepath, flattened_events=flattened_events, pad_token=pad_token)
        case ".bin":
            # np.fromfile because we want a clone of the data; np.memmap would modify the data if we change it.
            data = np.fromfile(dataset_filepath, dtype=np.float64)
            data = data.reshape(num_events, num_particles_max, NUM_FEATURES_PER_PARTICLE_RAW)
        case ".npy":
            data = np.load(dataset_filepath)
            data = data.reshape(num_events, num_particles_max, NUM_FEATURES_PER_PARTICLE_RAW)
        case __:
            raise NotImplementedError("This file format is not supported!")

    return data

def save_geant4_dataset(dataset, output_filepath: Path):
    """
    Saves data to specified file format in the shape of (pdgid, e, px, py, pz)
    """
    
    num_events = len(dataset)
    num_particles_max = 0
    match output_filepath.suffix:
        case ".csv":
            with open(output_filepath, "w") as output:
                for event in dataset:
                    event_len = len(event)
                    if event_len % NUM_FEATURES_PER_PARTICLE_RAW != 0:
                        raise RuntimeError(f"Event length is incorrect, must be a multiple of {NUM_FEATURES_PER_PARTICLE_RAW}, current length = {len(event)}")
                    
                    num_particles = int(event_len // NUM_FEATURES_PER_PARTICLE_RAW)
                    if num_particles > num_particles_max:
                        num_particles_max = num_particles
                    
                    for start in range(0, event_len, NUM_FEATURES_PER_PARTICLE_RAW):
                        pdgid, energy, px, py, pz = event[start:start + 5]
                        output.write(f"{int(pdgid)} {energy:.5f} {px:.5f} {py:.5f} {pz:.5f};")
                        
                    output.write("\n")
        case ".bin":
            dataset = quickmaths.pad_jagged_1d_fast(dataset)
            dataset.tofile(output_filepath)
            num_particles_max = int(dataset.shape[1] // NUM_FEATURES_PER_PARTICLE_RAW)
        case ".npy":
            dataset = quickmaths.pad_jagged_1d_fast(dataset)
            np.save(output_filepath, dataset)
            num_particles_max = int(dataset.shape[1] // NUM_FEATURES_PER_PARTICLE_RAW)
        case __:
            raise NotImplementedError("This output format is not supported!")
        
    metadata = {
        "output_data_filepath": paths.project_relative_path(output_filepath),
        "num_events": num_events,
        "num_particles_max": num_particles_max
    }
    
    output_metadata_filepath = output_filepath.parent / _get_metadata_filename(output_filepath)
    with output_metadata_filepath.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Wrote metadata: {output_metadata_filepath}")
    

def load_verbose_dataset(dataset_filepath, flattened_events=False, pad_token=np.nan):
    """
    Ultra-fast loader for verbose dataset files.
    A verbose dataset file refers to data in the format "pdgid e px py pz pt eta theta phi; pdgid e px py pz pt eta theta phi; ...".
    Each event is a line in the file, and each particle in the event is separated by a semicolon.
    """
    
    if not dataset_filepath.exists():
        return np.empty((0, 0, 5), dtype=np.float64)

    all_data = []
    lengths = []
    num_features = 9

    with open(dataset_filepath, 'r') as f:
        for line in f:
            if ';' not in line:
                continue

            line = line.replace(';', ' ').strip()
            floats = np.fromstring(line, sep=' ')
            if floats.size == 0:
                continue

            reshaped = floats.reshape(-1, num_features)
            all_data.append(reshaped)
            lengths.append(reshaped.shape[0])

    if not all_data:
        return np.empty((0, 0, num_features), dtype=np.float64)

    max_particles = max(lengths)
    n_events = len(all_data)

    # Preallocate final array and copy using slice assignment
    result = np.full((n_events, max_particles, num_features), pad_token, dtype=np.float64)
    for i, (ev, length) in enumerate(zip(all_data, lengths)):
        result[i, :length, :] = ev

    if flattened_events:
        return result.reshape(n_events, -1)
    return result

def convert_data_4vector_to_features(input_data, pad_token=np.nan):
    """
    Converts 4-vector data to featured data.
    
    Params
    input_data: np.ndarray
        Input data of shape (n_events, n_particles_per_event, n_features).
        n_features should be 5 (pdgid, e, px, py, pz).
    
    Returns
    np.ndarray
        Converted data of shape (n_events, n_particles_per_event, n_features).
        n_features will be 4 (pdgid, pt, eta, phi).
    """
    
    # output_data = np.full_like(input_data, pad_token)
    output_data = np.full((input_data.shape[0], input_data.shape[1], 4), pad_token)

    pdgid = input_data[:, :, 0]
    e = input_data[:, :, 1]
    px = input_data[:, :, 2]
    py = input_data[:, :, 3]
    pz = input_data[:, :, 4]

    pt = np.sqrt(px ** 2 + py ** 2)
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    theta = np.arccos(pz / p)
    eta = -np.log(np.tan(theta / 2))
    phi = np.arctan2(py, px)

    output_data[:, :, 0] = pdgid
    output_data[:, :, 1] = pt
    output_data[:, :, 2] = eta
    output_data[:, :, 3] = phi
    
    np.nan_to_num(output_data, copy=False, nan=pad_token)
    return output_data

def convert_data_features_to_4vector(input_data, pad_token=np.nan):
    """
    Converts featured data back to 4-vector format. Reconstruction uses mass from particle library.
    
    Params
    input_data: np.ndarray
        Input data of shape (n_events, n_particles_per_event, n_features).
        n_features should be 4 (pdgid, pt, eta, phi).
    
    Returns
    np.ndarray
        Converted data of shape (n_events, n_particles_per_event, n_features).
        n_features will be 5 (pdgid, e, px, py, pz).
    """
    
    output_data = np.full((input_data.shape[0], input_data.shape[1], 5), pad_token)

    pdgid = input_data[:, :, 0]
    pt = input_data[:, :, 1]
    eta = input_data[:, :, 2]
    phi = input_data[:, :, 3]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(phi)
    particle = Particle.from_pdgid(pdgid)
    e = np.sqrt(px * px + py * py + pz * pz + particle.mass * particle.mass)

    output_data[:, :, 0] = pdgid
    output_data[:, :, 1] = e
    output_data[:, :, 2] = px
    output_data[:, :, 3] = py
    output_data[:, :, 4] = pz
    
    np.nan_to_num(output_data, copy=False, nan=pad_token)
    return output_data
