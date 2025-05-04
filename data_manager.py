import numpy as np
from concurrent.futures import ThreadPoolExecutor

def load_geant4_dataset(dataset_filepath, flattened_events=False, pad_token=np.nan):
    """
    Ultra-fast loader for Geant4 MC dataset files.
    """
    
    if not dataset_filepath.exists():
        return np.empty((0, 0, 5), dtype=np.float64)

    all_data = []
    lengths = []
    num_features = 5

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
        n_features will be 5 (pdgid, e, pt, eta, phi).
    """
    
    output_data = np.full_like(input_data, pad_token)

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
    output_data[:, :, 1] = e
    output_data[:, :, 2] = pt
    output_data[:, :, 3] = eta
    output_data[:, :, 4] = phi
    
    np.nan_to_num(output_data, copy=False, nan=pad_token)
    return output_data

def convert_data_features_to_4vector(input_data, pad_token=np.nan):
    """
    Converts featured data back to 4-vector format.
    
    Params
    input_data: np.ndarray
        Input data of shape (n_events, n_particles_per_event, n_features).
        n_features should be 5 (pdgid, e, pt, eta, phi).
    
    Returns
    np.ndarray
        Converted data of shape (n_events, n_particles_per_event, n_features).
        n_features will be 5 (pdgid, e, px, py, pz).
    """
    
    output_data = np.full_like(input_data, pad_token)

    pdgid = input_data[:, :, 0]
    e = input_data[:, :, 1]
    pt = input_data[:, :, 2]
    eta = input_data[:, :, 3]
    phi = input_data[:, :, 4]
    
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(phi)

    output_data[:, :, 0] = pdgid
    output_data[:, :, 1] = e
    output_data[:, :, 2] = px
    output_data[:, :, 3] = py
    output_data[:, :, 4] = pz
    
    np.nan_to_num(output_data, copy=False, nan=pad_token)
    return output_data