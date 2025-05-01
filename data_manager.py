import numpy as np

def load_geant4_dataset(dataset_filepath, flattened_events=False, pad_token=np.nan):
    """
    Parses a MCGenerators created dataset and returns a np.ndarray of the
    data with shape (n_events, n_particles_per_event, n_features).
    
    Expects input data in the form "pdgid, e, px, py, pz;...;'
    
    Params
    dataset_filepath: str
        Path to the dataset file.
    flattened_events: bool
        If True, returns np.ndarray of shape (n_events, n_particles_per_event * n_features).
    pad_token:
        Value to pad the data with. Default is np.nan.
    """

    if not dataset_filepath.exists():
        return np.array([], dtype=np.float64)

    # Read the dataset file and accumulate data
    accumulated_data = []
    max_particles = 0
    with open(dataset_filepath, 'r') as in_file:
        for event in in_file:
            particles = event.strip().split(';')[:-1]
            event_arr = []
            for particle in particles:
                pdgid, e, px, py, pz = map(np.float64, particle.split())
                event_arr.append([pdgid, e, px, py, pz])
            accumulated_data.append(event_arr)
            max_particles = max(max_particles, len(event_arr))
    
    # Pad data
    num_features = 5
    padded_data = np.full((len(accumulated_data), max_particles, num_features), pad_token, dtype=np.float64)
    for i, event in enumerate(accumulated_data):
        for j, particle in enumerate(event):
            if j < max_particles:
                padded_data[i, j] = particle

    if flattened_events:
        return padded_data.reshape(len(accumulated_data), max_particles * num_features)
    
    return padded_data

def convert_data_4vector_to_angular(input_data, pad_token=np.nan):
    """
    Converts 4-vector data to angular data.
    
    Params
    input_data: np.ndarray
        Input data of shape (n_events, n_particles_per_event, n_features).
        n_features should be 5 (pdgid, e, px, py, pz).
    
    Returns
    np.ndarray
        Converted data of shape (n_events, n_particles_per_event, n_features).
        n_features will be 5 (pdgid, e, eta, theta, phi).
    """
    
    output_data = np.full_like(input_data, pad_token)

    pdgid = input_data[:, :, 0]
    e = input_data[:, :, 1]
    px = input_data[:, :, 2]
    py = input_data[:, :, 3]
    pz = input_data[:, :, 4]

    # pt = np.sqrt(px ** 2 + py ** 2)
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)

    theta = np.arccos(pz / p)
    eta = -np.log(np.tan(theta / 2))
    phi = np.arctan2(py, px)

    output_data[:, :, 0] = pdgid
    output_data[:, :, 1] = e
    output_data[:, :, 2] = eta
    output_data[:, :, 3] = theta
    output_data[:, :, 4] = phi
    
    np.nan_to_num(output_data, copy=False, nan=pad_token)
    return output_data

def convert_data_4vector_to_featured(input_data, pad_token=np.nan):
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

def convert_data_angular_to_4vector(input_data, pad_token=np.nan):
    """
    Converts angular data back to 4-vector format.
    
    Params
    input_data: np.ndarray
        Input data of shape (n_events, n_particles_per_event, n_features).
        n_features should be 5 (pdgid, e, eta, theta, phi).
    
    Returns
    np.ndarray
        Converted data of shape (n_events, n_particles_per_event, n_features).
        n_features will be 5 (pdgid, e, px, py, pz).
    """
    
    output_data = np.full_like(input_data, pad_token)

    pdgid = input_data[:, :, 0]
    e = input_data[:, :, 1]
    eta = input_data[:, :, 2]
    theta = input_data[:, :, 3]
    phi = input_data[:, :, 4]
    
    px = e * np.sin(theta) * np.cos(phi)
    py = e * np.sin(theta) * np.sin(phi)
    pz = e * np.cos(theta)

    output_data[:, :, 0] = pdgid
    output_data[:, :, 1] = e
    output_data[:, :, 2] = px
    output_data[:, :, 3] = py
    output_data[:, :, 4] = pz
    
    np.nan_to_num(output_data, copy=False, nan=pad_token)
    return output_data

def convert_data_featured_to_4vector(input_data, pad_token=np.nan):
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