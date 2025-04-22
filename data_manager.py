
import pandas as pd
import numpy as np
from numba import njit, int32, int64, float64

from dictionary import Dictionary

class SchemeStandard:
    def __init__(self, dictionary_filename, input_data_filename, output_data_filename):
        self.dictionary_filename = dictionary_filename
        self.input_data_filename = input_data_filename
        self.output_data_filename = output_data_filename
        self.dictionary = Dictionary(self.dictionary_filename)

    def tokenize_events_in_range(self):
        real_data = []
        with open(self.input_data_filename, 'r') as f:
            for event in f:
                event_arr = []
                particles = event.strip().split(';')
                for particle in particles:
                    particle = particle.split()
                    particle = int(particle[0]), *map(float, particle[1:])
                    event_arr.extend(particle)
                real_data.append(event_arr)
                
        # Load mapping
        particles_index = self.dictionary.dictionary_data['particles_index']
        particles_id = self.dictionary.dictionary_data['particles_id']
        pdgid_to_index = pd.Series({
            int(pid): particles_index[name] for pid, name in particles_id.items()
        })
        pdgid_to_index[0] = 0
        
        tokenized_events = []
        for event in real_data:
            event = np.array(event)
            particles = event.reshape(-1, 5)
            pdgids = particles[:, 0].astype(int)
            energy = particles[:, 1]
            px     = particles[:, 2]
            py     = particles[:, 3]
            pz     = particles[:, 4]

            # Kinematics
            r = np.sqrt(px**2 + py**2 + pz**2)
            theta = np.arccos(np.clip(pz / (r + 1e-8), -1, 1))
            phi   = np.arctan2(py, px)
            eta   = -np.log(np.tan(theta / 2))

            if np.any(np.abs(eta) > 4):
                continue

            # Vectorized ID mapping
            particle_index = pdgid_to_index.reindex(pdgids).fillna(0).astype(int).to_numpy()

            # Vectorized digitization
            e_bin     = np.digitize(energy, self.dictionary.e_bins).astype(int)
            eta_bin   = np.digitize(eta, self.dictionary.eta_bins).astype(int)
            theta_bin = np.digitize(theta, self.dictionary.theta_bins).astype(int)
            phi_bin   = np.digitize(phi, self.dictionary.phi_bins).astype(int)

            # Token construction (vectorized)
            tokens = np.stack([
                np.full(len(pdgids), self.dictionary.dictionary_data['special_tokens']['particle_start'], dtype=int),
                particle_index + self.dictionary.PDGID_OFFSET,
                e_bin + self.dictionary.ENERGY_OFFSET,
                eta_bin + self.dictionary.ETA_OFFSET,
                theta_bin + self.dictionary.THETA_OFFSET,
                phi_bin + self.dictionary.PHI_OFFSET,
                np.full(len(pdgids), self.dictionary.dictionary_data['special_tokens']['particle_end'], dtype=int),
            ], axis=1)
            
            tokens = tokens.flatten()
            tokens = np.concatenate([np.array([1]), tokens, np.array([2])], dtype=int)
            tokenized_events.append(tokens)
        
        return tokenized_events
            
    def tokenize_data(self):
        tokenized_data = self.tokenize_events_in_range()
        
        pad_block = np.array([3, 0, 0, 0, 0, 0, 4], dtype=np.int32)

        max_sequence_length = max(len(event) for event in tokenized_data)
        max_num_particles = (max_sequence_length - 2) // 7

        padded_data = []
        for event in tokenized_data:
            num_particles = (len(event) - 2) // 7
            pad_count = max_num_particles - num_particles
            padded_event = np.concatenate([event, np.tile(pad_block, pad_count)])
            padded_data.append(padded_event)
        
        with open(self.output_data_filename, 'w') as f:
            for row in padded_data:
                f.write(' '.join(map(str, row)))
                f.write('\n')

@njit(int32[:](float64[:], float64[:]), fastmath=True, inline='always', cache=True)
def digitize_optimized(values, bins):
    result = np.empty(len(values), dtype=int32)
    n_bins = len(bins)
    last_bin = n_bins - 1
    last_val = bins[-1]
    
    for j in range(len(values)):
        value = values[j]
        if value >= last_val:
            result[j] = last_bin
            continue
            
        # Binary search implementation
        low = 0
        high = n_bins
        while low < high:
            mid = (low + high) // 2
            if bins[mid] <= value:
                low = mid + 1
            else:
                high = mid
        result[j] = low - 1
        
    return result

@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def tokenize_events_in_range_compiled(real_data, e_bins, eta_bins, theta_bins, phi_bins,
                                    pdgid_offset, energy_offset, eta_offset,
                                    theta_offset, phi_offset, pdgid_map_keys, pdgid_map_vals):
    tokenized_events = []
    num_events = real_data.shape[0]
    pdgid_map_len = len(pdgid_map_keys)
    
    # Precompute constants
    one_7th = 1/7
    log_half = np.log(0.5)
    
    # Create PDGID lookup table if possible
    max_pdg = np.max(pdgid_map_keys) if pdgid_map_len > 0 else 0
    min_pdg = np.min(pdgid_map_keys) if pdgid_map_len > 0 else 0
    pdgid_range = max_pdg - min_pdg + 1
    use_lookup = pdgid_range <= 10000  # Only use if range is reasonable
    
    if use_lookup and pdgid_map_len > 0:
        pdgid_lookup = np.full(pdgid_range, -1, dtype=int64)
        for j in range(pdgid_map_len):
            pdgid_lookup[pdgid_map_keys[j] - min_pdg] = pdgid_map_vals[j]
    else:
        use_lookup = False
    
    for event_idx in range(num_events):
        event = real_data[event_idx]
        num_values = len(event)
        if num_values % 5 != 0:
            continue

        n_particles = num_values // 5
        particles = event.reshape(n_particles, 5)

        # Direct column access with explicit types
        pdgids = particles[:, 0].astype(int64)
        energy = particles[:, 1]
        px = particles[:, 2]
        py = particles[:, 3]
        pz = particles[:, 4]

        # Optimized spherical coordinate calculations
        px_sq = px * px
        py_sq = py * py
        pz_sq = pz * pz
        r = np.sqrt(px_sq + py_sq + pz_sq)
        r_inv = 1.0 / (r + 1e-8)
        
        # Theta calculation with manual clipping
        theta = np.empty(n_particles, dtype=float64)
        for i in range(n_particles):
            val = pz[i] * r_inv[i]
            if val > 1.0:
                val = 1.0
            elif val < -1.0:
                val = -1.0
            theta[i] = np.arccos(val)
        
        # Phi and eta calculations
        phi = np.arctan2(py, px)
        tan_theta_half = np.tan(0.5 * theta)
        eta = -np.log(tan_theta_half)

        # Early exit for bad eta values
        bad_event = False
        for i in range(n_particles):
            if np.abs(eta[i]) > 4.0:
                bad_event = True
                break
        if bad_event:
            continue

        # PDGID mapping
        particle_index = np.zeros(n_particles, dtype=int64)
        if use_lookup:
            for i in range(n_particles):
                lookup_idx = pdgids[i] - min_pdg
                if 0 <= lookup_idx < pdgid_range:
                    mapped_val = pdgid_lookup[lookup_idx]
                    if mapped_val != -1:
                        particle_index[i] = mapped_val
        else:
            for i in range(n_particles):
                pdgid = pdgids[i]
                for j in range(pdgid_map_len):
                    if pdgid == pdgid_map_keys[j]:
                        particle_index[i] = pdgid_map_vals[j]
                        break

        # Digitization with optimized binary search
        e_bin = digitize_optimized(energy, e_bins)
        eta_bin = digitize_optimized(eta, eta_bins)
        theta_bin = digitize_optimized(theta, theta_bins)
        phi_bin = digitize_optimized(phi, phi_bins)

        # Token construction with precalculation
        token_len = n_particles * 7 + 2
        tokens = np.empty(token_len, dtype=int64)
        tokens[0] = 1
        tokens[-1] = 2

        for i in range(n_particles):
            base = 1 + i * 7
            tokens[base] = 3
            tokens[base + 1] = particle_index[i] + pdgid_offset
            tokens[base + 2] = e_bin[i] + energy_offset
            tokens[base + 3] = eta_bin[i] + eta_offset
            tokens[base + 4] = theta_bin[i] + theta_offset
            tokens[base + 5] = phi_bin[i] + phi_offset
            tokens[base + 6] = 4

        tokenized_events.append(tokens)

    return tokenized_events
    
class SchemeStandardCompiled:
    def __init__(self, dictionary_filename, input_data_filename, output_data_filename):
        self.dictionary_filename = dictionary_filename
        self.input_data_filename = input_data_filename
        self.output_data_filename = output_data_filename
        self.dictionary = Dictionary(self.dictionary_filename)
    
    def load_and_pad_data(self, filename):
        # Read the file line by line
        with open(filename, 'r') as f:
            lines = f.readlines()

        parsed_events = []
        max_len = 0

        for line in lines:
            line = line.replace(';', ' ')
            parts = np.fromstring(line, sep=' ', dtype=np.float64)
            parsed_events.append(parts)
            if len(parts) > max_len:
                max_len = len(parts)

        # Pad all events to max length
        padded_array = np.zeros((len(parsed_events), max_len), dtype=np.float64)
        for i, event in enumerate(parsed_events):
            padded_array[i, :len(event)] = event

        return padded_array

    def tokenize_data(self):
        # Read and parse raw data
        real_data = self.load_and_pad_data(self.input_data_filename)

        # Build mapping from PDGID to index
        particles_index = self.dictionary.dictionary_data['particles_index']
        particles_id = self.dictionary.dictionary_data['particles_id']
        pdgid_to_index = {int(pid): particles_index[name] for pid, name in particles_id.items()}
        pdgid_to_index[0] = 0

        pdgid_map_keys = np.array(list(pdgid_to_index.keys()), dtype=np.int64)
        pdgid_map_vals = np.array(list(pdgid_to_index.values()), dtype=np.int64)
        
        # Warmup because numba jit compilation is lazy
        # _ = tokenize_events_in_range_compiled(
        #     real_data,
        #     np.array(self.dictionary.e_bins, dtype=np.float64),
        #     np.array(self.dictionary.eta_bins, dtype=np.float64),
        #     np.array(self.dictionary.theta_bins, dtype=np.float64),
        #     np.array(self.dictionary.phi_bins, dtype=np.float64),
        #     self.dictionary.PDGID_OFFSET,
        #     self.dictionary.ENERGY_OFFSET,
        #     self.dictionary.ETA_OFFSET,
        #     self.dictionary.THETA_OFFSET,
        #     self.dictionary.PHI_OFFSET,
        #     pdgid_map_keys,
        #     pdgid_map_vals
        # )

        # Tokenize using compiled function
        tokenized_data = tokenize_events_in_range_compiled(
            real_data,
            np.array(self.dictionary.e_bins, dtype=np.float64),
            np.array(self.dictionary.eta_bins, dtype=np.float64),
            np.array(self.dictionary.theta_bins, dtype=np.float64),
            np.array(self.dictionary.phi_bins, dtype=np.float64),
            self.dictionary.PDGID_OFFSET,
            self.dictionary.ENERGY_OFFSET,
            self.dictionary.ETA_OFFSET,
            self.dictionary.THETA_OFFSET,
            self.dictionary.PHI_OFFSET,
            pdgid_map_keys,
            pdgid_map_vals
        )

        # Pad tokenized data
        pad_block = np.array([3, 0, 0, 0, 0, 0, 4], dtype=np.int64)
        max_sequence_length = max(len(event) for event in tokenized_data)
        max_num_particles = (max_sequence_length - 2) // 7

        padded_data = []
        for event in tokenized_data:
            num_particles = (len(event) - 2) // 7
            pad_count = max_num_particles - num_particles
            padded_event = np.concatenate([event, np.tile(pad_block, pad_count)])
            padded_data.append(padded_event)

        # Save to file
        with open(self.output_data_filename, 'w') as f:
            for row in padded_data:
                f.write(' '.join(map(str, row)))
                f.write('\n')

class SchemePaddingV2:
    def __init__(self, dictionary_filename, input_data_filename, output_data_filename):
        self.dictionary_filename = dictionary_filename
        self.input_data_filename = input_data_filename
        self.output_data_filename = output_data_filename
        self.dictionary = Dictionary(self.dictionary_filename)

    def tokenize_events_in_range(self):
        real_data = []
        with open(self.input_data_filename, 'r') as f:
            for event in f:
                event_arr = []
                particles = event.strip().split(';')
                for particle in particles:
                    particle = particle.split()
                    particle = int(particle[0]), *map(float, particle[1:])
                    event_arr.extend(particle)
                real_data.append(event_arr)
                
        # Load mapping
        particles_index = self.dictionary.dictionary_data['particles_index']
        particles_id = self.dictionary.dictionary_data['particles_id']
        pdgid_to_index = pd.Series({
            int(pid): particles_index[name] for pid, name in particles_id.items()
        })
        pdgid_to_index[0] = 0
        
        tokenized_events = []
        for event in real_data:
            event = np.array(event)
            particles = event.reshape(-1, 5)
            pdgids = particles[:, 0].astype(int)
            energy = particles[:, 1]
            px     = particles[:, 2]
            py     = particles[:, 3]
            pz     = particles[:, 4]

            # Kinematics
            r = np.sqrt(px**2 + py**2 + pz**2)
            theta = np.arccos(np.clip(pz / (r + 1e-8), -1, 1))
            phi   = np.arctan2(py, px)
            eta   = -np.log(np.tan(theta / 2))

            if np.any(np.abs(eta) > 4):
                continue

            # Vectorized ID mapping
            particle_index = pdgid_to_index.reindex(pdgids).fillna(0).astype(int).to_numpy()

            # Vectorized digitization
            e_bin     = np.digitize(energy, self.dictionary.e_bins).astype(int)
            eta_bin   = np.digitize(eta, self.dictionary.eta_bins).astype(int)
            theta_bin = np.digitize(theta, self.dictionary.theta_bins).astype(int)
            phi_bin   = np.digitize(phi, self.dictionary.phi_bins).astype(int)

            # Token construction (vectorized)
            tokens = np.stack([
                np.full(len(pdgids), self.dictionary.dictionary_data['special_tokens']['particle_start'], dtype=int),
                particle_index + self.dictionary.PDGID_OFFSET,
                e_bin + self.dictionary.ENERGY_OFFSET,
                eta_bin + self.dictionary.ETA_OFFSET,
                theta_bin + self.dictionary.THETA_OFFSET,
                phi_bin + self.dictionary.PHI_OFFSET,
                np.full(len(pdgids), self.dictionary.dictionary_data['special_tokens']['particle_end'], dtype=int),
            ], axis=1)
            
            tokens = tokens.flatten()
            tokens = np.concatenate([np.array([1]), tokens, np.array([2])], dtype=int)
            tokenized_events.append(tokens)
        
        return tokenized_events
            
    def tokenize_data(self):
        tokenized_data = self.tokenize_events_in_range()
        
        # Pad to max sequence length
        max_sequence_length = max(len(event) for event in tokenized_data)
        padded_data = []
        for event in tokenized_data:
            padded = np.pad(event, (0, max_sequence_length - len(event)), constant_values=0)
            padded_data.append(padded)
            
        print(padded_data[0])
        np.savetxt(self.output_data_filename, padded_data, delimiter=' ')