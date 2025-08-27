# Utility for retrieving bins, tokens, vocabulary, and humanizing the dictionary

import json
import numpy as np
from pathlib import Path
from particle import Particle
from enum import Enum

from scipy.stats import norm
from scipy.interpolate import interp1d

import data_manager as dm

script_dir = Path(__file__).resolve().parent

class ETokenTypes(Enum):
    PADDING = 0
    SPECIAL = 1
    PDGID = 2
    MATERIAL = 3
    ENERGY = 4
    ETA = 5
    THETA = 6
    PHI = 7
    PT = 8
    PX = 9
    PY = 10
    PZ = 11

# Custom arange because np.arange and np.linspace suffer from floating point precision issues.
def arange(start, stop, step_size):
    if step_size <= 0:
        return np.array([], dtype=np.float64)
    
    decimal_places = int(np.ceil(np.log10(1 / step_size)))
    result = []
    i = start
    while i < stop:
        result.append(round(i, decimal_places))
        i += step_size
    return np.array(result, dtype=np.float64)

def linear_space(start, stop, n_bins):
    if n_bins == 0:
        return np.array([], dtype=np.float64)
    
    step_size = (stop - start) / n_bins
    decimal_places = int(np.ceil(np.log10(1 / step_size)))

    result = []
    i = start
    while i < stop:
        result.append(round(i, decimal_places))
        i += step_size
    return np.array(result, dtype=np.float64)

# Gaussian tokenization
def gaussian_space(start, stop, num, center, sigma=1.0):
    """
    More advanced Gaussian spacing with peak control
    
    Generate a Gaussian-spaced array of points between start and stop,
    with the highest density of points near `center`.

    Parameters:
        start (float): Start of range.
        stop (float): End of range.
        num (int): Number of points to generate.
        center (float): Location where spacing is most dense.
        sigma (float): Controls sharpness of peak (standard deviation of Gaussian).

    Returns:
        np.ndarray: Array of values spaced according to a Gaussian profile.
    """
    # Create a fine-grained range to define the target density
    x_fine = np.linspace(start, stop, 10_000)
    pdf = norm.pdf(x_fine, loc=center, scale=sigma)
    cdf = np.cumsum(pdf)
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())  # normalize to [0, 1]

    # Invert CDF: get target values from uniform
    inv_cdf = interp1d(cdf, x_fine, bounds_error=False, fill_value=(start, stop))
    uniform_probs = np.linspace(0, 1, num)
    return inv_cdf(uniform_probs)

# Detokenization of gaussian bins using bin medians
def detokenize_gaussian_bins(tokens, bin_edges):
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
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Clip tokens to valid range
    tokens = np.clip(tokens, 0, len(bin_centers) - 1)
    return bin_centers[tokens]

# Helper for detokenization of gaussian bins using bin means
def build_gaussian_bin_means(values, bin_edges):
    bin_indices = np.digitize(values, bin_edges[1:-1], right=True)  # tokens range from 0 to len(bin_edges) - 2
    num_bins = len(bin_edges) - 1  # Number of actual bins
    bin_means = np.zeros(num_bins)
    for i in range(num_bins):
        in_bin = values[bin_indices == i]
        if len(in_bin) > 0:
            bin_means[i] = in_bin.mean()
        else:
            bin_means[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])
    return bin_means

# Quantile binning
def quantile_detokenize(tokens, bin_edges, token_min=0):
    indices = tokens - token_min
    # Clip just in case
    indices = np.clip(indices, 0, len(bin_edges) - 2)
    # Use bin centers (can replace with per-bin means for more precision)
    return 0.5 * (bin_edges[indices] + bin_edges[indices + 1])

def build_quantile_bin_means(values, bin_edges):
    bin_indices = np.digitize(values, bin_edges[1:-1], right=True)
    bin_means = np.zeros(len(bin_edges) - 1)
    for i in range(len(bin_means)):
        in_bin = values[bin_indices == i]
        bin_means[i] = in_bin.mean() if len(in_bin) > 0 else 0.5 * (bin_edges[i] + bin_edges[i + 1])
    return bin_means

def truncate_quantiles(quantile_tokens, original_bin_size, truncated_bin_size):
    quan_bin_bins = np.linspace(0, original_bin_size, truncated_bin_size)
    quan_bin_bin_tokenized = np.digitize(quantile_tokens, quan_bin_bins)
    return quan_bin_bin_tokenized

# Creates a list of all values of a given type from the dataset (i.e. energy, eta, etc.).
# This is used to create quantile bins.
def get_all_of(dictionary, input_data_filepath, type_str):
    type_idx = dictionary.tokenization_schema.index(type_str)
    in_dataset = dm.load_geant4_dataset(input_data_filepath, flattened_events=False, pad_token=np.nan)
    all_of_type = []
    for event in in_dataset:
        for particle in event:
            if not np.isnan(particle[type_idx]):
                all_of_type.append(particle[type_idx])
    return all_of_type

class Dictionary():
    def __init__(self, dictionary_filename):
        self.dictionary_filename = dictionary_filename
        
        with open(dictionary_filename, 'r') as f:
            self.dictionary_data = json.load(f)
        
        self.dataset_filepath   = script_dir / 'data' / self.dictionary_data['dataset']
        assert self.dataset_filepath.exists(), f"Dataset file {self.dataset_filepath} does not exist."
        self.preparation_name   = self.dictionary_data.get('preparation_name', None)
        assert self.preparation_name is not None, "preparation_name must be defined in the dictionary."

        self.num_special_tokens = len(self.dictionary_data['special_tokens'])
        self.num_particles      = len(self.dictionary_data['pdgids'])
        self.num_materials      = len(self.dictionary_data['materials_named'])
        
        self.tokenization_schema = [''] * len(self.dictionary_data['tokenization'])
        for pos_str, tokenization_type_str in self.dictionary_data['tokenization'].items():
            self.tokenization_schema[int(pos_str)] = tokenization_type_str
        
        self.padding_sequence = [''] * len(self.dictionary_data['padding'])
        for pos_str, padding_str in self.dictionary_data['padding'].items():
            self.padding_sequence[int(pos_str)] = padding_str
        
        self.num_tokens_per_particle = len(self.tokenization_schema)
        
        self.e_bins     = self._create_bins('e')
        self.eta_bins   = self._create_bins('eta')
        self.theta_bins = self._create_bins('theta')
        self.phi_bins   = self._create_bins('phi')
        self.pt_bins    = self._create_bins('pt')
        self.px_bins    = self._create_bins('px')
        self.py_bins    = self._create_bins('py')
        self.pz_bins    = self._create_bins('pz')

        self.vocab_size = self.num_special_tokens + self.num_particles + self.num_materials + len(self.e_bins) + len(self.eta_bins) + len(self.theta_bins) + len(self.phi_bins) + len(self.pt_bins) + len(self.px_bins) + len(self.py_bins) + len(self.pz_bins)
        
        # Offsets for tokenization (since we need to eliminate repeat tokens)
        self.SPECIAL_TOKENS_OFFSET = 0
        self.PDGID_OFFSET = self.SPECIAL_TOKENS_OFFSET + self.num_special_tokens
        self.MATERIAL_OFFSET = self.PDGID_OFFSET + self.num_particles
        self.ENERGY_OFFSET = self.MATERIAL_OFFSET + self.num_materials
        self.ETA_OFFSET = self.ENERGY_OFFSET + len(self.e_bins)
        self.THETA_OFFSET = self.ETA_OFFSET + len(self.eta_bins)
        self.PHI_OFFSET = self.THETA_OFFSET + len(self.theta_bins)
        self.PT_OFFSET = self.PHI_OFFSET + len(self.phi_bins)
        self.PX_OFFSET = self.PT_OFFSET + len(self.pt_bins)
        self.PY_OFFSET = self.PX_OFFSET + len(self.px_bins)
        self.PZ_OFFSET = self.PY_OFFSET + len(self.py_bins)

        # Converts input particle ID to the relevant index
        self.pdgids = self.dictionary_data['pdgids']
        
        self.table_data = [
            ["Type",            "Num",                    "Token Range",                                                              "Min",                    "Max",                    "Step Size"],
            ["Special tokens",  self.num_special_tokens,  self.token_range_str(self.SPECIAL_TOKENS_OFFSET, self.num_special_tokens),  "N/A",                    "N/A",                    "N/A"],
            ["Particles",       self.num_particles,       self.token_range_str(self.PDGID_OFFSET, self.num_particles),                "N/A",                    "N/A",                    "N/A"],
            ["Materials",       self.num_materials,       self.token_range_str(self.MATERIAL_OFFSET, self.num_materials),             "N/A",                    "N/A",                    "N/A"],
            ["Energy bins",     len(self.e_bins),         self.token_range_str(self.ENERGY_OFFSET, len(self.e_bins)),                 self.token_min('e'),      self.token_max('e'),      self.token_step_size('e')],
            ["Eta bins",        len(self.eta_bins),       self.token_range_str(self.ETA_OFFSET, len(self.eta_bins)),                  self.token_min('eta'),    self.token_max('eta'),    self.token_step_size('eta')],
            ["Theta bins",      len(self.theta_bins),     self.token_range_str(self.THETA_OFFSET, len(self.theta_bins)),              self.token_min('theta'),  self.token_max('theta'),  self.token_step_size('theta')],
            ["Phi bins",        len(self.phi_bins),       self.token_range_str(self.PHI_OFFSET, len(self.phi_bins)),                  self.token_min('phi'),    self.token_max('phi'),    self.token_step_size('phi')],
            ["Pt bins",         len(self.pt_bins),        self.token_range_str(self.PT_OFFSET, len(self.pt_bins)),                    self.token_min('pt'),     self.token_max('pt'),     self.token_step_size('pt')],
            ["Px bins",         len(self.px_bins),        self.token_range_str(self.PX_OFFSET, len(self.px_bins)),                    self.token_min('px'),     self.token_max('px'),     self.token_step_size('px')],
            ["Py bins",         len(self.py_bins),        self.token_range_str(self.PY_OFFSET, len(self.py_bins)),                    self.token_min('py'),     self.token_max('py'),     self.token_step_size('py')],
            ["Pz bins",         len(self.pz_bins),        self.token_range_str(self.PZ_OFFSET, len(self.pz_bins)),                    self.token_min('pz'),     self.token_max('pz'),     self.token_step_size('pz')]
        ]
    
    def _create_bins(self, type_str):
        token_bin_key_name = f'{type_str}_bin_data'
        if token_bin_key_name not in self.dictionary_data:
            return np.array([], dtype=np.float64)
        
        # Figure out the spacing function.
        tokenization_function = 'linear'
        if 'tokenization' in self.dictionary_data[token_bin_key_name]:
            tokenization_function = self.dictionary_data[token_bin_key_name]['tokenization']
        
        if tokenization_function == 'linear':
            # For linear determine if we need to use arange or linspace.
            linear_spacing_function = None
            if 'step_size' in self.dictionary_data[token_bin_key_name]:
                linear_spacing_function = arange
                spacing_val = self.dictionary_data[token_bin_key_name].get('step_size', None)
            elif 'n_bins' in self.dictionary_data[token_bin_key_name]:
                linear_spacing_function = linear_space
                spacing_val = self.dictionary_data[token_bin_key_name].get('n_bins', None)
            assert linear_spacing_function is not None, f"Invalid tokenization function for {type_str}."
            assert spacing_val is not None and spacing_val > 0, f"Missing or invalid spacing value for {type_str}."
            
            return linear_spacing_function(self.token_min(type_str), self.token_max(type_str), spacing_val)
        elif tokenization_function == 'gaussian':
            gaussian_center = self.dictionary_data[token_bin_key_name].get('gaussian_center', None)
            gaussian_sigma = self.dictionary_data[token_bin_key_name].get('gaussian_sigma', 1.0)
            n_gaussian_bins = self.dictionary_data[token_bin_key_name].get('n_bins', None)
            assert gaussian_center is not None, f"Missing Gaussian center for {type_str}: {tokenization_function}"
            assert gaussian_sigma is not None and gaussian_sigma > 0, f"Missing or invalid Gaussian sigma for {type_str}: {gaussian_sigma}"
            assert n_gaussian_bins is not None and n_gaussian_bins > 0, f"Missing or invalid number of Gaussian bins for {type_str}: {n_gaussian_bins}"
            
            return gaussian_space(self.token_min(type_str), self.token_max(type_str), n_gaussian_bins, gaussian_center, gaussian_sigma)
        elif tokenization_function == 'quantile':
            quantile_edges_filepath = self.dictionary_filename.parent / f'{type_str}_quantile_edges.npy'
            
            # Retrieve from cache if exists
            if quantile_edges_filepath.exists():
                quantile_edges = np.load(quantile_edges_filepath)
                return quantile_edges
            
            # Quantile bins are created in the tokenizer for speed reasons as they require reading all the data.
            n_quantile_bins = self.dictionary_data[token_bin_key_name].get('n_quantile_bins', None)
            assert n_quantile_bins is not None and n_quantile_bins > 0, f"Missing or invalid number of quantile bins for {type_str}: {n_quantile_bins}"

            # To create quantile bins, we need a list of every possible value in the range.
            all_values = get_all_of(self, self.dataset_filepath, type_str)
            quantile_edges = np.quantile(all_values, q=np.linspace(0, 1, n_quantile_bins + 1))
            quantile_edges = quantile_edges[1:-1] # Remove first and last edges to get actual bins
            
            # Since this takes a while, we only want to do this once.
            np.save(quantile_edges_filepath, quantile_edges)
            return quantile_edges

        return np.array([], dtype=np.float64)
            
    # Functions to make the table look nicer
    def token_min(self, type_str):
        if f'{type_str}_bin_data' not in self.dictionary_data:
            return 0
        return self.dictionary_data[f'{type_str}_bin_data']['min']
    def token_max(self, type_str):
        if f'{type_str}_bin_data' not in self.dictionary_data:
            return 0
        return self.dictionary_data[f'{type_str}_bin_data']['max']
    def token_step_size(self, type_str):
        if f'{type_str}_bin_data' not in self.dictionary_data:
            return 0
        return self.dictionary_data[f'{type_str}_bin_data'].get('step_size', None)
    def token_n_bins(self, type_str):
        if f'{type_str}_bin_data' not in self.dictionary_data:
            return 0
        return self.dictionary_data[f'{type_str}_bin_data']['n_bins']
    def token_range(self, type_str):
        if f'{type_str}_bin_data' not in self.dictionary_data:
            return 0
        return self.token_max(type_str) - self.token_min(type_str)
    def token_range_str(self, offset, num_tokens):
        return f'{offset} - {(offset + num_tokens - 1)}'
    
    # Actual class functions
    @property
    def padding_token(self):
        return self.dictionary_data['special_tokens']['padding']
    @property
    def event_start_token(self):
        return self.dictionary_data['special_tokens']['event_start']
    @property
    def event_end_token(self):
        return self.dictionary_data['special_tokens']['event_end']
    @property
    def particle_start_token(self):
        return self.dictionary_data['special_tokens']['particle_start']
    @property
    def particle_end_token(self):
        return self.dictionary_data['special_tokens']['particle_end']
    @property
    def scheme(self):
        return self.dictionary_data.get('scheme', '')
    
    @property
    def pdgids_to_index(self):
        # Return reversed pdgids. This maps PDGIDs to their index in the dictionary.
        return {v: k for k, v in self.pdgids.items() if v != 0}
    
    def get_padding_sequence(self):
        padding_sequence = []
        for padding_str in self.padding_sequence:
            if padding_str == 'padding':
                padding_sequence.append(self.padding_token)
            elif padding_str == 'particle_start':
                padding_sequence.append(self.particle_start_token)
            elif padding_str == 'particle_end':
                padding_sequence.append(self.particle_end_token)
        return padding_sequence
    
    # Returns token type given the current token value (uses token range for evaluation)
    def get_token_type(self, token):
        if token == self.padding_token:
            return ETokenTypes.PADDING
        elif token in self.dictionary_data['special_tokens'].values():
            return ETokenTypes.SPECIAL
        elif token >= self.PDGID_OFFSET and token < self.PDGID_OFFSET + self.num_particles:
            return ETokenTypes.PDGID
        elif token >= self.MATERIAL_OFFSET and token < self.MATERIAL_OFFSET + self.num_materials:
            return ETokenTypes.MATERIAL
        elif token >= self.ENERGY_OFFSET and token < self.ENERGY_OFFSET + len(self.e_bins):
            return ETokenTypes.ENERGY
        elif token >= self.ETA_OFFSET and token < self.ETA_OFFSET + len(self.eta_bins):
            return ETokenTypes.ETA
        elif token >= self.THETA_OFFSET and token < self.THETA_OFFSET + len(self.theta_bins):
            return ETokenTypes.THETA
        elif token >= self.PHI_OFFSET and token < self.PHI_OFFSET + len(self.phi_bins):
            return ETokenTypes.PHI
        elif token >= self.PT_OFFSET and token < self.PT_OFFSET + len(self.pt_bins):
            return ETokenTypes.PT
        elif token >= self.PX_OFFSET and token < self.PX_OFFSET + len(self.px_bins):
            return ETokenTypes.PX
        elif token >= self.PY_OFFSET and token < self.PY_OFFSET + len(self.py_bins):
            return ETokenTypes.PY
        elif token >= self.PZ_OFFSET and token < self.PZ_OFFSET + len(self.pz_bins):
            return ETokenTypes.PZ

    def output_humanized_dictionary(self, output_file_path):
        # Define column widths for formatting
        col_widths = [max(len(str(row[col])) for row in self.table_data) + 3 for col in range(len(self.table_data[0]))]

        # Write the table to a text file
        with open(output_file_path, "w") as output_file:
            for row in self.table_data:
                formatted_row = "".join(str(cell).ljust(col_widths[idx]) + "|  " for idx, cell in enumerate(row))
                output_file.write(formatted_row + "\n")
                
            output_file.write("\n")

            # Print particles and their corresponding token values
            output_file.write("Special Token - Token Value\n")
            for special_token, token_value in self.dictionary_data['special_tokens'].items():
                output_file.write(f"{special_token} - {token_value + self.SPECIAL_TOKENS_OFFSET}\n")
                
            output_file.write("\n")
                
            # Print particles and their corresponding token values
            output_file.write("Token - PDGID - Particle Name\n")
            for idx, pdgid in self.dictionary_data['pdgids'].items():
                if pdgid == 0:
                    output_file.write(f"{int(idx) + self.PDGID_OFFSET} - {pdgid} - none\n")
                    continue
                p = Particle.from_pdgid(pdgid)
                output_file.write(f"{int(idx) + self.PDGID_OFFSET} - {pdgid} - {p.name}\n")

            # Print materials and their corresponding token values
            output_file.write("\nMaterial Name - Token Value\n")
            for material_name, token_value in self.dictionary_data['materials_named'].items():
                output_file.write(f"{material_name} - {token_value + self.MATERIAL_OFFSET}\n")

    # Updates the particles list in dictionary.json based on the input data
    def update_dictionary_particle_list(self, input_data_filename, dictionary_filename):
        # Store all unique particle PDGIDs from input_data_filename
        with open(input_data_filename, 'r') as f:
            particle_ids = set()
            for event in f:
                for particle in event.split(';'):
                    pid = particle.split()[0]
                    particle_ids.add(int(pid))

        # Create pdgids dictionary
        read_pdgids = {}
        for idx, pid, in enumerate(particle_ids):
            read_pdgids[idx] = pid
            
        if len(read_pdgids) > 75:
            raise ValueError("More than 75 unique PDGIDs found. This is currently unsupported.")
        
        # Make sure we have 75 slots by marking unused with 0 since that is not a valid pdgid
        for i in range(75):
            if i not in read_pdgids:
                read_pdgids[i] = 0

        # Update dictionary.json
        with open(dictionary_filename, 'r') as f:
            dictionary = json.load(f)

        with open(dictionary_filename, 'w') as f:
            dictionary["pdgids"] = read_pdgids
            json.dump(dictionary, f, indent=2)