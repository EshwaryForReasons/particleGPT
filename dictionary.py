# Utility for retrieving bins, tokens, vocabulary, and humanizing the dictionary

import json
import numpy as np
from pathlib import Path
from particle import Particle
from enum import Enum

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

# Custom arange because np.arange and np.linspace suffer from floating point precision issues.
# This is needed so the C++ version matches.
def custom_arange(start, stop, step_size):
    if step_size <= 0:
        return []
    
    decimal_places = int(np.ceil(np.log10(1 / step_size)))
    result = []
    i = start
    while i < stop:
        result.append(round(i, decimal_places))
        i += step_size
    return result

def custom_linspace(start, stop, n_bins):
    if n_bins == 0:
        return []
    
    step_size = (stop - start) / n_bins
    decimal_places = int(np.ceil(np.log10(1 / step_size)))

    result = []
    i = start
    while i < stop:
        result.append(round(i, decimal_places))
        i += step_size
    return result

class Dictionary():
    def __init__(self, dictionary_filename):
        self.dictionary_filename = dictionary_filename
        
        with open(dictionary_filename, 'r') as f:
            self.dictionary_data = json.load(f)
            
        self.num_special_tokens = len(self.dictionary_data['special_tokens'])
        self.num_particles      = len(self.dictionary_data['pdgids'])
        self.num_materials      = len(self.dictionary_data['materials_named'])
        
        def create_bins(type_str):
            def ret_same(x):
                return x
            def ret_log(x):
                return np.log(max(1, x))
            
            token_bin_jey_name = f'{type_str}_bin_data'
            if token_bin_jey_name not in self.dictionary_data:
                return []
            
            # See if we should use arange or linspace. step_size implies arange and n_bins implies linspace.
            b_use_linspace = 'n_bins' in self.dictionary_data[token_bin_jey_name]
            bin_generation_func = custom_linspace if b_use_linspace else custom_arange
            spacing_func = self.token_n_bins if b_use_linspace else self.token_step_size
            
            # Determine the transform function.
            transform_type = 'linear'
            if 'transform' in self.dictionary_data[token_bin_jey_name]:
                transform_type = self.dictionary_data[token_bin_jey_name]['transform']
            transform_func = ret_log if transform_type == 'log' else ret_same
            
            local_bins = bin_generation_func(transform_func(self.token_min(type_str)), transform_func(self.token_max(type_str)), spacing_func(type_str))
            return local_bins
        
        self.e_bins     = create_bins('e')
        self.eta_bins   = create_bins('eta')
        self.theta_bins = create_bins('theta')
        self.phi_bins   = create_bins('phi')
        self.pt_bins    = create_bins('pt')
        
        self.vocab_size = self.num_special_tokens + self.num_particles + self.num_materials + len(self.e_bins) + len(self.eta_bins) + len(self.theta_bins) + len(self.phi_bins) + len(self.pt_bins)
        
        # Offsets for tokenization (since we need to eliminate repeat tokens)
        self.SPECIAL_TOKENS_OFFSET = 0
        self.PDGID_OFFSET = self.SPECIAL_TOKENS_OFFSET + self.num_special_tokens
        self.MATERIAL_OFFSET = self.PDGID_OFFSET + self.num_particles
        self.ENERGY_OFFSET = self.MATERIAL_OFFSET + self.num_materials
        self.ETA_OFFSET = self.ENERGY_OFFSET + len(self.e_bins)
        self.THETA_OFFSET = self.ETA_OFFSET + len(self.eta_bins)
        self.PHI_OFFSET = self.THETA_OFFSET + len(self.theta_bins)
        self.PT_OFFSET = self.PHI_OFFSET + len(self.phi_bins)

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
            ["Pt bins",         len(self.pt_bins),        self.token_range_str(self.PT_OFFSET, len(self.pt_bins)),                    self.token_min('pt'),     self.token_max('pt'),     self.token_step_size('pt')]
        ]
    
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
        return self.dictionary_data[f'{type_str}_bin_data'].get('step_size', 'none')
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