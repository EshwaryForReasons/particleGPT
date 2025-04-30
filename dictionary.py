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

class Dictionary():
    def __init__(self, dictionary_filename):
        self.dictionary_filename = dictionary_filename
        
        with open(dictionary_filename, 'r') as f:
            self.dictionary_data = json.load(f)
            
        self.num_special_tokens = len(self.dictionary_data['special_tokens'])
        self.num_particles = len(self.dictionary_data['particles_index'])
        self.num_materials = len(self.dictionary_data['materials_named'])

        self.num_special_tokens = len(self.dictionary_data['special_tokens'])
        self.num_particles      = len(self.dictionary_data['particles_index'])
        self.num_materials      = len(self.dictionary_data['materials_named'])
        # Generate the bins based on the data in the file
        self.e_bins     = custom_arange(self.token_min('e'), self.token_max('e'), self.token_step_size('e'))
        self.eta_bins   = custom_arange(self.token_min('eta'), self.token_max('eta'), self.token_step_size('eta'))
        self.theta_bins = custom_arange(self.token_min('theta'), self.token_max('theta'), self.token_step_size('theta'))
        self.phi_bins   = custom_arange(self.token_min('phi'), self.token_max('phi'), self.token_step_size('phi'))
        self.pt_bins    = custom_arange(self.token_min('pt'), self.token_max('pt'), self.token_step_size('pt'))

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
        self.particles_index = self.dictionary_data['particles_index']
        self.particles_id = self.dictionary_data['particles_id']
        
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
        return self.dictionary_data[f'{type_str}_bin_data']['step_size']
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
            output_file.write("Particle Name - Token Value\n")
            for particle_name, token_value in self.dictionary_data['particles_index'].items():
                output_file.write(f"{particle_name} - {token_value + self.PDGID_OFFSET}\n")

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

        # Create particles_id and particles_index dictionaries
        dictionary_particles_id = {}
        dictionary_particles_index = {}
        for idx, pid in enumerate(particle_ids):
            p = Particle.from_pdgid(pid)
            dictionary_particles_id[pid] = p.name
            dictionary_particles_index[p.name] = idx

        # Update dictionary.json
        with open(dictionary_filename, 'r') as f:
            dictionary = json.load(f)

        with open(dictionary_filename, 'w') as f:
            dictionary["particles_id"] = dictionary_particles_id
            dictionary["particles_index"] = dictionary_particles_index
            json.dump(dictionary, f, indent=2)