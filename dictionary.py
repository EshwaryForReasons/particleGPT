# Utility for retrieving bins, tokens, vocabulary, and humanizing the dictionary

from particle import Particle
import numpy as np
import json
from enum import Enum
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configurator

script_dir = os.path.dirname(os.path.abspath(__file__))

dictionary_filename = Path(script_dir, 'data', configurator.dataset, 'dictionary.json')
with open(dictionary_filename, 'r') as f:
    data = json.load(f)

class ETypes(Enum):
    SPECIAL_TOKENS = 0
    PDGID = 1
    MATERIAL = 2
    ENERGY = 3
    ETA = 4
    THETA = 5
    PHI = 6

# Function to evaluate radian expressions in the dictionary
def eval_rad(expression):
   return eval(expression.replace('pi', 'np.pi').replace('-np.pi', '-1*np.pi').replace('np.pi', '*np.pi'))

# Generate the bins based on the data in the file
e_bins = np.arange(data['e_bin_data']['min'], data['e_bin_data']['max'], data['e_bin_data']['step_size']).tolist()
eta_bins = np.arange(data['eta_bin_data']['min'], data['eta_bin_data']['max'], data['eta_bin_data']['step_size']).tolist()
theta_bins = np.arange(eval_rad(data['theta_bin_data']['min']), eval_rad(data['theta_bin_data']['max']), data['theta_bin_data']['step_size']).tolist()
phi_bins = np.arange(eval_rad(data['phi_bin_data']['min']), eval_rad(data['phi_bin_data']['max']), data['phi_bin_data']['step_size']).tolist()

num_special_tokens = len(data['special_tokens'])
num_particles = len(data['particles_index'])
num_materials = len(data['materials_named'])

vocab_size = num_special_tokens + num_particles + num_materials + len(e_bins) + len(eta_bins) + len(theta_bins) + len(phi_bins)

# Offsets for tokenization (since we need to eliminate repeat tokens)
SPECIAL_TOKENS_OFFSET = 0
PDGID_OFFSET = SPECIAL_TOKENS_OFFSET + num_special_tokens
MATERIAL_OFFSET = PDGID_OFFSET + num_particles
ENERGY_OFFSET = MATERIAL_OFFSET + num_materials
ETA_OFFSET = ENERGY_OFFSET + len(e_bins)
THETA_OFFSET = ETA_OFFSET + len(eta_bins)
PHI_OFFSET = THETA_OFFSET + len(theta_bins)

# Converts input particle ID to the relevant index
particles_index = data['particles_index']
particles_id = data['particles_id']

def get_vocab_size():
    return vocab_size

def get_special_tokens():
    return data['special_tokens']

table_data = [
    ["Type",            "Num",              "Token Range",                                                                      "Min",                          "Max",                          "Step Size"],
    ["Special tokens",  num_special_tokens, f"{SPECIAL_TOKENS_OFFSET} - {SPECIAL_TOKENS_OFFSET + num_special_tokens - 1}",      "N/A",                          "N/A",                          "N/A"],
    ["Particles",       num_particles,      f"{PDGID_OFFSET} - {PDGID_OFFSET + num_particles - 1}",                             "N/A",                          "N/A",                          "N/A"],
    ["Materials",       num_materials,      f"{MATERIAL_OFFSET} - {MATERIAL_OFFSET + num_materials - 1}",                       "N/A",                          "N/A",                          "N/A"],
    ["Energy bins",     len(e_bins),        f"{ENERGY_OFFSET} - {ENERGY_OFFSET + len(e_bins) - 1}",                             data['e_bin_data']['min'],      data['e_bin_data']['max'],      data['e_bin_data']['step_size']],
    ["Eta bins",        len(eta_bins),      f"{ETA_OFFSET} - {ETA_OFFSET + len(eta_bins) - 1}",                                 data['eta_bin_data']['min'],    data['eta_bin_data']['max'],    data['eta_bin_data']['step_size']],
    ["Theta bins",      len(theta_bins),    f"{THETA_OFFSET} - {THETA_OFFSET + len(theta_bins) - 1}",                           data['theta_bin_data']['min'],  data['theta_bin_data']['max'],  data['theta_bin_data']['step_size']],
    ["Phi bins",         len(phi_bins),     f"{PHI_OFFSET} - {PHI_OFFSET + len(phi_bins) - 1}",                                 data['phi_bin_data']['min'],    data['phi_bin_data']['max'],    data['phi_bin_data']['step_size']]
]

def output_humanized_dictionary(output_file_path):
    # Define column widths for formatting
    col_widths = [max(len(str(row[col])) for row in table_data) + 3 for col in range(len(table_data[0]))]

    # Write the table to a text file
    with open(output_file_path, "w") as output_file:
        for row in table_data:
            formatted_row = "".join(str(cell).ljust(col_widths[idx]) + "|  " for idx, cell in enumerate(row))
            output_file.write(formatted_row + "\n")
            
        output_file.write("\n")

        # Print particles and their corresponding token values
        output_file.write("Special Token - Token Value\n")
        for special_token, token_value in data['special_tokens'].items():
            output_file.write(f"{special_token} - {token_value + SPECIAL_TOKENS_OFFSET}\n")
            
        output_file.write("\n")
            
        # Print particles and their corresponding token values
        output_file.write("Particle Name - Token Value\n")
        for particle_name, token_value in data['particles_index'].items():
            output_file.write(f"{particle_name} - {token_value + PDGID_OFFSET}\n")

        # Print materials and their corresponding token values
        output_file.write("\nMaterial Name - Token Value\n")
        for material_name, token_value in data['materials_named'].items():
            output_file.write(f"{material_name} - {token_value + MATERIAL_OFFSET}\n")

# Updates the particles list in dictionary.json based on the input data
def update_dictionary_particle_list(input_data_filename, dictionary_filename):
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