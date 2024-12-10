# Utility for retrieving bins, tokens, vocabulary, and humanizing the dictionary

import numpy as np
import json
from enum import Enum

class ETypes(Enum):
    SPECIAL_TOKENS = 0
    PDGID = 1
    MATERIAL = 2
    ENERGY = 3
    ETA = 4
    THETA = 5
    PHI = 6

with open('dictionary.json', 'r') as f:
    data = json.load(f)

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

def particle_id_to_index(particle_id):
    particle_name = particles_id.get(str(particle_id))
    # -2 return will become -3 after 0 based indexing shift
    return particles_index.get(particle_name, -1) if particle_name else -2

def get_bins(type):
    if type == ETypes.ENERGY:
        return e_bins
    elif type == ETypes.ETA:
        return eta_bins
    elif type == ETypes.THETA:
        return theta_bins
    elif type == ETypes.PHI:
        return phi_bins

def get_offsets(type):
    if type == ETypes.SPECIAL_TOKENS:
        return SPECIAL_TOKENS_OFFSET
    elif type == ETypes.PDGID:
        return PDGID_OFFSET
    elif type == ETypes.MATERIAL:
        return MATERIAL_OFFSET
    elif type == ETypes.ENERGY:
        return ENERGY_OFFSET
    elif type == ETypes.ETA:
        return ETA_OFFSET
    elif type == ETypes.THETA:
        return THETA_OFFSET
    elif type == ETypes.PHI:
        return PHI_OFFSET

def get_vocab_size():
    return vocab_size

def get_special_tokens():
    return data['special_tokens']

table_data = [
    ["Type", "Num", "Token Range"],
    ["Special tokens", num_special_tokens, f"{SPECIAL_TOKENS_OFFSET} - {SPECIAL_TOKENS_OFFSET + num_special_tokens - 1}"],
    ["Particles", num_particles, f"{PDGID_OFFSET} - {PDGID_OFFSET + num_particles - 1}"],
    ["Materials", num_materials, f"{MATERIAL_OFFSET} - {MATERIAL_OFFSET + num_materials - 1}"],
    ["Energy bins", len(e_bins), f"{ENERGY_OFFSET} - {ENERGY_OFFSET + len(e_bins) - 1}"],
    ["Eta bins", len(eta_bins), f"{ETA_OFFSET} - {ETA_OFFSET + len(eta_bins) - 1}"],
    ["Theta bins", len(theta_bins), f"{THETA_OFFSET} - {THETA_OFFSET + len(theta_bins) - 1}"],
    ["Phi bins", len(phi_bins), f"{PHI_OFFSET} - {PHI_OFFSET + len(phi_bins) - 1}"]
]

def output_humanized_dictionary(output_file_path):
    # Define column widths for formatting
    col_widths = [max(len(str(row[col])) for row in table_data) + 2 for col in range(len(table_data[0]))]

    # Write the table to a text file
    with open(output_file_path, "w") as output_file:
        for row in table_data:
            formatted_row = "".join(str(cell).ljust(col_widths[idx]) for idx, cell in enumerate(row))
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
            output_file.write(f"{particle_name} - {token_value + PDGID_OFFSET - 1}\n")

        # Print materials and their corresponding token values
        output_file.write("\nMaterial Name - Token Value\n")
        for material_name, token_value in data['materials_named'].items():
            output_file.write(f"{material_name} - {token_value + MATERIAL_OFFSET - 1}\n")