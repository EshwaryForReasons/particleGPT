# Utility functions for tokenizing and padding data

import numpy as np
import math
import os
import sys

from data.dictionary import ETypes
import data.dictionary as dictionary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configurator

# PDGID's found here: https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
# We only consider the interval [-4, 4] for pseudorapidity

script_dir = os.path.dirname(os.path.abspath(__file__))

# Adds event_start and event_end tokens and padding placeholders
def preprocess_data(in_filename, out_filename):
    with open(os.path.join(script_dir, configurator.output_dir_name, in_filename), 'r') as file:
        max_generations = max(len(line.strip().split(';')) for line in file)
    
    padding_placeholder = "PADDING"
    with open(os.path.join(script_dir, configurator.output_dir_name, in_filename), 'r') as file, open(os.path.join(script_dir, configurator.output_dir_name, out_filename), 'w') as output_file:
        for line in file:
            generations = line.strip().split(';') + ["EVENT_END"] + [padding_placeholder] * (max_generations - len(line.strip().split(';')))
            output_file.write("EVENT_START;" + ';'.join(generations) + "\n")

def tokenize_particle(particle):
    if particle[0] == 'EVENT_START':
        return dictionary.get_special_tokens()['event_start']
    elif particle[0] == 'EVENT_END':
        return dictionary.get_special_tokens()['event_end']
    elif particle[0] == 'PADDING':
        return [dictionary.get_special_tokens()['particle_start']] + [dictionary.get_special_tokens()['padding']] * 5 + [dictionary.get_special_tokens()['particle_end']]
    
    pdgid = int(particle[0])
    e, px, py, pz = map(float, (particle[1], particle[2], particle[3], particle[4]))

    # Ensure non-zero momentum
    if px == py == pz == 0:
        return [dictionary.get_special_tokens()['padding']] * 5
    
    # Convert to spherical (theta - polar; phi - azimuthal)
    r           = math.sqrt(px * px + py * py + pz * pz)
    theta       = np.arccos(pz / r)
    phi         = math.atan2(py, px)
    eta         = -np.log(np.tan(theta / 2))

    # If eta larger than 4, we ignore this event
    if np.abs(eta) > 4:
        return 'IGNORE'

    # -1 to convert to 0-based indexing (the superior kind)
    pdgid_idx   = dictionary.particle_id_to_index(pdgid) + dictionary.get_offsets(ETypes.PDGID)
    e_idx       = np.digitize(e, dictionary.get_bins(ETypes.ENERGY)) + dictionary.get_offsets(ETypes.ENERGY)
    eta_idx     = np.digitize(eta, dictionary.get_bins(ETypes.ETA)) + dictionary.get_offsets(ETypes.ETA)
    theta_idx   = np.digitize(theta, dictionary.get_bins(ETypes.THETA)) + dictionary.get_offsets(ETypes.THETA)
    phi_idx     = np.digitize(phi, dictionary.get_bins(ETypes.PHI)) + dictionary.get_offsets(ETypes.PHI)
    return [dictionary.get_special_tokens()['particle_start'], pdgid_idx, e_idx, eta_idx, theta_idx, phi_idx, dictionary.get_special_tokens()['particle_end']]

# Input file is expected to be preprocessed
def tokenize_data(in_filename, out_filename):
    input_file = open(os.path.join(script_dir, configurator.output_dir_name, in_filename), 'r')
    output_file = open(os.path.join(script_dir, configurator.output_dir_name, out_filename), 'w')

    for event in input_file:
        tokenized_sequence = np.array([], dtype=np.int32)
        particles = event.split(';')
        
        keep_event = True
        for particle in particles:
            data = particle.split()
            tokenized_particle = tokenize_particle(data)
            # If any conditions for ignoring is met, we ignore the event
            if tokenized_particle == 'IGNORE':
                keep_event = False
                break
            tokenized_sequence = np.append(tokenized_sequence, tokenized_particle)
        
        if keep_event:
            output_file.write(" ".join(map(str, tokenized_sequence)) + '\n')
    
    input_file.close()
    output_file.close()

# Particle here is an array of five numbers
def untokenize_particle(particle):
    # Ignore padding particles in case we use that tokenization scheme
    if int(particle[0]) == 0 and int(particle[1]) == 0 and int(particle[2]) == 0 and int(particle[3]) == 0 and int(particle[4]) == 0:
        return "0 0 0 0 0"
    
    # print(particle)
    
    pdgid_idx   = int(particle[0]) - dictionary.get_offsets(ETypes.PDGID) + 1
    e_idx       = int(particle[1]) - dictionary.get_offsets(ETypes.ENERGY) + 1
    eta_idx     = int(particle[2]) - dictionary.get_offsets(ETypes.ETA) + 1
    theta_idx   = int(particle[3]) - dictionary.get_offsets(ETypes.THETA) + 1
    phi_idx     = int(particle[4]) - dictionary.get_offsets(ETypes.PHI) + 1
    
    # print(pdgid_idx, e_idx, eta_idx, theta_idx, phi_idx)
    
    pdgid = dictionary.particle_index_to_id(pdgid_idx)
    e = dictionary.get_bin_median(ETypes.ENERGY, e_idx)
    # eta = get_bin_median(ETypes.ETA, eta_idx)
    theta = dictionary.get_bin_median(ETypes.THETA, theta_idx)
    phi = dictionary.get_bin_median(ETypes.PHI, phi_idx)
    
    p = e
    pz = p * math.cos(theta)
    px = p * math.sin(theta) * math.cos(phi)
    py = p * math.sin(theta) * math.sin(phi)
    
    return f"{pdgid} {e:.5f} {px:.5f} {py:.5f} {pz:.5f};"

# Input file is assumed to be filtered GPT output (already tokenized)
def untokenize_data(in_filename, out_filename):
    input_file = open(in_filename, 'r')
    output_file = open(out_filename, 'w')

    for event in input_file:
        # print(event)
        untokenized_sequence = np.array([], dtype=np.int32)
        particles = event.split()
        particles = [particles[i:i+5] for i in range(0, len(particles), 5)]
        
        for particle in particles:
            untokenized_particle = untokenize_particle(particle)
            untokenized_sequence = np.append(untokenized_sequence, untokenized_particle)
        
        output_file.write(" ".join(map(str, untokenized_sequence)) + '\n')
    
    input_file.close()
    output_file.close()