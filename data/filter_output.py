
import numpy as np
import sys
import os

from data.dictionary import ETypes
import data.dictionary as dictionary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pLogging
import configurator

# [1] Make sure all events are valid (start with EVENT_START and end with EVENT_END)
# [2] Each particle should contain 5 tokens. Any event that does not follow this rule is eliminated in pass 2.
# [3] Each token should be in its range (e.g 0-9 are reserved for special tokens). Pass 3 eliminates any event that does not follow this rule.

logger_idx = pLogging.create_logger("filter_output")
pLogging.info(logger_idx, "Started filtering generated samples.")

def init_data():
    global generated_events
    global survivors

    # Array containing all events
    generated_events = []
    generated_samples_file = os.path.join(configurator.samples_storage_dir, 'generated_samples.txt')
    with open(generated_samples_file, 'r') as f:
        for line in f:
            event = [np.array(line.split(), dtype=np.uint16)]
            generated_events.extend(event)
            
    survivors = len(generated_events)

# PASS 1: Make sure events start with EVENT_START and end with EVENT_END
def ensure_event_borders():
    global generated_events
    global survivors
    
    for event in generated_events:
        if event[0] != dictionary.get_special_tokens()['event_start'] or event[-1] != dictionary.get_special_tokens()['event_end']:
            generated_events.remove(event)
    
    pass_survivors = len(generated_events)
    pLogging.info(logger_idx, f"Removed {survivors - pass_survivors} invalid events.")
    survivors = pass_survivors
    
    # Remove all special tokens as they are only for training (except padding)
    generated_events = [[x for x in subarray if x not in [1, 2, 3, 4]] for subarray in generated_events]

# PASS 2: Remove any malformed events
def remove_malformed_events():
    global generated_events
    global survivors
    
    generated_events = [event for event in generated_events if len(event) % 5 == 0]
    num_pass_two_survivors = len(generated_events)
    pLogging.info(logger_idx, f"Removed {survivors - num_pass_two_survivors} malformed events.")
    survivors = num_pass_two_survivors

# PASS 3: Ensure valid token ranges
def ensure_valid_token_ranges():
    global generated_events
    global survivors
    
    token_ranges = [
        [dictionary.get_offsets(ETypes.PDGID), dictionary.get_offsets(ETypes.PDGID) + dictionary.get_num_tokens(ETypes.PDGID) - 1],
        [dictionary.get_offsets(ETypes.ENERGY), dictionary.get_offsets(ETypes.ENERGY) + dictionary.get_num_tokens(ETypes.ENERGY) - 1],
        [dictionary.get_offsets(ETypes.ETA), dictionary.get_offsets(ETypes.ETA) + dictionary.get_num_tokens(ETypes.ETA) - 1],
        [dictionary.get_offsets(ETypes.THETA), dictionary.get_offsets(ETypes.THETA) + dictionary.get_num_tokens(ETypes.THETA) - 1],
        [dictionary.get_offsets(ETypes.PHI), dictionary.get_offsets(ETypes.PHI) + dictionary.get_num_tokens(ETypes.PHI) - 1]
    ]

    for event in generated_events:
        running_count = 0
        for token in event:
            if token < token_ranges[running_count][0] - 1 or token > token_ranges[running_count][1]:
                generated_events.remove(event)
                break
            running_count = (running_count + 1) % 5

    pass_survivors = len(generated_events)
    pLogging.info(logger_idx, f"Removed {survivors - pass_survivors} events with invalid token ranges.")
    survivors = pass_survivors
    
# Write filtered events to file
def write_to_file():
    filtered_samples_file = os.path.join(configurator.samples_storage_dir, 'filtered_samples.txt')
    with open(filtered_samples_file, 'w') as f:
        for event in generated_events:
            f.write(' '.join(map(str, event)) + '\n')
            
# Input file is the data to work on. Output is where we store the extracted leading particles.
def extract_leading_particle(input_filename, output_filename):
    # Reorder each event to keep input event first and sort the rest by energy
    with open(input_filename, 'r') as in_file, open(output_filename, 'w') as out_file:
        for line in in_file:
            particles = line.strip().split(';')
            # Does not include primary particle
            num_particles = len(particles) - 1
            # Secondaries are sorted by energy
            secondary_particles = [particle.strip().split() for particle in particles[1:] if particle.strip() if particle]
            secondary_particles.sort(key=lambda x: float(x[1]), reverse=True)
            out_file.write(f'{num_particles} {" ".join(token for token in secondary_particles[0])}\n')

pLogging.info(logger_idx, f"Finished filtering output.")