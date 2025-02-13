
import numpy as np
import sys
import os

from dictionary import ETypes
from dictionary import get_offsets
from dictionary import get_num_tokens

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pLogging

script_dir = os.path.dirname(os.path.abspath(__file__))

# [1] Make sure all events are valid (start with EVENT_START and end with EVENT_END)
# [2] Each particle should contain 5 tokens. Any event that does not follow this rule is eliminated in pass 2.
# [3] Each token should be in its range (e.g 0-9 are reserved for special tokens). Pass 3 eliminates any event that does not follow this rule.

# Since this file can be called manually or through sample.py, we need to account for both cases when logging
logger_idx = -1
def set_logger(in_logger_idx):
    global logger_idx
    logger_idx = in_logger_idx
    
if logger_idx == -1:
    logger_idx = pLogging.create_logger("filter_output")
    
pLogging.info(logger_idx, "Started filtering generated samples.")

dataset_storage_dir = sys.argv[1]
output_file = os.path.join(dataset_storage_dir, 'generated_sample.txt')
filtered_output_file = os.path.join(dataset_storage_dir, 'filtered_sample.txt')
# output_file = os.path.join(script_dir, 'outputs/outputs.txt')
# filtered_output_file = os.path.join(script_dir, 'outputs/filtered_outputs.txt')

# Array containing all events
generated_events = []
with open(output_file, 'r') as f:
    for line in f:
        event = [np.array(line.split(), dtype=np.uint16)]
        generated_events.extend(event)
        
num_events_loaded = len(generated_events)

# Pass 1: Ensure event starts with EVENT_START and ends with EVENT_END
for event in generated_events:
    if event[0] != 1 or event[-1] != 2:
        generated_events.remove(event)
         
num_pass_one_survivors = len(generated_events)

pLogging.info(logger_idx, f"Removed {num_events_loaded - num_pass_one_survivors} invalid events.")

# Remove all special tokens as they are only for training (except padding)
generated_events = [[x for x in subarray if x not in [1, 2, 3, 4]] for subarray in generated_events]

# Pass 2: Remove any malformed events [2]
generated_events = [event for event in generated_events if len(event) % 5 == 0]

num_pass_two_survivors = len(generated_events)

pLogging.info(logger_idx, f"Removed {num_pass_one_survivors - num_pass_two_survivors} malformed events.")

# Pass 3: Ensure valid token ranges
token_ranges = [
    [get_offsets(ETypes.PDGID), get_offsets(ETypes.PDGID) + get_num_tokens(ETypes.PDGID) - 1],
    #[get_offsets(ETypes.MATERIAL), get_offsets(ETypes.MATERIAL) + get_num_tokens(ETypes.MATERIAL) - 1],
    [get_offsets(ETypes.ENERGY), get_offsets(ETypes.ENERGY) + get_num_tokens(ETypes.ENERGY) - 1],
    [get_offsets(ETypes.ETA), get_offsets(ETypes.ETA) + get_num_tokens(ETypes.ETA) - 1],
    [get_offsets(ETypes.THETA), get_offsets(ETypes.THETA) + get_num_tokens(ETypes.THETA) - 1],
    [get_offsets(ETypes.PHI), get_offsets(ETypes.PHI) + get_num_tokens(ETypes.PHI) - 1]
]

for event in generated_events:
    running_count = 0
    for token in event:
        if token < token_ranges[running_count][0] - 1 or token > token_ranges[running_count][1]:
            generated_events.remove(event)
            print(token, token_ranges[running_count][0], token_ranges[running_count][1], running_count, event)
            break
        running_count = (running_count + 1) % 5

num_pass_three_survivors = len(generated_events)

pLogging.info(logger_idx, f"Removed {num_pass_two_survivors - num_pass_three_survivors} events with invalid token ranges.")
    
# Write filtered events to file
with open(filtered_output_file, 'w') as f:
    for event in generated_events:
        f.write(' '.join(map(str, event)) + '\n')
        
pLogging.info(logger_idx, f"Finished filtering output.")