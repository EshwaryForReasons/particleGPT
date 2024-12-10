
import numpy as np

# [1] GPT generates events until we tell it to stop, it has no notion of stop generation when 'EVENT_END' token is reached.
#     This means that the last event in the output file may be incomplete. Pass 1 filters out all incomplete events.
# [2] Each particle should contain 5 tokens. Any event that does not follow this rule is eliminated in pass 2.
# [3] Each token should be in its range (e.g 0-9 are reserved for special tokens). Pass 3 eliminates any event that does not follow this rule.

output_file = 'output.txt'
filtered_output_file = 'filtered_output.txt'

# Array containing all events
generated_events = []

# Pass 1: Remove all incomplete events [1]
with open(output_file, 'r') as f:
    for line in f:
        arr = np.array(line.split(), dtype=np.uint16)
        split_indices = np.where(arr == 2)[0] + 1
        split_arrays = np.split(arr, split_indices)
        
        # Any event that does not end in a '2' token (EVENT_END) is invalid
        filtered_split_arrays = [arr for arr in split_arrays if arr[-1] == 2]
        
        generated_events.extend(filtered_split_arrays)

# Pass 1.5: Remove all special tokens as they are only for training (except padding)
generated_events = [[x for x in subarray if x not in [1, 2, 3, 4]] for subarray in generated_events]

# Pass 2: Remove any malformed events [2]
filtered_events = [event for event in generated_events if (len(event) - 2) % 5 == 0]

# Pass 3: Ensure valid token ranges


# Write filtered events to file
with open(filtered_output_file, 'w') as f:
    for event in filtered_events:
        f.write(' '.join(map(str, event)) + '\n')

print("This script filters any malformed events.")
print("-----------------------------------------")
print()
print(f"Read {len(generated_events)} events.")
print(f"Filtered {len(generated_events) - len(filtered_events)} malformed events!")
print(f"{len(filtered_events)} events remaining.")
print()
print(f"Well-formed events written to '{filtered_output_file}'.")
