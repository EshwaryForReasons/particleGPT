import os

# Sampling will require a leading set of tokens which GPT will then complete.
# To achieve accurate sampling that leading set should be the same as the input data.
# So we isolate each leading particle into a file so we can pass this into the sampling script
# and get accurate sampling for the data.

script_dir = os.path.dirname(os.path.abspath(__file__))

tokenized_data_filename = os.path.join(script_dir, 'outputs/temp_tokenized.csv')
sampling_lead_filename = os.path.join(script_dir, 'outputs/temp_sampling_lead.csv')

with open(tokenized_data_filename, 'r') as in_file, open(sampling_lead_filename, 'w') as out_file:
    for event in in_file:
        first_event = event.split(' 4 ')[0]
        first_event += ' 4\n'
        out_file.write(first_event)