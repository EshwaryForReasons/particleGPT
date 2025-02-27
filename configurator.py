import json
import sys
import os
import torch

config_file_path = sys.argv[1]

# Common

dataset = ''
output_dir_name = ''

# Sampling variables

samples_storage_dir = ''
max_new_tokens = 500
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# Job submission variables

nodes = 1
time = "00:10:00"
constraint = "gpu"
gpus = 1
cpus_per_task = 32
ntasks_per_node = 1
account = ""
quality_of_service = "debug"
use_shifter = False
shifter_image = ""
command = ""

# Configurator
    
print("FILE LOCATED: ", config_file_path)
with open(config_file_path, 'r') as f:
    config = json.load(f)

globals()['config_file_path'] = config_file_path

for key, value in config.items():
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            globals()[sub_key] = sub_value
    else:
        globals()[key] = value
        
# If no output_dir_name is set, then use the config file name.
# This saves a lot of pain with accidentally overwriting files.
if (globals()['output_dir_name'] == ''):
    config_file_name = os.path.basename(config_file_path)
    config_file_name_stripped = os.path.splitext(config_file_name)
    globals()['output_dir_name'] = config_file_name_stripped[0]