import json
import sys

config_file_path = sys.argv[1]
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
        
# If no output_dir_name is set, then use the dataset name
if (globals()['output_dir_name'] == ''):
    globals()['output_dir_name'] = globals()['dataset']