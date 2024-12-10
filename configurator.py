import json
import sys

config_file_path = sys.argv[1]
with open(config_file_path, 'r') as f:
    config = json.load(f)

for key, value in config.items():
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            globals()[sub_key] = sub_value
    else:
        globals()[key] = value