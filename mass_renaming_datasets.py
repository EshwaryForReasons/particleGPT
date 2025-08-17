from pathlib import Path
import json

script_dir = Path(__file__).resolve().parent

# Update dataset names in config files
config_dir = script_dir / "config"
config_files = list(config_dir.glob("*.json"))
for config_file in config_files:
    with open(config_file, "r") as f:
        config = json.load(f)
        
    if config['dataset'] == "dataset_5.csv":
        config["dataset"] = "dataset_10M.csv"
    elif config['dataset'] == "dataset_2.csv":
        config["dataset"] = "dataset_100k.csv"
    elif config['dataset'] == "dataset_6.csv":
        config["dataset"] = "dataset_100M.csv"
        
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

# Update the model_2, model_5 prefixes in the config file names to model_100k, model_10M, etc. 
model_rename_map = {
    "model_2_": "model_100k_",
    "model_5_": "model_10M_",
    "model_6_": "model_100M_",
}

for old_prefix, new_prefix in model_rename_map.items():
    for file_path in config_dir.glob(f"{old_prefix}*.json"):
        new_name = file_path.name.replace(old_prefix, new_prefix, 1)
        new_path = file_path.with_name(new_name)
        print(f"Renaming: {file_path.name} → {new_name}")
        file_path.rename(new_path)
        
# Update dataset names in dictionary.json files
data_dir = script_dir / "data"
dict_files = list(data_dir.glob("**/dictionary.json"))
for dict_file in dict_files:
    with open(dict_file, "r") as f:
        data = json.load(f)
        
    if data['dataset'] == "dataset_5.csv":
        data["dataset"] = "dataset_10M.csv"
    elif data['dataset'] == "dataset_2.csv":
        data["dataset"] = "dataset_100k.csv"
    elif config['dataset'] == "dataset_6.csv":
        config["dataset"] = "dataset_100M.csv"
        
    with open(dict_file, "w") as f:
        json.dump(data, f, indent=4)

# Update the preparation_2, preparation_5 prefixes in the config file 'preparation_name' to preparation_100k, preparation_10M, etc. 
preparation_rename_map = {
    "preparation_2_": "preparation_100k_",
    "preparation_5_": "preparation_10M_",
    "preparation_6_": "preparation_100M_",
}

for config_file in config_files:
    with open(config_file, "r") as f:
        config = json.load(f)

    config['preparation_name'] = config['preparation_name'].replace('preparation_2_', 'preparation_100k_')
    config['preparation_name'] = config['preparation_name'].replace('preparation_5_', 'preparation_10M_')
    config['preparation_name'] = config['preparation_name'].replace('preparation_6_', 'preparation_100M_')
        
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

# Update the preparation_2, preparation_5 prefixes in preparation directories to preparation_100k, preparation_10M, etc. 
for old_prefix, new_prefix in preparation_rename_map.items():
    for subdir in data_dir.glob(f"{old_prefix}*"):
        if subdir.is_dir():
            new_name = subdir.name.replace(old_prefix, new_prefix, 1)
            new_path = subdir.with_name(new_name)
            print(f"Renaming directory: {subdir.name} → {new_name}")
            subdir.rename(new_path)
            
# Rename the models in the trained_models dir to match the new dataset names
models_dir = script_dir / "trained_models"
model_dirs = list(models_dir.glob("model_*/"))

for old_prefix, new_prefix in model_rename_map.items():
    for dir_path in models_dir.glob(f"{old_prefix}*"):
        if dir_path.is_dir():  # ensure it's a directory
            new_name = dir_path.name.replace(old_prefix, new_prefix, 1)
            new_path = dir_path.with_name(new_name)
            print(f"Renaming dir: {dir_path.name} → {new_name}")
            dir_path.rename(new_path)
            
for dir_path in models_dir.glob("*"):
    model_name = dir_path.stem
    # So far, no separate model name has been specified
    associated_config_file_name = model_name + ".json"
    associated_config_filepath = script_dir / 'config' / associated_config_file_name
    
    with open(associated_config_filepath, 'r') as f:
        config = json.load(f)
    
    entire_file = None
    with open(dir_path / 'train_log_1.jsonl', 'r') as f:
        entire_file = f.read()
    
    modified_file = []
    for line in entire_file.splitlines():
        jline = json.loads(line)
        if 'config_file_path' in jline:
            jline['config_file_path'] = 'config/' + associated_config_file_name
        if 'model_name' in jline:
            jline['model_name'] = model_name
        if 'preparation' in jline:
            jline['preparation'] = config["preparation_name"]
        if 'meta_path' in jline:
            jline['meta_path'] = 'data/' + config["preparation_name"] + '/meta.pkl'
        modified_file.append(jline)
    
    with open(dir_path / 'train_log_1.jsonl', 'w') as f:
        for line in modified_file:
            f.write(json.dumps(line) + '\n')

# Rename the model names in the generated_samples directory
generated_samples_dir = script_dir / "generated_samples"

for old_prefix, new_prefix in model_rename_map.items():
    for dir_path in generated_samples_dir.glob(f"{old_prefix}*"):
        if dir_path.is_dir():  # ensure it's a directory
            new_name = dir_path.name.replace(old_prefix, new_prefix, 1)
            new_path = dir_path.with_name(new_name)
            print(f"Renaming dir: {dir_path.name} → {new_name}")
            dir_path.rename(new_path)