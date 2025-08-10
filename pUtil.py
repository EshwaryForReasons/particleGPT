
import json
import csv
from pathlib import Path

script_dir = Path(__file__).resolve().parent

CONFIG_DIR_NAME = 'config'
GENERATED_SAMPLES_DIR_NAME = 'generated_samples'
TRAINED_MODELS_DIR_NAME = 'trained_models'
DATASETS_DIR_NAME = 'data'
TEMP_DIR_NAME = 'temp'

def get_sampling_dir(model_name):
    return script_dir / GENERATED_SAMPLES_DIR_NAME / model_name

def get_training_dir(model_name):
    return script_dir / TRAINED_MODELS_DIR_NAME / model_name

def get_preparation_dir(preparation_name):
    return script_dir / DATASETS_DIR_NAME / preparation_name

def get_data_dir():
    return script_dir / DATASETS_DIR_NAME

# If model_name is None, it returns the global temp directory
def get_temp_dir(model_name=None):
    if model_name is None:
        return script_dir / TEMP_DIR_NAME
    raise ValueError("model_name must be None to get the global temp directory")




def get_model_config_filepath(model_name):
    # The model name will be the name of the config file unless specified otherwise within the file.
    config_dir = script_dir / CONFIG_DIR_NAME
    config_files = list(config_dir.glob("*.json"))
    
    correct_config_file = None
    for config_file in config_files:
        with open(config_file, "r") as f:
            data = json.load(f)
            if config_file.stem == model_name:
                correct_config_file = config_file
                break
            elif 'model_name' in data and data['model_name'] == model_name:
                correct_config_file = config_file
                break
    
    if correct_config_file is None:
        raise ValueError(f"No config file found for model name {model_name}.")
    
    return correct_config_file
                    
# Gets the meta file of the preparation a model is trained on
def get_model_meta_filepath(model_name):
    config_filepath = get_model_config_filepath(model_name)
    with open(config_filepath, 'r') as f:
        config = json.load(f)
        preparation_name = config.get('preparation_name', None)
    
    if preparation_name is None:
        raise ValueError(f"No preparation name found in config for model {model_name}.")

    meta_filepath = script_dir / DATASETS_DIR_NAME / preparation_name / 'meta.pkl'
    return meta_filepath

# Gets the preparation directory of a model given the model name
def get_model_preparation_dir(model_name):
    config_filepath = get_model_config_filepath(model_name)
    with open(config_filepath, 'r') as f:
        config = json.load(f)
        preparation_name = config.get('preparation_name', None)
    
    if preparation_name is None:
        raise ValueError(f"No preparation name found in config for model {model_name}.")

    preparation_dir = script_dir / DATASETS_DIR_NAME / preparation_name
    return preparation_dir
    
def get_model_preparation_name(model_name):
    config_filepath = get_model_config_filepath(model_name)
    with open(config_filepath, 'r') as f:
        config = json.load(f)
        preparation_name = config.get('preparation_name', None)
    
    if preparation_name is None:
        raise ValueError(f"No preparation name found in config for model {model_name}.")

    return preparation_name

def get_model_dataset_name(model_name):
    config_filepath = get_model_config_filepath(model_name)
    with open(config_filepath, 'r') as f:
        config = json.load(f)
        dataset_name = config.get('dataset', None)
    
    if dataset_name is None:
        raise ValueError(f"No dataset name found in config for model {model_name}.")

    return dataset_name

# Sampling

def get_latest_sampling_id(model_name):
    generated_samples_dir = script_dir / Path(GENERATED_SAMPLES_DIR_NAME) / model_name
    largest_sampling = -1
    for folder in generated_samples_dir.glob("sampling_*"):
        if folder.is_dir():
            try:
                sampling_id = int(folder.name.split("_")[-1])
                if sampling_id > largest_sampling:
                    largest_sampling = sampling_id
            except ValueError:
                continue
    return largest_sampling

def get_latest_sampling_dir(model_name):
    latest_sampling_id = get_latest_sampling_id(model_name)
    return get_sampling_dir(model_name) / f'sampling_{latest_sampling_id}'

# csv file utils

def count_rows(csv_filename):
    # This takes about a minute to run on 100M events. Fastest way I can think of.
    with open(csv_filename, 'rb') as f:
        return sum(buf.count(b'\n') for buf in iter(lambda: f.read(1024 * 1024), b''))
    
def count_columns(csv_filename, delim=' '):
    # Since the csv is uniform, simply the first line can provide the max sequence length
    with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        first_line = next(reader)
        return len(first_line[0].split(delim))

def concat_csv_files(csv_filepaths, out_filename, log_outputs=False):
    with open(out_filename, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.writer(outfile)

        for file_path in csv_filepaths:
            if log_outputs:
                print(f"Processing {file_path}...")

            with open(file_path, "r", encoding="utf-8") as infile:
                reader = csv.reader(infile)
                writer.writerows(reader)