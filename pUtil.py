
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

# If model_name is None, it returns the global temp directory
def get_temp_dir(model_name=None):
    if model_name is None:
        return script_dir / TEMP_DIR_NAME
    raise ValueError("model_name must be None to get the global temp directory")

# Gets the config file of a model given the model name
def get_model_config_filename(model_name):
    atlas_filename = script_dir / 'atlas.json'
    with open(atlas_filename, 'r') as f:
        atlas = json.load(f)
    
    config_filename = atlas['models'][model_name]['config_file']
    config_filename = script_dir / CONFIG_DIR_NAME / config_filename
    return config_filename

# Gets the meta file of the preparation a model is trained on
def get_model_meta_filename(model_name):
    atlas_filename = script_dir / 'atlas.json'
    with open(atlas_filename, 'r') as f:
        atlas = json.load(f)
    
    preparation_dirname = atlas['models'][model_name]['preparation_name']
    meta_filename = script_dir / DATASETS_DIR_NAME / preparation_dirname / 'meta.pkl'
    return meta_filename

# Gets the preparation directory of a model given the model name
def get_model_preparation_dir(model_name):
    atlas_filename = script_dir / 'atlas.json'
    with open(atlas_filename, 'r') as f:
        atlas = json.load(f)
    
    preparation_dirname = atlas['models'][model_name]['preparation_name']
    preparation_dir = script_dir / DATASETS_DIR_NAME / preparation_dirname
    return preparation_dir

def get_all_model_names():
    atlas_filename = script_dir / 'atlas.json'
    with open(atlas_filename, 'r') as f:
        atlas = json.load(f)
    
    model_names = list(atlas['models'].keys())
    return model_names

# Sampling

def get_latest_sampling_id(model_name):
    generated_samples_dir = Path(GENERATED_SAMPLES_DIR_NAME) / model_name
    largest_sampling = 0
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