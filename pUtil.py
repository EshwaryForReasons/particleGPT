
from pathlib import Path

script_dir = Path(__file__).resolve().parent

GENERATED_SAMPLES_DIR_NAME = 'generated_samples'
TRAINED_MODELS_DIR_NAME = 'trained_models'
DATASETS_DIR_NAME = 'data'

def get_sampling_dir(in_dataset_out_name):
    return script_dir / GENERATED_SAMPLES_DIR_NAME / in_dataset_out_name

def get_training_dir(in_dataset_out_name):
    return script_dir / TRAINED_MODELS_DIR_NAME / in_dataset_out_name

def get_dataset_dir(in_dataset):
    return script_dir / DATASETS_DIR_NAME / in_dataset

# Sampling

def get_latest_sampling_id(in_dataset):
    generated_samples_dir = Path('generated_samples') / in_dataset
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

def get_latest_sampling_dir(in_dataset):
    latest_sampling_id = get_latest_sampling_id(in_dataset)
    return get_sampling_dir(in_dataset) / f'sampling_{latest_sampling_id}'