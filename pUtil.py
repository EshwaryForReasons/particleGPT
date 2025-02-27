
import glob
import os
import sys
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))

GENERATED_SAMPLES_DIR_NAME = 'generated_samples'
TRAINED_MODELS_DIR_NAME = 'trained_models'
DATASETS_DIR_NAME = 'data'

def get_sampling_dir(in_dataset_out_name):
    return os.path.join(script_dir, GENERATED_SAMPLES_DIR_NAME, in_dataset_out_name)

def get_training_dir(in_dataset_out_name):
    return os.path.join(script_dir, TRAINED_MODELS_DIR_NAME, in_dataset_out_name)

def get_dataset_dir(in_dataset):
    return os.path.join(script_dir, DATASETS_DIR_NAME, in_dataset)

# Sampling

def get_latest_sampling_id(in_dataset):
    generated_samples_dir = os.path.join('generated_samples', in_dataset)
    largest_sampling = 0
    for folder in glob.glob(os.path.join(generated_samples_dir, "*")):
        if os.path.isdir(folder):
            sampling_id = int(folder.split("_")[-1])
            if sampling_id > largest_sampling:
                largest_sampling = sampling_id
    return largest_sampling

def get_latest_sampling_dir(in_dataset):
    latest_sampling_id = get_latest_sampling_id(in_dataset)
    return os.path.join(get_sampling_dir(in_dataset), f'sampling_{latest_sampling_id}')