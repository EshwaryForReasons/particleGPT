
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pUtil
import data.filter_output as filter_output
import data.tokenizer as tokenizer
import configurator

import pTokenizerModule as pTokenizer

script_dir = os.path.dirname(os.path.abspath(__file__))

latest_sampling_dir = pUtil.get_latest_sampling_dir(configurator.output_dir_name)
configurator.samples_storage_dir = latest_sampling_dir

filtered_samples_file = os.path.join(configurator.samples_storage_dir, 'filtered_samples.csv')
untokenized_samples_file = os.path.join(configurator.samples_storage_dir, 'untokenized_samples.csv')
sorted_samples_file = os.path.join(configurator.samples_storage_dir, 'sorted_samples.csv')
sampled_leading_particles_file = os.path.join(configurator.samples_storage_dir, 'sampled_leading_particles.csv')

input_file = os.path.join('data', configurator.dataset, 'data.csv')
dictionary_file = os.path.join('data', configurator.dataset, 'dictionary.json')
sorted_inputs_file = os.path.join(configurator.samples_storage_dir, 'sorted_inputs.csv')
input_leading_particles_file = os.path.join(configurator.samples_storage_dir, 'input_leading_particles.csv')

# First we filter the generated_samples 
filter_output.init_data()
filter_output.ensure_event_borders()
filter_output.remove_malformed_events()
filter_output.ensure_valid_token_ranges()
filter_output.write_to_file()

# Next we untokenize the filtered_samples and extract the leading particles
pTokenizer.untokenize_data(dictionary_file, filtered_samples_file, untokenized_samples_file)
filter_output.extract_leading_particle(untokenized_samples_file, sampled_leading_particles_file)
filter_output.extract_leading_particle(input_file, input_leading_particles_file)

# # Now we output the data into the correct folder
input_dist_path = f'{latest_sampling_dir}/input_leading_particles.csv'
sample_dist_path = f'{latest_sampling_dir}/sampled_leading_particles.csv'

columns = ["num_particles", "pdgid", "e", "px", "py", "pz", "eta", "theta", "phi"]
bin_settings = {
    "num_particles": {"min": 0, "max": 50, "bins": 50},
    # "pdgid": {"min": -300, "max": 0, "bins": 10, "ymin": 0, "ymax": 10000},
    "e": {"min": 0, "max": 35000, "bins": 35},
    "px": {"min": 0, "max": 35000, "bins": 35},
    "py": {"min": 0, "max": 35000, "bins": 35},
    "pz": {"min": 0, "max": 35000, "bins": 35},
    "eta": {"min": -4, "max": 4, "bins": 81},
    "theta": {"min": -2 * np.pi, "max": 2 * np.pi, "bins": 125},
    "phi": {"min": -2 * np.pi, "max": 2 * np.pi, "bins": 125},
}

df1 = pd.read_csv(input_dist_path, sep=" ", names=columns, engine="python")
df2 = pd.read_csv(sample_dist_path, sep=" ", names=columns, engine="python")

for column, settings in bin_settings.items():
    min_val = settings["min"]
    max_val = settings["max"]
    bins = settings["bins"]
    
    df1_weights = np.ones_like(df1[column]) / len(df1[column])
    df2_weights = np.ones_like(df2[column]) / len(df2[column])

    plt.figure(figsize=(21, 6))
    plt.subplot(1, 2, 1)
    plt.hist(df1[column], bins=bins, weights=df1_weights, range=(min_val, max_val), edgecolor="black", alpha=0.7, color="blue", label="Input")
    plt.hist(df2[column], bins=bins, weights=df2_weights, range=(min_val, max_val), edgecolor="black", alpha=0.7, color="orange", label="Sampled")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{latest_sampling_dir}/histogram_{column}.png", bbox_inches='tight')