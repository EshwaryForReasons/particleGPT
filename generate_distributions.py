
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path
import pUtil
import configurator

import pTokenizerModule as pTokenizer

script_dir = Path(__file__).resolve().parent

latest_sampling_dir = pUtil.get_latest_sampling_dir(configurator.output_dir_name)
configurator.samples_storage_dir = latest_sampling_dir

generated_samples_filename = Path(configurator.samples_storage_dir, 'generated_samples.csv')
filtered_samples_filename = Path(configurator.samples_storage_dir, 'filtered_samples.csv')
untokenized_samples_filename = Path(configurator.samples_storage_dir, 'untokenized_samples.csv')
sampled_leading_particles_filename = Path(configurator.samples_storage_dir, 'sampled_leading_particles.csv')
real_leading_test_particles_data_filename = Path('data', configurator.dataset, 'outputs', 'real_leading_test_particles.csv')
dictionary_filename = Path('data', configurator.dataset, 'dictionary.json')

pTokenizer.load_dictionary(dictionary_filename.as_posix())
pTokenizer.load_tokenized_data(generated_samples_filename.as_posix())
pTokenizer.filter_data(filtered_samples_filename.as_posix())
pTokenizer.untokenize_data(untokenized_samples_filename.as_posix())
pTokenizer.output_generated_leading_particle_information(sampled_leading_particles_filename.as_posix())

with open(dictionary_filename) as dictionary_file:
    dictionary = json.load(dictionary_file)

columns = ["num_particles", "pdgid", "e", "px", "py", "pz", "eta", "theta", "phi"]
bin_settings = {
    "num_particles": {"min": 0, "max": 50, "bins": 50},
    # "pdgid": {"min": -300, "max": 0, "bins": 10, "ymin": 0, "ymax": 10000},
    "e": {"min": dictionary["e_bin_data"]["min"], "max": dictionary["e_bin_data"]["max"], "bins": int((dictionary["e_bin_data"]["max"] - dictionary["e_bin_data"]["min"]) // dictionary["e_bin_data"]["step_size"])},
    "px": {"min": dictionary["e_bin_data"]["min"], "max": dictionary["e_bin_data"]["max"], "bins": int((dictionary["e_bin_data"]["max"] - dictionary["e_bin_data"]["min"]) // 1000)},
    "py": {"min": dictionary["e_bin_data"]["min"], "max": dictionary["e_bin_data"]["max"], "bins": int((dictionary["e_bin_data"]["max"] - dictionary["e_bin_data"]["min"]) // 1000)},
    "pz": {"min": dictionary["e_bin_data"]["min"], "max": dictionary["e_bin_data"]["max"], "bins": int((dictionary["e_bin_data"]["max"] - dictionary["e_bin_data"]["min"]) // 1000)},
    "eta": {"min": dictionary["eta_bin_data"]["min"], "max": dictionary["eta_bin_data"]["max"], "bins": int((dictionary["eta_bin_data"]["max"] - dictionary["eta_bin_data"]["min"]) // dictionary["eta_bin_data"]["step_size"])},
    "theta": {"min": -2 * np.pi, "max": 2 * np.pi, "bins": int((4 * np.pi) // dictionary["theta_bin_data"]["step_size"])},
    "phi": {"min": -2 * np.pi, "max": 2 * np.pi, "bins": int((4 * np.pi) // dictionary["phi_bin_data"]["step_size"])},
}

# bin_settings = {
#     "num_particles": {"min": 0, "max": 50, "bins": 50},
#     # "pdgid": {"min": -300, "max": 0, "bins": 10, "ymin": 0, "ymax": 10000},
#     "e": {"min": 0, "max": 35000, "bins": 35},
#     "px": {"min": 0, "max": 35000, "bins": 35},
#     "py": {"min": 0, "max": 35000, "bins": 35},
#     "pz": {"min": 0, "max": 35000, "bins": 35},
#     "eta": {"min": -4, "max": 4, "bins": 81},
#     "theta": {"min": -2 * np.pi, "max": 2 * np.pi, "bins": 125},
#     "phi": {"min": -2 * np.pi, "max": 2 * np.pi, "bins": 125},
# }

df1 = pd.read_csv(real_leading_test_particles_data_filename, sep=" ", names=columns, engine="python", header=None)
df2 = pd.read_csv(sampled_leading_particles_filename, sep=" ", names=columns, engine="python", header=None)

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