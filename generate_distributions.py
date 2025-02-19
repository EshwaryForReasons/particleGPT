
import os
import matplotlib.pyplot as plt
import pandas as pd
import pUtil
import data.filter_output as filter_output
import data.tokenizer as tokenizer
import configurator

script_dir = os.path.dirname(os.path.abspath(__file__))

# exec(open('configurator.py').read())

latest_sampling_dir = pUtil.get_latest_sampling_dir(configurator.output_dir_name)
configurator.samples_storage_dir = latest_sampling_dir

filtered_samples_file = os.path.join(configurator.samples_storage_dir, 'filtered_samples.txt')
untokenized_samples_file = os.path.join(configurator.samples_storage_dir, 'untokenized_samples.csv')
sorted_samples_file = os.path.join(configurator.samples_storage_dir, 'sorted_samples.csv')
sampled_leading_particles_file = os.path.join(configurator.samples_storage_dir, 'sampled_leading_particles.csv')

input_file = os.path.join('data', configurator.dataset, 'data.csv')
sorted_inputs_file = os.path.join(configurator.samples_storage_dir, 'sorted_inputs.csv')
input_leading_particles_file = os.path.join(configurator.samples_storage_dir, 'input_leading_particles.csv')

# First we filter the generated_samples
filter_output.init_data()
filter_output.ensure_event_borders()
filter_output.remove_malformed_events()
filter_output.ensure_valid_token_ranges()
filter_output.write_to_file()

# Next we untokenize the filtered_samples and extract the leading particles
tokenizer.untokenize_data(filtered_samples_file, untokenized_samples_file)
filter_output.extract_leading_particle(untokenized_samples_file, sampled_leading_particles_file)
filter_output.extract_leading_particle(input_file, input_leading_particles_file)

# Now we output the data into the correct folder
input_dist_path = f'{latest_sampling_dir}/input_leading_particles.csv'
sample_dist_path = f'{latest_sampling_dir}/sampled_leading_particles.csv'

columns = ["num_particles", "pdgid", "e", "px", "py", "pz"]
bin_settings = {
    "num_particles": {"min": 0, "max": 50, "bins": 50, "ymin": 0, "ymax": 1000},
    # "pdgid": {"min": -300, "max": 0, "bins": 10, "ymin": 0, "ymax": 10000},
    "e": {"min": 0, "max": 35000, "bins": 35, "ymin": 0, "ymax": 10000},
    "px": {"min": 0, "max": 35000, "bins": 20, "ymin": 0, "ymax": 2000},
    "py": {"min": 0, "max": 35000, "bins": 20, "ymin": 0, "ymax": 2000},
    "pz": {"min": 0, "max": 35000, "bins": 20, "ymin": 0, "ymax": 2000},
}

df1 = pd.read_csv(input_dist_path, sep=" ", names=columns, engine="python")
df2 = pd.read_csv(sample_dist_path, sep=" ", names=columns, engine="python")

for column, settings in bin_settings.items():
    min_val = settings["min"]
    max_val = settings["max"]
    bins = settings["bins"]
    ymin = settings["ymin"]
    ymax = settings["ymax"]

    # Filter values to ensure they fall within the specified range
    filtered_data1 = df1[column][(df1[column] >= min_val) & (df1[column] <= max_val)]
    filtered_data2 = df2[column][(df2[column] >= min_val) & (df2[column] <= max_val)]

    plt.figure(figsize=(21, 6))
    plt.subplot(1, 2, 1)
    plt.hist(filtered_data1, bins=bins, range=(min_val, max_val), edgecolor="black", alpha=0.7, color="blue", label="Input")
    plt.hist(filtered_data2, bins=bins, range=(min_val, max_val), edgecolor="black", alpha=0.7, color="orange", label="Sampled")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{latest_sampling_dir}/histogram_{column}.png", bbox_inches='tight')