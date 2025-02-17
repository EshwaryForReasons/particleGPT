
import sys
import os
import subprocess
import glob
import datetime
import matplotlib.pyplot as plt
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

dataset = ""
output_dir_name = ""
exec(open('configurator.py').read())

def get_latest_folder():
    latest_time = None
    latest_file = None
    for file in glob.glob(os.path.join('generated_samples', output_dir_name, "*")):
        filename = os.path.basename(file)
        parts = filename.split("-")
        month, day, year, hour, minute, second = map(int, parts)
        
        # Convert to datetime object
        file_time = datetime.datetime(year, month, day, hour, minute, second)
        
        # Find the latest file
        if latest_time is None or file_time > latest_time:
            latest_time = file_time
            latest_file = file
    return latest_file

# Cache this in case we generate more samples so this doesn't change mid-use
latest_folder = os.path.join(get_latest_folder())

# First we filter the generated_samples
res = subprocess.run([sys.executable, os.path.join('data', 'filter_output.py'), dataset, latest_folder], cwd='.', capture_output=True, text=True)
print(res.stderr, res.stdout)

# Next we untokenize the filtered_samples and extract the learning particles
res = subprocess.run([sys.executable, os.path.join('data', 'generate_distribution.py'), dataset, latest_folder], cwd='.', capture_output=True, text=True)
print(res.stderr, res.stdout)

# Now we output the data into the correct folder
input_dist_path = f'{latest_folder}/input_leading_particles.csv'
sample_dist_path = f'{latest_folder}/sampled_leading_particles.csv'

columns = ["num_particles", "pdgid", "e", "px", "py", "pz"]
bin_settings = {
    "num_particles": {"min": 0, "max": 50, "bins": 50, "ymin": 0, "ymax": 10000},
    # "pdgid": {"min": -300, "max": 0, "bins": 10, "ymin": 0, "ymax": 10000},
    "e": {"min": 0, "max": 35000, "bins": 35, "ymin": 0, "ymax": 1000},
    "px": {"min": 0, "max": 35000, "bins": 20, "ymin": 0, "ymax": 10000},
    "py": {"min": 0, "max": 35000, "bins": 20, "ymin": 0, "ymax": 10000},
    "pz": {"min": 0, "max": 35000, "bins": 20, "ymin": 0, "ymax": 10000},
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
    plt.savefig(f"{latest_folder}/histogram_{column}.png", bbox_inches='tight')