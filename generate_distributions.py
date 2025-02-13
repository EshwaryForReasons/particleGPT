
import sys
import os
import subprocess
import glob
import datetime

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

base_dir = os.path.join(get_latest_folder())

# First we filter the generated_samples
res = subprocess.run([sys.executable, os.path.join('data', dataset, 'filter_output.py'), base_dir], cwd='.', capture_output=True, text=True)
print(res.stderr, res.stdout)

# Next we untokenize the filtered_samples
res = subprocess.run([sys.executable, os.path.join('data', dataset, 'generate_distribution.py'), base_dir], cwd='.', capture_output=True, text=True)
print(res.stderr, res.stdout)