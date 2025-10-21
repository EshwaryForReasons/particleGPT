from pathlib import Path
import shutil
from tqdm import tqdm

# Set these paths
source_dir = Path("/pscratch/sd/e/eshy/particleGPT/data")
backup_dir = Path("/global/cfs/cdirs/m3443/data/LLMStudy/HadronicInteractions/particleGPT_data")

# Create destination directory if it doesn't exist
backup_dir.mkdir(parents=True, exist_ok=True)

# Copy all dataset_*.csv files from dir_A
for file_path in source_dir.glob("dataset_*.csv"):
    destination = backup_dir / file_path.name
    if not destination.exists():
        shutil.copy(file_path, destination)
    else:
        tqdm.write(f"Skipped: {destination.name} already exists in {backup_dir}")

# Search one level of subdirectories for specific files
specific_files = ["tokenized_data.csv", "dictionary.json", "humanized_dictionary.txt"]
subdirs = [subdir for subdir in source_dir.iterdir() if subdir.is_dir()]
for subdir in tqdm(subdirs, desc="Subdirectories", unit="dir"):
    if not subdir.is_dir():
        continue
    
    backup_subdir = backup_dir / subdir.name
    
    # Copy the specific files to the backup directory
    for filename in specific_files:
        src = subdir / filename
        dst = backup_subdir / filename
        
        if not src.exists():
            tqdm.write(f"Note: {filename} not found in {subdir}")
        elif dst.exists():
            tqdm.write(f"Skipped: {dst.name} already exists in {backup_subdir}")
        else:
            backup_subdir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

print("Files backup up successfully.")