import numpy as np
import configurator as config

script_dir = Path(__file__).resolve().parent

# This needs to account for the split as well, I think?
def get_prob_dist(feature, split: str):
    assert split in ['train', 'val'], "Split must be 'train' or 'val'!"
    
    # Read the data for that feature (eta, phi, etc.)
    tokenized_data_bin_filepath = script_dir / 'data' / config.generic.preparation_name / f'{split}.bin'
    tokenized_data = np.memmap(tokenized_data_bin_filepath, dtype=np.float32, mode='r')