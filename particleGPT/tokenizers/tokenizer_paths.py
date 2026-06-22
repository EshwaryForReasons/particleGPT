
from pathlib import Path
from dataclasses import dataclass

@dataclass
class TokenizerPaths:
    # Input raw data
    input_data_filepath: Path = None
    # Output tokenized data
    tokenized_data_filepath: Path = None
    tokenized_lens_filepath: Path = None
    # Directory to store temp tokenized files before concatenation
    temp_data_dir: Path = None
    dictionary_filepath: Path = None