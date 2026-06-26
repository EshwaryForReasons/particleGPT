
"""
Run from the main project directory with:

python tokenize.py path/to/dictionary.json
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import paths as paths
from particleGPT.dictionary import Dictionary
from particleGPT.tokenizers import (
    BaseTokenizer,
    EventPerSequenceParticleFeatureTokenizer,
    PackedEventStreamParticleFeatureTokenizer,
)

def main():
    parser = argparse.ArgumentParser(description="Tokenizes dataset provided a dictionary file.")
    parser.add_argument("dictionary_path", type=Path)
    args = parser.parse_args()
    
    if args.dictionary_path is None:
        raise ValueError("Dictionary path is required.")
    
    dictionary = Dictionary(args.dictionary_path)
    humanized_dictionary_filepath = paths.PROJECT_DIR / 'data' / 'tokenized' / dictionary.tokenization_name / 'humanized_dictionary.md'
    # dictionary.update_dictionary_particle_list(relevant_paths.input_data_filepath, relevant_paths.dictionary_filepath)
    dictionary.output_humanized_dictionary(humanized_dictionary_filepath)
    
    tokenizer_class = globals()[dictionary.tokenizer_class_str]
    if tokenizer_class is None:
        raise RuntimeError("tokenizer_class cannot be none!")
    
    temp_dir = paths.PROJECT_DIR / 'data' / 'tokenized' / dictionary.tokenization_name / 'temp'
    raw_data_filepath = paths.PROJECT_DIR / 'data' / 'raw' / dictionary.dataset_name
    tokenized_data_filepath = paths.PROJECT_DIR / 'data' / 'tokenized' / dictionary.tokenization_name / 'tokenized_data.bin'
    
    tokenizer = tokenizer_class(dictionary, temp_dir=temp_dir, dtype=np.uint16)
    tokenizer.encode_dataset(raw_data_filepath)
    tokenizer.postprocess_data()
    tokenizer.save_data(tokenized_data_filepath)
    
if __name__ == "__main__":
    main()