
import json
from pathlib import Path
import argparse
import warnings

import paths
from paths import path_constants as pc
import pUtil
from particleGPT.dictionary import Dictionary
from particleGPT.preparation import TokenizedMetadataConfig
# import particleGPT.untokenizer as untokenizer
from analysis.analyzer import Analyzer


def untokenize_generated_data(sampling_metadata_filepath: Path):
    # Load sampling metadata to get tokenized_metadata_file
    with sampling_metadata_filepath.open('r') as f:
        sampling_metadata = json.load(f)
    
    samples_filename = Path(sampling_metadata['output_filepath'])
    if not samples_filename.exists():
        raise FileNotFoundError("Generated samples.csv needs to exist before they can be untokenized!")
    
    tokenized_metadata_filepath = Path(sampling_metadata['tokenized_metadata_filepath'])
    tmd_conf = TokenizedMetadataConfig(tokenized_metadata_filepath)
    dictionary = Dictionary(tmd_conf.dictionary_filepath)
    
    samples_decoded_filepath = sampling_metadata_filepath.parent / paths.as_bin(pc.samples_decoded_filename)
    tokenizer = tmd_conf.tokenizer_class(dictionary=dictionary, temp_dir=None)
    tokenizer.decode_dataset_from_file(samples_filename)
    tokenizer.save_data(samples_decoded_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handles the analysis of generated particle data. Assumes the model has been sampled already.")
    parser.add_argument("metadata_file", type=Path)
    parser.add_argument("--no-metrics", action="store_true", help="If provided, skip metric calculations.")
    parser.add_argument("--no-distributions", action="store_true", help="If provided, skip distribution generation.")
    parser.add_argument("--no-untokenize", action="store_true", help="If provided, skip untokenization of generated data. Assumes the data is already untokenized.")
    args = parser.parse_args()
    
    sampling_metadata_filepath = Path(args.metadata_file)
    if not sampling_metadata_filepath.exists():
        raise FileNotFoundError(f"Samples metadata file not found at {sampling_metadata_filepath}. The metadata file is required to perform analysis.")
    
    if args.no_untokenize:
        warnings.warn(
            "flag --no-untokenize is set. Will skip untokenize distribution. "
            "This is fail horribly if the data is not already untokenized!",
            RuntimeWarning
        )
        print("Skipping untokenization of generated data.")
    if args.no_distributions:
        print("Skipping distribution generation.")
    if args.no_metrics:
        print("Skipping metric calculations.")
    
    print(f'Analyzing sampled data at {sampling_metadata_filepath.parent}')

    # Untokenize data
    if not args.no_untokenize:
        print("Untokenizing data")
        untokenize_generated_data(sampling_metadata_filepath)
        
    dataset_analyzer = Analyzer(sampling_metadata_filepath)
    
    # Generate distributions
    if not args.no_distributions:
        print("Generating distributions")
        dataset_analyzer.generate_distributions()
        
    # Calculate metrics
    if not args.no_metrics:
        print("Calculating metrics")
        dataset_analyzer.calculate_metrics()

    print('Analysis finished successfully.')
