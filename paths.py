
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

CONFIG_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"
TRAINED_MODELS_DIR = PROJECT_DIR / "trained_models"

def project_relative_path(filepath: Path) -> str:
    """Return filepath relative to PROJECT_DIR"""
    return os.path.relpath(filepath.resolve(), PROJECT_DIR.resolve())

def as_csv(filename: Path | str):
    return Path(filename).with_suffix(".csv")

def as_json(filename: Path | str):
    return Path(filename).with_suffix(".json")

def as_bin(filename: Path | str):
    return Path(filename).with_suffix(".bin")


class _PathConstants:
    """
    Standardizes filenames across the project so they do not accidentally drift over time.
    - Avoids annoying errors due to incorrect filenames
    - Avoids needing to check filenames if writing new code after a while.
    - Makes changing names easy, just change here and applies across the project
    - The above assumes I use this consistently, idk.
    """
    
    # Sampling, decoding, filtering, and analysis
    samples_filename: str                       = "samples"
    samples_leading_particles_filename: str     = "samples_leading_particles" # @TODO: do away with this
    samples_decoded_filename: str               = "samples_decoded"
    samples_decoded_metadata_filename: str      = "samples_decoded_metadata"
    samples_decoded_invalid_filename: str       = "samples_decoded_invalid"
    samples_metrics_results_filename: str       = "metrics_results"
    
    real_test_decoded_filepath                  = 'real_test_untokenized_samples'
    real_test_decoded_invalid_events_filepath   = 'real_test_invalid_token_events'
    
    
    

    def __setattr__(self, name, value):
        """Blocks changing instance attributes."""
        raise AttributeError("Cannot modify read-only constants.")

    def __delattr__(self, name):
        """Blocks deleting instance attributes."""
        raise AttributeError("Cannot delete read-only constants.")
    
    
    # self.real_test_tokens_filepath                  = self.latest_sampling_dir / 'real_test_tokens.csv'
    #     self.real_test_untokenized_filepath             = self.latest_sampling_dir / 'real_test_untokenized_samples.csv'
    #     self.real_test_untokenizing_metadata_filepath   = self.latest_sampling_dir / 'real_test_untokenizing_metadata.json'
    #     self.real_test_invalid_tokens_filepath          = self.latest_sampling_dir / 'real_test_invalid_token_events.json'
    #     # self.generated_samples_filepath                 = Path(self.sampling_metadata['output_filepath'])
    #     # self.filtered_samples_filepath                  = self.latest_sampling_dir / 'filtered_samples.csv'
    #     # self.sampled_leading_particles_filepath         = self.latest_sampling_dir / 'sampled_leading_particles.csv'
    #     # self.verbose_particles_filepath                 = self.latest_sampling_dir / 'untokenized_samples_verbose.csv'
    #     # self.untokenized_samples_filepath               = self.latest_sampling_dir / 'untokenized_samples.csv'
    #     # self.untokenizing_metadata_filepath             = self.latest_sampling_dir / 'untokenized_samples_metadata.json'
    #     # self.invalid_tokens_filepath                    = self.latest_sampling_dir / 'invalid_token_events.json'
    #     # self.metrics_results_filepath                   = self.latest_sampling_dir / 'metrics_results.json'
    #     self.plotted_distributions_dir                  = self.latest_sampling_dir / 'plotted_distributions'

# Instantiate the class directly inside the module to create the singleton
path_constants = _PathConstants()