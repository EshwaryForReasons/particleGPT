
import json
import re
from pathlib import Path
import paths
import particleGPT.configurator as conf
from particleGPT.preparation import ESplitTypes, DataloaderSplitConfig, TokenizedMetadataConfig
from particleGPT.dictionary import Dictionary
from pydantic import validate_call

from particleGPT.tokenizers import (
    EventPerSequenceParticleFeatureTokenizer,
    PackedEventStreamParticleFeatureTokenizer,
)

script_dir = Path(__file__).resolve().parent

CONFIG_DIR_NAME = 'config'
GENERATED_SAMPLES_DIR_NAME = 'generated_samples'
TRAINED_MODELS_DIR_NAME = 'trained_models'
DATASETS_DIR_NAME = 'data'
PREPARATIONS_DIR_NAME = 'preparations'
TEMP_DIR_NAME = 'temp'


def get_dictionary(preparation_config_filepath: Path | str) -> Dictionary:
    """
    Load the dictionary for the current configured tokenized dataset.

    The dictionary path is resolved from tokenized metadata, not from the old
    preparation directory.
    """
    preparation_config_filepath = paths.PROJECT_DIR / Path(preparation_config_filepath)
    dls_conf = DataloaderSplitConfig(ESplitTypes.TEST, preparation_config_filepath)
    dictionary = Dictionary(dls_conf.tmd_conf.dictionary_filepath)
    return dictionary


def tokenizer_class_from_str(tokenizer_class: str):
    selected = None
    match tokenizer_class:
        case "EventPerSequenceParticleFeatureTokenizer":
            selected = EventPerSequenceParticleFeatureTokenizer
        case "PackedEventStreamParticleFeatureTokenizer":
            selected = PackedEventStreamParticleFeatureTokenizer
        case __:
            raise NotImplementedError("This tokenizer class is not supported!")
    return selected
    

def get_training_dir(model_name):
    return script_dir / TRAINED_MODELS_DIR_NAME / model_name


def get_model_config_filepath(model_name):
    # The model name will be the name of the config file unless specified otherwise within the file.
    config_dir = script_dir / CONFIG_DIR_NAME
    config_files = list(config_dir.rglob("*.json"))
    
    correct_config_file = None
    for config_file in config_files:
        with open(config_file, "r") as f:
            data = json.load(f)
            if config_file.stem == model_name:
                correct_config_file = config_file
                break
            elif 'model_name' in data and data['model_name'] == model_name:
                correct_config_file = config_file
                break
    
    if correct_config_file is None:
        raise ValueError(f"No config file found for model name {model_name}.")
    
    return correct_config_file
                    

@validate_call
def get_latest_sampling_dir(model_name: str, sampling_idx_override: int | None = None) -> Path:
    """
    Return the selected sampling directory for the current sampling pipeline.
    
    The sampler writes generated samples to:
        PROJECT_DIR/generated_samples/<model_name>/sampling_<idx>/
        
    untokenizer.resolve_sampling_idx handles explicit sample_idx config values
    and otherwise chooses the newest sampling_<idx> directory.
    """
    generated_samples_dir = paths.PROJECT_DIR / 'generated_samples' / model_name
    if not generated_samples_dir.exists():
        raise FileNotFoundError(f"Generated samples directory does not exist: {generated_samples_dir}")
    
    if sampling_idx_override is not None:
        sampling_idx = sampling_idx_override
    else:
        max_idx = -1
        for path in generated_samples_dir.iterdir():
            if not path.is_dir():
                continue
            match = re.fullmatch(r"sampling_(\d+)", path.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        
        if max_idx < 0:
            raise FileNotFoundError(f"No sampling_N directories found in {generated_samples_dir}.")
        
        sampling_idx = max_idx

    return generated_samples_dir / f'sampling_{sampling_idx}'