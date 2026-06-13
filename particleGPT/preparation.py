
from __future__ import annotations

import json
from pathlib import Path
from enum import Enum

import numpy as np
import particleGPT.configurator as conf

class ESplitTypes(Enum):
    NONE        = 0,
    TRAIN       = 1,
    VALIDATION  = 2,
    TEST        = 3

class DataloaderSplitConfig():
    """Represents one bin's config from the preparation config"""
    split_type:                  ESplitTypes = ESplitTypes.NONE
    num_sequences:               int | None = None
    skip_sequences:              int | None = None
    from_end:                    bool | None = None
    tokenized_metadata_filepath: Path | None = None
    
    split_config_key_map = {
        ESplitTypes.NONE: "",
        ESplitTypes.TRAIN: "train_bin",
        ESplitTypes.VALIDATION: "validation_bin",
        ESplitTypes.TEST: "test_bin",
    }
    
    def __init__(self, split_type: ESplitTypes, preparation_config_filepath: Path):
        if not preparation_config_filepath.exists():
            raise FileNotFoundError("preparation_config_file does not exist. Please make sure the provided file exists!")
        
        self.split_type = split_type
        
        try:
            with open(preparation_config_filepath, "r") as f:
                prep_conf_json = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failure while trying to load json! Exception:\n{exc}") from exc
        
        # Create DataloaderSplitConfig for this split
        split_str = self.split_config_key_map[self.split_type]
        self.num_sequences = int(prep_conf_json[split_str]['num_sequences'])
        self.skip_sequences = int(prep_conf_json[split_str]['skip_sequences'])
        self.from_end = prep_conf_json[split_str]['from_end']
        self.tokenized_metadata_filepath = Path(prep_conf_json['tokenized_metadata_file'])
        
        if self.num_sequences is None or self.num_sequences <= 0:
            raise RuntimeError("preparation config: num sequences cannot be none or less than zero!")
        if self.skip_sequences is None or self.skip_sequences < 0:
            raise RuntimeError("preparation config: skip sequences cannot be none or less than zero!")
        if not isinstance(self.from_end, bool):
            raise RuntimeError("preparation config: from end must be a bool!")
    
    def verify(self) -> bool:
        """Ensures no members are None. No members should ever be None"""
        return (self.num_sequences is not None 
            and self.skip_sequences is not None
            and self.from_end is not None
            and self.tokenized_metadata_filepath is not None)

class TokenizedMetadataConfig():
    """
    These represent properties of the entire tokenized dataset.
    While all are loaded, only some "meta" quantities like sequence_length and dtype are useful.
    """
    dtype:                   np.dtype | None = None
    sequence_length:         int | None = None
    vocab_size:              int | None = None
    total_sequences:         int | None = None
    total_tokens:            int | None = None
    num_full_sequences:      int | None = None
    tokenized_data_filepath: Path | None = None 
    
    def __init__(self, tokenized_metadata_filepath: Path) -> None:
        if not tokenized_metadata_filepath.exists():
            raise FileNotFoundError("tokenized_metadata_filepath does not exist. Please make sure the provided file exists!")
        
        try:
            with open(tokenized_metadata_filepath, "r") as f:
                tokenized_mdata_json = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failure while trying to load json! Exception:\n{exc}") from exc
        
        dtype_str = tokenized_mdata_json["dtype"]
        self.dtype = np.dtype(dtype_str).type
        self.vocab_size = int(tokenized_mdata_json["vocab_size"])
        self.sequence_length = int(tokenized_mdata_json["sequence_length"])
        self.total_sequences = int(tokenized_mdata_json["total_sequences"])
        self.total_tokens = int(tokenized_mdata_json["total_tokens"])
        self.num_full_sequences = int(tokenized_mdata_json["num_full_sequences"])
        self.tokenized_data_filepath = Path(tokenized_mdata_json["tokenized_data_file"])
        
        if not np.issubdtype(np.dtype(self.dtype), np.integer):
            raise TypeError(f"Tokenizer metadata specifies dtype={self.dtype}, which is not an integer dtype!")
        if self.vocab_size > np.iinfo(self.dtype).max + 1:
            raise RuntimeError(
                f"Tokenizer metadata specifies vocab_size={self.vocab_size}, which exceeds the capacity of the dtype {self.dtype} "
                f"(max value {np.iinfo(self.dtype).max}). Reduce the vocab size or use a larger dtype."
            )
        if self.sequence_length is None or self.sequence_length <= 0:
            raise RuntimeError("tokenizer metadata: sequence length cannot be none or less than zero!")
        if self.total_sequences is None or self.total_sequences < 0:
            raise RuntimeError("tokenizer metadata: total sequences cannot be none or less than zero!")
        if self.total_tokens is None or self.total_tokens < 0:
            raise RuntimeError("tokenizer metadata: total tokens cannot be none or less than zero!")
        if self.num_full_sequences is None or self.num_full_sequences < 0:
            raise RuntimeError("tokenizer metadata: num full sequences cannot be none or less than zero!")
        if not self.tokenized_data_filepath.exists():
            raise FileNotFoundError("tokenized_data_filepath does not exist. Please make sure the provided file exists!")
        
    def verify(self) -> bool:
        """Ensures no members are None. No members should ever be None"""
        return (self.dtype is not None 
            and self.sequence_length is not None
            and self.vocab_size is not None
            and self.total_sequences is not None
            and self.total_tokens is not None
            and self.num_full_sequences is not None
            and self.tokenized_data_filepath is not None)
