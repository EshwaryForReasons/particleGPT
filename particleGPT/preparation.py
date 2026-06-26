
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
    
class TokenizedMetadataConfig():
    """
    These represent properties of the entire tokenized dataset.
    """
    dtype:                   np.dtype | None = None
    sequence_length:         int | None = None
    vocab_size:              int | None = None
    total_sequences:         int | None = None
    total_tokens:            int | None = None
    num_full_sequences:      int | None = None
    tokenized_data_filepath: Path | None = None
    dictionary_filepath:     Path | None = None
    
    tokenizer_class:         str | None = None
    
    def __init__(self, tokenized_metadata_filepath: Path) -> None:
        if not tokenized_metadata_filepath.exists():
            raise FileNotFoundError("tokenized_metadata_filepath does not exist. Please make sure the provided file exists!")
        
        try:
            with open(tokenized_metadata_filepath, "r") as f:
                tokenized_mdata_json = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failure while trying to load json! Exception:\n{exc}") from exc
        
        dtype_str = tokenized_mdata_json["tokenized_dtype"]
        self.dtype = np.dtype(dtype_str).type
        self.vocab_size = int(tokenized_mdata_json["vocab_size"])
        self.sequence_length = int(tokenized_mdata_json["sequence_length"])
        self.total_sequences = int(tokenized_mdata_json["total_sequences"])
        self.total_tokens = int(tokenized_mdata_json["total_tokens"])
        self.num_full_sequences = int(tokenized_mdata_json["num_full_sequences"])
        self.dictionary_filepath = Path(tokenized_mdata_json["dictionary_filepath"])
        self.tokenized_data_filepath = Path(tokenized_mdata_json["output_data_filepath"])
        
        self.tokenizer_class = str(tokenized_mdata_json["tokenizer_class"])
        
        # ===== Make sure all derived values are reasonable (i.e. not None, out-of-bounds, etc.) =====
        
        if not np.issubdtype(np.dtype(self.dtype), np.integer):
            raise TypeError(f"Tokenizer metadata specifies dtype={self.dtype}, which is not an integer dtype!")
        if self.vocab_size is None or self.vocab_size <= 0:
            raise RuntimeError("tokenizer metadata: vocab size cannot be none or less than zero!")
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
        if not self.dictionary_filepath.exists():
            raise FileNotFoundError("dictionary_filepath does not exist. Please make sure the provided file exists!")
        if not self.tokenized_data_filepath.exists():
            raise FileNotFoundError("tokenized_data_filepath does not exist. Please make sure the provided file exists!")
        
        if self.tokenizer_class is None:
            raise RuntimeError("tokenizer metadata: tokenizer class cannot be none!")
        
        # ===== Verify file consistency =====
            
        # Ensure configured datatype makes sense
        file_bytes = self.tokenized_data_filepath.stat().st_size
        dtype_bytes = np.dtype(self.dtype).itemsize
        if file_bytes % dtype_bytes != 0:
            raise ValueError(
                f"Size of tokenized data file ({file_bytes} bytes) is not divisible by the size of the dtype {self.dtype} "
                f"({dtype_bytes} bytes). The file may be corrupted or the metadata may be wrong."
            )
        
        # Fancy num tokens calculation and checks
        data_total_tokens = file_bytes // dtype_bytes
        if data_total_tokens != self.total_tokens:
            raise ValueError(
                f"Tokenized data contains {data_total_tokens:,} tokens, but expected {self.total_tokens:,} "
                "based on the tokenized metadata. Regenerate the prepared data or fix the metadata."
            )
        
        # Sequence length check
        if data_total_tokens % self.sequence_length != 0:
            raise ValueError(
                f"Tokenized data contains {data_total_tokens:,} tokens, which is not divisible by the sequence length "
                f"{self.sequence_length}. Regenerate the prepared data or fix the metadata."
            )
        
        # Sequence count check
        data_total_sequences = data_total_tokens // self.sequence_length
        if data_total_sequences != self.num_full_sequences:
            raise ValueError(
                f"Tokenized data contains {data_total_sequences:,} full sequences, but expected {self.num_full_sequences:,} "
                "based on the tokenized metadata. Regenerate the prepared data or fix the metadata."
            )

class DataloaderSplitConfig():
    """Represents one bin/split's config from the preparation config"""
    split_type:                  ESplitTypes = ESplitTypes.NONE
    num_sequences:               int | None = None
    skip_sequences:              int | None = None
    from_end:                    bool | None = None
    tmd_conf:                    TokenizedMetadataConfig | None = None
    
    tokenized_metadata_filepath: Path
    
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
        
        if self.num_sequences is None or self.num_sequences <= 0:
            raise RuntimeError("preparation config: num sequences cannot be none or less than zero!")
        if self.skip_sequences is None or self.skip_sequences < 0:
            raise RuntimeError("preparation config: skip sequences cannot be none or less than zero!")
        if self.from_end is None or not isinstance(self.from_end, bool):
            raise RuntimeError("preparation config: from end must be a bool!")
        
        self.tokenized_metadata_filepath = Path(prep_conf_json['tokenized_metadata_file'])
        if self.tokenized_metadata_filepath is None or not self.tokenized_metadata_filepath.exists():
            raise RuntimeError("preparation config: tokenized metadata file must exist!")
        self.tmd_conf = TokenizedMetadataConfig(self.tokenized_metadata_filepath)
        
        """
        Calculate raw (before block_size adjustment) start and end token indices and verify the dataset can provide it.
        This is for verification for the config.
        Example:
            total_tokens = 100
            sequence_length = 10
            skip_sequences = 8      # start at token 80
            num_sequences = 3       # raw request is tokens 80:110, invalid
            block_size = 19
        Raw split asks for 30 tokens, but usable split becomes 20 tokens. This makes the final range
        80:100, which is valid, but the config is misleading because it implies 30 tokens when only 20 are usable.
        The following check catches this style of issue.
        """
        raw_split_tokens = self.num_sequences * self.tmd_conf.sequence_length
        if self.from_end:
            raw_end = (self.tmd_conf.num_full_sequences - self.skip_sequences) * self.tmd_conf.sequence_length
            raw_start = raw_end - raw_split_tokens
        else:
            raw_start = self.skip_sequences * self.tmd_conf.sequence_length
            raw_end = raw_start + raw_split_tokens
        
        if raw_start < 0 or raw_end > self.tmd_conf.total_tokens or raw_start >= raw_end:
            raise ValueError(
                f"Invalid raw split range: raw_start={raw_start}, raw_end={raw_end}, "
                f"total_tokens={self.tmd_conf.total_tokens}."
            )
        
        self.raw_end = raw_end
        self.raw_start = raw_start

    def get_raw_tokens_range(self) -> tuple[int, int]:
        return (self.raw_start, self.raw_end)
