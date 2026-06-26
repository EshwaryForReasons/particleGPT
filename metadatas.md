
# Metadata Files

This file provides a reference for what each metadata file from various components of particleGPT contain.

## Base rules

- All file paths will be relative to the project directory.
- All file paths' keys will be called `*_filepath`.
- All file names' keys will be called `*_filename`.
- All dir path keys will be called `*_dir`.

## Metadatas

#### Sampling metadata

Metadata generated for sampling datasets will be of the following format
```JSON
{
    "checkpoint_filepath": "trained_models/model_test_2/ckpt.pt",
    "output_dir": "generated_samples/model_test_2/sampling_0",
    "output_filepath": "generated_samples/model_test_2/sampling_0/samples.csv",
    "output_metadata_filepath": "generated_samples/model_test_2/sampling_0/sampling_metadata.json",
    "sampling_elapsed_seconds": 114.03914737701416,
    "model_name": "model_test_2",
    "sample_idx": 0,
    "num_gpus_used": 1,
    "num_devices_used": 1
    "device_names": [
        "cuda:0"
    ],
    "batch_size": 128,
    "num_starter_tokens": 5,
    "max_new_tokens": 500,
    "temperature": 0.8,
    "top_k": 200,
    "seed": 1337,
    "compile": true,
    "require_batch_generate": false,
    "tokenized_data_filepath": "data/tokenized/tdataset_full_50M_r1_v2/tokenized_data.bin",
    "tokenized_metadata_filepath": "data/tokenized/tdataset_full_50M_r1_v2/tokenized_data.bin.json",
    "tokenized_dtype": "uint16",
    "split": "test_bin",
    "sequence_length": 1024,
    "split_start_token_idx": 3231712256,
    "num_sample_sequences": 1000000
}
```

#### Tokenized metadata

Metadata generated for tokenzied datasets will be of the following format
```JSON
{
    "tokenizer_class": "EventPerSequenceParticleFeatureTokenizer",
    "tokenization_schema": [
        "pdgid",
        "pt",
        "eta",
        "phi"
    ],
    "tokenized_dtype": "uint16",
    "byte_order": "little",
    "vocab_size": 244,
    "sequence_length": 170,
    "total_sequences": 49606592,
    "total_events": 49606592,
    "total_tokens": 8433120640,
    "num_full_sequences": 49606592,
    "remainder_tokens": 0,
    "input_data_filepath": "data/raw/rdataset_full_50M.csv",
    "output_data_filepath": "data/tokenized/tdataset_full_50M_r1_v1/tokenized_data.bin",
    "dictionary_filepath": "data/tokenized/tdataset_full_50M_r1_v1/dictionary.json"
}
```

#### Untokenzied metadata

Metadata generated for untokenized datasets will be of the following format
```JSON
{
    "input_data_filepath": paths.project_relative_path(self.input_data_filepath),
    "untokenized_samples_filepath": paths.project_relative_path(self.output_data_filepath),
    "tokenization_schema": self.dictionary.tokenization_schema,
    # "float_precision": self.float_precision, @TODO: implement this
    "total_events_written": self.total_events_written,
    "total_tokens_written": self.total_tokens_written,
    "total_empty_samples": self.total_empty_samples,
    "total_invalid_events": len(self.failed_events),
}
```
