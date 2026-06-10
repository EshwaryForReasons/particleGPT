# `particleGPT/tokenizer.py`

## Overview

`tokenizer.py` converts raw particle collision events — structured as arrays of per-particle feature values — into flat sequences of integer tokens that the GPT model can consume.

The tokenizer relies on a `Dictionary` object to know:
- Which features to encode (from `tokenization_schema`)
- What bin edges to use for each continuous feature
- What the integer offset is for each token type

The output is a binary (`.bin`) memory-mapped file of integer tokens, plus a `tokenized_metadata.json` sidecar file that records the vocabulary size, sequence length, total token count, and data dtype.

One tokenized sequence represents one particle collision event:

```
event_start  [particle_start PDGID material E eta phi pt particle_end] × N_particles  event_end  [padding…]
```

All sequences are padded to the same length (`sequence_length`) so the dataset is a regular 2-D array on disk.

---

## Quick Start

```bash
# Tokenize a raw dataset using a model config
python -m particleGPT.tokenizer config/models/my_model.json

# Output:
# data/tokenized/<tokenization_name>/tokenized_data.bin
# data/tokenized/<tokenization_name>/tokenized_metadata.json
```

---

## Key Functions

### `tokenize(dictionary: Dictionary, input_data_filepath: Path, output_dir: Path) → None`

Main tokenization function. Reads the raw dataset, encodes each event, pads to `sequence_length`, and writes the output binary.

**Process:**
1. Loads the raw dataset using `data_manager.load_geant4_dataset()`
2. For each event, encodes each particle's features according to `dictionary.tokenization_schema`
3. Wraps each particle with `particle_start` / `particle_end` tokens
4. Wraps the full event with `event_start` / `event_end` tokens
5. Pads with `get_padding_sequence()` tokens until the sequence reaches `sequence_length`
6. Writes all sequences as a flat `uint16` (or larger if vocab requires) binary file
7. Writes `tokenized_metadata.json` alongside

**Parameters**

| Param | Type | Description |
|---|---|---|
| `dictionary` | `Dictionary` | Loaded vocabulary/bins object |
| `input_data_filepath` | `Path` | Path to the raw Geant4 CSV |
| `output_dir` | `Path` | Directory to write `.bin` and `.json` |

---

### `encode_particle(dictionary: Dictionary, particle: np.ndarray) → list[int]`

Encodes a single particle's feature array into a list of token IDs. Iterates over `dictionary.tokenization_schema` and calls the appropriate digitization function for each feature type.

**Feature encoding:**
- `"pdgid"`: looks up PDGID in `dictionary.pdgids_to_index`, adds `PDGID_OFFSET`
- `"material"`: looks up material index, adds `MATERIAL_OFFSET`
- `"energy"`: `np.digitize(value, e_bins) + ENERGY_OFFSET`
- `"eta"`, `"theta"`, `"phi"`, `"pt"`, `"px"`, `"py"`, `"pz"`: same pattern with respective bins and offsets

Out-of-range values are clipped to the nearest valid bin.

**Parameters**

| Param | Type | Description |
|---|---|---|
| `dictionary` | `Dictionary` | Vocabulary object |
| `particle` | `np.ndarray` | 1-D array of feature values; length matches `num_tokens_per_particle` |

**Returns**: `list[int]` of token IDs, length = `num_tokens_per_particle`

---

### `encode_event(dictionary: Dictionary, event: list[np.ndarray]) → list[int]`

Encodes a full event (list of particles) into a token sequence, including special tokens and padding.

```
[event_start, particle_start, tok1, tok2, …, particle_end, particle_start, …, event_end, pad, pad, …]
```

**Parameters**

| Param | Type | Description |
|---|---|---|
| `dictionary` | `Dictionary` | Vocabulary object |
| `event` | `list[np.ndarray]` | List of particle feature arrays |

**Returns**: `list[int]` of length `sequence_length`

---

### `compute_sequence_length(dictionary: Dictionary, max_particles: int) → int`

Computes the padded sequence length needed for events with up to `max_particles` particles:

```
sequence_length = 1 (event_start)
               + max_particles × (1 + num_tokens_per_particle + 1)  [start + features + end]
               + 1 (event_end)
```

Padding fills the remainder.

---

### `write_metadata(output_dir, vocab_size, sequence_length, total_sequences, total_tokens, dtype, data_filepath) → None`

Writes `tokenized_metadata.json` to `output_dir`. This file is required by both `train.py` and `sample.py` to locate and interpret the binary.

---

## Tokenized Data Format

The output binary is a flat sequence of integers stored in dtype `uint16` (for vocab sizes ≤ 65535) or `uint32` otherwise:

```
[event_0_tok_0, event_0_tok_1, ..., event_0_tok_{L-1},
 event_1_tok_0, ...,
 ...
 event_{N-1}_tok_{L-1}]
```

Total file size = `N_events × sequence_length × dtype_bytes`.

The corresponding metadata JSON:

```json
{
  "dtype": "uint16",
  "vocab_size": 512,
  "sequence_length": 256,
  "total_sequences": 1000000,
  "total_tokens": 256000000,
  "num_full_sequences": 1000000,
  "tokenized_data_file": "data/tokenized/my_tokenization/tokenized_data.bin"
}
```

---

## Gotchas

- **Out-of-range values are silently clipped**. If a feature value falls outside the bin range defined in the dictionary (e.g. energy exceeds `e_bin_data.max`), it is mapped to the nearest edge bin. This is intentional for robustness, but check your distribution plots to confirm no large fraction of values is being clipped.

- **Unknown PDGIDs default to index 0**. If a PDGID appears in the raw data but not in `dictionary.pdgids`, it is mapped to the zero slot. Index 0 is typically a placeholder (pdgid=0 in the dictionary). Review `dictionary.output_humanized_dictionary()` output to verify coverage.

- **Quantile bin computation is slow**. On first run, quantile bins scan the entire dataset. Subsequent runs use the cached `.npy` file. If you change the dataset, delete the cache files before re-running.

- **`sequence_length` is fixed at tokenization time**. All events are padded or truncated to this length. Events with more particles than `max_particles` will be truncated — the excess particles are silently dropped. Monitor truncation rates in your raw data.

- **dtype is chosen automatically** based on `vocab_size`: `uint16` for ≤ 65535, `uint32` otherwise. `train.py` reads this from the metadata and uses the matching dtype for the memmap.

---

## Related

| Module | How it connects |
|---|---|
| `dictionary.py` | Provides all bins, offsets, and schema for encoding |
| `data_manager.py` | Loads the raw Geant4 CSV that gets tokenized |
| `particleGPT/untokenizer.py` | The inverse operation |
| `train.py` | Reads the `.bin` file via `TokenBlockDataset` |
| `sample.py` | Reads the `.bin` file for test starters |
