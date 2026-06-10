# `dictionary.py`

## Overview

`dictionary.py` is the **vocabulary layer** of particleGPT. It defines how continuous physics quantities (energy, η, φ, pT, …) and discrete quantities (PDGID, material) are mapped to integer tokens, and how those tokens are mapped back to values.

The central class is `Dictionary`, which loads a `dictionary.json` config file and exposes:
- Bin edge arrays for each physical quantity
- Token offset constants for each token type (so every token type occupies a unique non-overlapping integer range)
- Helper methods for encoding, decoding, and humanizing the vocabulary
- Three binning strategies: **linear**, **Gaussian**, and **quantile**

The vocabulary is structured as a flat integer range:

```
[0, num_special)         → special tokens (padding, event_start, event_end, …)
[num_special, +num_pdg)  → particle PDGIDs
[+num_pdg, +num_mat)     → detector materials
[+num_mat, +len(e_bins)) → energy bins
[...and so on for eta, theta, phi, pt, px, py, pz]
```

---

## Quick Start

```python
from dictionary import Dictionary

d = Dictionary("config/models/my_dictionary.json")

# Inspect the vocabulary
print(f"Vocab size: {d.vocab_size}")
print(f"Energy token range: {d.ENERGY_OFFSET} – {d.ENERGY_OFFSET + len(d.e_bins) - 1}")

# Get token type from a token value
from dictionary import ETokenTypes
token_type = d.get_token_type(153)          # e.g. ETokenTypes.ENERGY

# Write a human-readable summary
d.output_humanized_dictionary("dictionary_summary.txt")
```

---

## Enum: `ETokenTypes`

```python
class ETokenTypes(Enum):
    PADDING  = 0
    SPECIAL  = 1
    PDGID    = 2
    MATERIAL = 3
    ENERGY   = 4
    ETA      = 5
    THETA    = 6
    PHI      = 7
    PT       = 8
    PX       = 9
    PY       = 10
    PZ       = 11
```

Used as the return type of `Dictionary.get_token_type()`.

---

## Binning Functions

### `arange(start, stop, step_size) → np.ndarray`

A custom `np.arange` that avoids floating-point accumulation errors by rounding to the appropriate number of decimal places at each step. Use instead of `np.arange` when step precision matters (e.g. `step_size=0.001`).

### `linear_space(start, stop, n_bins) → np.ndarray`

Like `np.linspace` but using the same precision-safe stepping logic as `arange`. Creates `n_bins` uniformly-spaced bin edges.

### `gaussian_space(start, stop, num, center, sigma=1.0) → np.ndarray`

Generates non-uniform bin edges with highest density near `center`. Internally builds a fine grid, fits a Gaussian PDF, inverts the CDF, and samples `num` points from it.

**Parameters**

| Param | Type | Description |
|---|---|---|
| `start` | `float` | Left edge of range |
| `stop` | `float` | Right edge of range |
| `num` | `int` | Number of bins |
| `center` | `float` | Location of highest bin density |
| `sigma` | `float` | Width of the Gaussian peak |

Useful when most values cluster around a known point (e.g. most η values near 0).

### `detokenize_gaussian_bins(tokens, bin_edges) → np.ndarray`

Maps token indices back to approximate physical values using bin center interpolation.

**Parameters**

| Param | Type | Description |
|---|---|---|
| `tokens` | `np.ndarray` | Integer token indices (0-based) |
| `bin_edges` | `np.ndarray` | Bin edges used at tokenization time |

### `build_gaussian_bin_means(values, bin_edges) → np.ndarray`

A higher-fidelity variant of the above: computes the mean of all training values that fall in each bin, rather than using the bin center. Falls back to bin center for empty bins.

### `quantile_detokenize(tokens, bin_edges, token_min=0) → np.ndarray`

Converts quantile-tokenized indices back to physical values via bin-center interpolation.

### `build_quantile_bin_means(values, bin_edges) → np.ndarray`

Same as `build_gaussian_bin_means` but for quantile-spaced bins.

### `truncate_quantiles(quantile_tokens, original_bin_size, truncated_bin_size) → np.ndarray`

Re-bins quantile tokens from a finer grid to a coarser one. Useful when you want to reduce the vocabulary size for a specific feature after tokenization.

### `get_all_of(dictionary, input_data_filepath, type_str) → list`

Reads the entire raw dataset and collects all values for a given feature type (e.g. `"energy"`). Used internally to compute quantile bin edges on first run.

---

## Class: `Dictionary`

```python
class Dictionary:
    def __init__(self, dictionary_filename: str)
```

### Constructor

Loads `dictionary_filename` (a JSON file), validates required fields, builds bin arrays, and computes all token offsets.

**Sets on construction:**

| Attribute | Description |
|---|---|
| `dictionary_data` | Raw dict from the JSON |
| `dataset_name` | Dataset filename |
| `dataset_filepath` | Resolved path to `data/raw/<dataset>` |
| `tokenization_name` | Name of the tokenization scheme |
| `num_special_tokens` | Count of special tokens |
| `num_particles` | Count of unique PDGIDs |
| `num_materials` | Count of detector materials |
| `tokenization_schema` | List of feature types per position in a particle sequence |
| `padding_sequence` | Token sequence used for padding a particle slot |
| `num_tokens_per_particle` | Length of one particle's token sequence |
| `e_bins`, `eta_bins`, `theta_bins`, `phi_bins`, `pt_bins`, `px_bins`, `py_bins`, `pz_bins` | Bin edge arrays for each feature |
| `vocab_size` | Total vocabulary size |
| `SPECIAL_TOKENS_OFFSET` | Always 0 |
| `PDGID_OFFSET` | Start of PDGID token range |
| `MATERIAL_OFFSET` | Start of material token range |
| `ENERGY_OFFSET` | Start of energy bin range |
| `ETA_OFFSET`, `THETA_OFFSET`, `PHI_OFFSET`, `PT_OFFSET`, `PX_OFFSET`, `PY_OFFSET`, `PZ_OFFSET` | Starts of other feature ranges |
| `pdgids` | `{index: pdgid}` mapping from the JSON |
| `table_data` | Pre-built table rows for `output_humanized_dictionary` |

---

### Properties

| Property | Type | Description |
|---|---|---|
| `padding_token` | `int` | Integer value of the padding token |
| `event_start_token` | `int` | Integer value of `event_start` |
| `event_end_token` | `int` | Integer value of `event_end` |
| `particle_start_token` | `int` | Integer value of `particle_start` |
| `particle_end_token` | `int` | Integer value of `particle_end` |
| `scheme` | `str` | Name of the tokenization scheme (from JSON) |
| `pdgids_to_index` | `dict` | Reversed `pdgids` map: `{pdgid: index}` |

---

### `_create_bins(type_str: str) → np.ndarray`

Internal factory that reads the `<type_str>_bin_data` section of the JSON and creates the bin array using the correct spacing function.

Dispatches on `tokenization_function`:
- `"linear"` → `arange` (if `step_size` present) or `linear_space` (if `n_bins` present)
- `"gaussian"` → `gaussian_space` with `gaussian_center` and `gaussian_sigma`
- `"quantile"` → loads cached `.npy` file if present, otherwise reads all data and computes quantiles, then caches

Returns `np.array([], dtype=np.float64)` if the feature is not in the dictionary.

---

### `get_padding_sequence() → list[int]`

Returns the sequence of token IDs used to represent a padding particle slot. Each element is resolved from the `padding` dict in the JSON (`"padding"` → `padding_token`, `"particle_start"` → `particle_start_token`, etc.).

---

### `get_token_type(token: int) → ETokenTypes`

Returns the type of a given token ID by checking which offset range it falls in.

```python
d.get_token_type(0)     # → ETokenTypes.PADDING
d.get_token_type(5)     # → ETokenTypes.PDGID (if PDGID_OFFSET=3, num_particles=10)
d.get_token_type(250)   # → ETokenTypes.ENERGY (if ENERGY_OFFSET=250, …)
```

---

### `output_humanized_dictionary(output_file_path: str) → None`

Writes a human-readable summary of the vocabulary to a text file:
- A table of token type → count, range, min, max, step size
- Special tokens with their integer values
- PDGIDs with their particle names (via `particle` library)
- Materials with their integer values

```python
d.output_humanized_dictionary("my_vocab_summary.txt")
```

---

### `update_dictionary_particle_list(input_data_filename, dictionary_filename) → None`

Scans a raw dataset file and updates the `pdgids` list in the dictionary JSON. Raises if more than 75 unique PDGIDs are found.

Useful when adapting the dictionary to a new dataset with a different particle set.

---

### Token-range helper methods

| Method | Returns | Description |
|---|---|---|
| `token_min(type_str)` | `float` | Minimum value for this feature's bin range |
| `token_max(type_str)` | `float` | Maximum value |
| `token_step_size(type_str)` | `float \| None` | Step size (only for linear binning) |
| `token_n_bins(type_str)` | `int` | Number of bins |
| `token_range(type_str)` | `float` | `max - min` |
| `token_range_str(offset, num)` | `str` | E.g. `"15 - 214"` (for display) |

---

## The `dictionary.json` Schema

A dictionary JSON must have:

```json
{
  "dataset": "my_dataset.csv",
  "tokenization_name": "my_tokenization",
  "special_tokens": {
    "padding": 0,
    "event_start": 1,
    "event_end": 2,
    "particle_start": 3,
    "particle_end": 4
  },
  "pdgids": {
    "0": 11, "1": -11, "2": 22, ...
  },
  "materials_named": {
    "silicon": 0, "lead": 1, ...
  },
  "tokenization": {
    "0": "pdgid", "1": "material", "2": "energy", "3": "eta", "4": "phi"
  },
  "padding": {
    "0": "particle_start", "1": "padding", "2": "padding", ...
  },
  "e_bin_data": {
    "min": 0.0, "max": 1000.0, "step_size": 1.0
  },
  "eta_bin_data": {
    "min": -5.0, "max": 5.0, "n_bins": 200, "tokenization": "gaussian",
    "gaussian_center": 0.0, "gaussian_sigma": 1.5
  }
}
```

---

## Gotchas

- **Quantile bins are cached as `.npy` files** next to the dictionary JSON. If you change the dataset or the number of quantile bins, delete the cached files before re-running — the code checks for the cache file first and returns it immediately if found.

- **PDGIDs beyond 75 are not supported**. The `update_dictionary_particle_list` method enforces this. If your dataset has more than 75 unique particle types, you need to either filter your dataset or modify the limit.

- **Unused PDGID slots are marked with `pdgid=0`** in the JSON. These placeholder entries consume a token slot but map to no real particle. `output_humanized_dictionary` prints them as `"none"`.

- **`tokenization_schema` order matters**. The `tokenizer.py` uses the position-to-feature mapping in `tokenization_schema` to know which column of the raw data maps to which token type. Changing this order breaks compatibility with existing tokenized data.

- **`linear_space` and `arange` are not identical to their NumPy equivalents**. The precision-safe versions may produce slightly different bin counts at edge cases. If you switch from `numpy.linspace` to `linear_space` mid-project, re-tokenize your data.

- **Bin edges, not bin centers**. The arrays stored in `e_bins`, `eta_bins`, etc. are **bin edges** (left boundaries). The actual number of bins is `len(e_bins)`, not `len(e_bins) - 1`, because digitization maps values into `[bin_i, bin_{i+1})` slots.

---

## Related

| Module | How it connects |
|---|---|
| `particleGPT/tokenizer.py` | Reads bin arrays and offsets from `Dictionary` to encode values |
| `particleGPT/untokenizer.py` | Uses offsets and bin arrays to decode tokens back to values |
| `analysis/analyzer.py` | Creates a `Dictionary` to interpret generated token sequences |
