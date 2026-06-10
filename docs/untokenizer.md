# `particleGPT/untokenizer.py`

## Overview

`untokenizer.py` is the inverse of `tokenizer.py`. It takes the raw integer token sequences produced by `sample.py` (one row per event in `samples.csv`) and reconstructs the physical event representation: a list of particles, each with continuous-valued features (energy, momentum, angles, etc.) and discrete attributes (PDGID, material).

The untokenizer relies on the same `Dictionary` object used for tokenization, using its bin edge arrays and offset constants to map token IDs back to physical values.

Output is typically a list of events, where each event is a list of particles, and each particle is a dict or array of feature values — ready to be fed into `metrics.py` and `plotting.py` for analysis.

---

## Quick Start

```python
from dictionary import Dictionary
from particleGPT.untokenizer import Untokenizer

d = Dictionary("config/models/my_dictionary.json")
u = Untokenizer(d)

# Load token sequences from samples.csv
import numpy as np
tokens = np.loadtxt("generated_samples/my_model/sampling_0/samples.csv", dtype=np.int64)

# Convert to physical events
events = u.untokenize_batch(tokens)
# events[0] → list of particles, each a dict of feature values
# events[0][0]["energy"] → float
# events[0][0]["pdgid"]  → int
```

---

## Key Classes

### `Untokenizer`

```python
class Untokenizer:
    def __init__(self, dictionary: Dictionary)
```

Holds a reference to the `Dictionary` and pre-builds reverse-lookup structures for fast decoding.

---

## Key Functions

### `Untokenizer.untokenize_batch(token_sequences: np.ndarray, ...) → list[list[dict]]`

Processes a 2-D array of token sequences (shape `(N_events, sequence_length)`) and returns a list of events. Each event is a list of particle dicts.

Internally calls `untokenize_sequence` for each row.

**Parameters**

| Param | Type | Description |
|---|---|---|
| `token_sequences` | `np.ndarray` | Shape `(N, seq_len)` — integer token IDs |
| `stop_at_event_end` | `bool` | Stop parsing when `event_end` token is seen (default: True) |
| `stop_at_padding` | `bool` | Stop parsing when padding token is seen (default: True) |

**Returns**: `list[list[dict]]` — outer list is events, inner list is particles

---

### `Untokenizer.untokenize_sequence(tokens: np.ndarray, ...) → list[dict]`

Decodes a single token sequence (shape `(seq_len,)`) into a list of particle dicts.

**Parsing logic:**
1. Scans for `event_start` token, begins parsing
2. For each `particle_start`...`particle_end` group, decodes features in tokenization schema order
3. Decodes each feature token using `decode_token(token, token_type)`
4. Stops at `event_end` or padding (if configured)
5. Returns a list of particle dicts

**Returns**: `list[dict]` — one dict per particle, keys are feature names (`"pdgid"`, `"energy"`, `"eta"`, etc.)

---

### `Untokenizer.decode_token(token: int, token_type: ETokenTypes) → float | int`

Converts a single token ID to its physical value.

| Token type | Decoding method |
|---|---|
| `PDGID` | Reverse-lookup in `dictionary.pdgids`: `token - PDGID_OFFSET → index → pdgid` |
| `MATERIAL` | Reverse-lookup in material names dict |
| `ENERGY` | Bin center: `(e_bins[i] + e_bins[i+1]) / 2` where `i = token - ENERGY_OFFSET` |
| `ETA`, `THETA`, `PHI`, `PT`, `PX`, `PY`, `PZ` | Same bin-center approach with respective bins |
| `PADDING`, `SPECIAL` | Returns `None` or raises (not a decodable physics value) |

---

### `untokenize_csv(csv_path: Path, dictionary: Dictionary, ...) → list[list[dict]]`

Convenience function. Loads a `samples.csv` file, tokenizes each row, and returns all events.

```python
from particleGPT.untokenizer import untokenize_csv
from dictionary import Dictionary

d = Dictionary("config/models/my_dictionary.json")
events = untokenize_csv("generated_samples/my_model/sampling_0/samples.csv", d)
```

---

### `events_to_dataframe(events: list[list[dict]]) → pd.DataFrame`

Flattens the nested event/particle structure into a pandas DataFrame with columns for each feature, plus an `event_id` column. Useful for passing to `plotting.py` and `metrics.py`.

```python
df = events_to_dataframe(events)
# df.columns: ['event_id', 'pdgid', 'material', 'energy', 'eta', 'phi', 'pt', ...]
```

---

### `compute_event_multiplicities(events: list[list[dict]]) → np.ndarray`

Returns a 1-D array of particle counts per event. Used for multiplicity distribution plots and W1 multiplicity metric.

---

## Decoding Details

### Continuous values: bin center

For any feature encoded as `token = np.digitize(value, bin_edges) + offset`, the decoded value is:

```python
bin_idx = token - offset
decoded = 0.5 * (bin_edges[bin_idx] + bin_edges[bin_idx + 1])
```

This introduces quantization error proportional to bin width. For the default linear binning, this is half a bin step. For Gaussian binning, the error is asymmetric (smaller near the center of the Gaussian).

### Higher-precision decoding: bin means

Instead of bin centers, you can use per-bin means from the training set (computed by `build_gaussian_bin_means` or `build_quantile_bin_means` in `dictionary.py`). This requires pre-computing the means once and passing them as an optional argument. The default (bin centers) is fast and sufficient for most analysis.

---

## Gotchas

- **Token IDs outside valid ranges** (e.g. out-of-vocabulary tokens that the model hallucinated) are handled by clamping to the nearest valid bin. If the model is well-trained, this should be rare. Monitor the fraction of out-of-range tokens in your analysis.

- **Reversed PDGID lookup depends on uniqueness**. The `pdgids_to_index` map requires every PDGID to appear exactly once in the dictionary. If two index slots have the same PDGID (e.g. two `0` placeholders), the reverse map will only keep one. Verify your dictionary JSON has unique PDGIDs.

- **`stop_at_event_end=True` is important for correctness**. Without it, the untokenizer will try to decode padding tokens and padding particles, producing spurious particles with `energy=0` or nonsensical PDGIDs. Always leave this enabled for analysis.

- **Particle order is preserved** within an event. The untokenizer returns particles in the order they appear in the token sequence, which is the order the model generated them. This order is not physically meaningful unless your tokenizer sorts particles before encoding.

- **Bin center decoding has a systematic bias**. For skewed distributions, the bin center is not the mean of the bin. Use `build_*_bin_means` for lower-bias decoding when statistical precision matters.

- **`float_precision` in `SamplingConfiguration`** truncates float values on output. The default of 5 decimal places is sufficient for most HEP quantities; reduce it if output file size is a concern.

---

## Related

| Module | How it connects |
|---|---|
| `dictionary.py` | Provides bin edges, offsets, and reverse maps |
| `particleGPT/tokenizer.py` | Forward operation (raw events → tokens) |
| `sample.py` | Produces the `samples.csv` that this module reads |
| `analysis/analyzer.py` | Calls `untokenize_csv()` or `untokenize_batch()` |
| `analysis/metrics.py` | Receives the untokenized events for metric computation |
