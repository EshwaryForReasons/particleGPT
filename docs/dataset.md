# `particleGPT/dataset.py`

## Overview

`particleGPT/dataset.py` contains dataset utilities specific to the `particleGPT` module — helpers for working with raw Geant4 data files, constructing event arrays, and interfacing between the raw data format and the tokenizer/untokenizer.

This module sits between the raw data on disk and the tokenization pipeline. It does **not** handle the tokenized binary files used for training (that is `train.py`'s `TokenBlockDataset`); it handles the **raw particle event arrays** that are inputs to `tokenizer.py` and outputs from `untokenizer.py`.

---

## Quick Start

```python
from particleGPT.dataset import ParticleDataset

ds = ParticleDataset("data/raw/my_dataset.csv", dictionary)

print(f"Events: {len(ds)}")
event = ds[0]          # list of particles, each a np.ndarray of feature values
print(event[0])        # [pdgid, material, energy, eta, phi, pt, ...]
```

---

## Key Classes

### `ParticleDataset`

```python
class ParticleDataset:
    def __init__(self, filepath: Path, dictionary: Dictionary, max_events: int | None = None)
```

Loads a raw Geant4 dataset from CSV into memory, respecting the feature order defined in `dictionary.tokenization_schema`.

**Parameters**

| Param | Type | Description |
|---|---|---|
| `filepath` | `Path` | Path to the raw CSV file |
| `dictionary` | `Dictionary` | Loaded vocabulary/schema object |
| `max_events` | `int \| None` | If set, only load the first `max_events` events (useful for quick inspection) |

**Key attributes**

| Attribute | Description |
|---|---|
| `events` | `list[list[np.ndarray]]` — outer list is events, inner is particles |
| `num_events` | Total number of loaded events |
| `feature_names` | Feature column names in the order they appear per particle |

---

### `__len__() → int`

Returns `num_events`.

---

### `__getitem__(idx: int) → list[np.ndarray]`

Returns the list of particle feature arrays for event `idx`. Each array has shape `(num_features_per_particle,)`.

---

### `get_feature_array(feature_name: str) → np.ndarray`

Returns a flat 1-D array of all values for a given feature across all events and particles. Useful for computing statistics or bin edges.

```python
all_energies = ds.get_feature_array("energy")
print(f"Energy range: {all_energies.min():.2f} – {all_energies.max():.2f}")
```

---

### `get_multiplicity_distribution() → np.ndarray`

Returns a 1-D array of particle counts per event.

---

## Key Functions

### `load_events_from_csv(filepath: Path, feature_columns: list[str]) → list[list[np.ndarray]]`

Low-level CSV reader. Reads a Geant4-format CSV where:
- Each row is one particle
- Events are delimited by a special separator row or by an event ID column
- Feature columns are selected by name

Returns a nested list: `events[event_idx][particle_idx]` = feature array.

---

### `filter_events(events, min_particles=1, max_particles=None) → list[list[np.ndarray]]`

Filters events by particle count.

| Param | Type | Description |
|---|---|---|
| `min_particles` | `int` | Minimum particles per event (default 1, removes empty events) |
| `max_particles` | `int \| None` | Maximum particles per event (truncates or removes, see `truncate` param) |

---

### `events_to_numpy(events: list[list[np.ndarray]], pad_value=np.nan, max_particles=None) → np.ndarray`

Converts the nested event list to a 3-D NumPy array of shape `(N_events, max_particles, N_features)` with NaN padding for events with fewer than `max_particles` particles. Useful for vectorized processing in `metrics.py`.

---

## Data Format Notes

### Raw CSV format

The raw dataset is a CSV where each row represents one secondary particle from a collision event. A typical header looks like:

```
event_id, pdgid, material, energy, eta, theta, phi, pt, px, py, pz
```

The `event_id` column groups particles into events. Particles within an event are not necessarily sorted.

### Feature ordering

The order of features in the raw CSV does **not** need to match `dictionary.tokenization_schema`. The `ParticleDataset` class handles the column-to-schema mapping. Only the features listed in `tokenization_schema` are loaded; others are ignored.

---

## Gotchas

- **Memory usage**: loading all events into memory as Python lists is not memory-efficient for very large datasets (millions of events). The tokenizer writes the data to a binary file for this reason. Use `max_events` during development to avoid OOM.

- **Event delimiter parsing** is format-specific. If your CSV uses a different event separator than the default (e.g. a blank line vs. a `event_id` column), you may need to adjust `load_events_from_csv`.

- **NaN handling in `events_to_numpy`**: NaN is used as a sentinel for "no particle" in padded slots. Make sure downstream code (`metrics.py`) handles NaN correctly — most NumPy reductions propagate NaN unless you use `nanmean`, `nanstd`, etc.

- **Feature ordering in the output arrays** follows `dictionary.tokenization_schema`, not the CSV column order. This means the same index in the particle array always refers to the same feature, regardless of CSV layout.

---

## Related

| Module | How it connects |
|---|---|
| `dictionary.py` | Schema and feature names come from `Dictionary` |
| `particleGPT/tokenizer.py` | Uses `ParticleDataset` (or equivalent) as input to tokenization |
| `data_manager.py` | Provides lower-level CSV loading used by this module |
| `analysis/metrics.py` | Receives arrays from `events_to_numpy()` or `untokenizer` |
