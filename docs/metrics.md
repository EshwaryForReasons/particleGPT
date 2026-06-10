# `analysis/metrics.py`

## Overview

`metrics.py` computes quantitative physics metrics to evaluate how well the model's generated events match the real (test-set) events. It wraps functions from the [JetNet](https://github.com/jet-net/JetNet) library and adds custom metrics specific to this pipeline.

The primary metrics are:

| Metric | Description |
|---|---|
| **W1** | Wasserstein-1 distance per feature — measures how similar the 1-D marginal distributions are |
| **W1M** | Wasserstein-1 distance on multiplicity (particle count per event) |
| **FPD** | Fréchet Physics Distance — like FID, but using physics features; sensitive to correlations |
| **KPD** | Kernel Physics Distance — another distance metric on event-level feature distributions |

All metrics are computed between the generated event set and the real event set. Lower values are better for all metrics.

---

## Quick Start

```python
from analysis.metrics import compute_all_metrics

# events are lists of particle dicts (from untokenizer.py)
results = compute_all_metrics(generated_events, real_events)

print(f"W1 energy:      {results['w1_energy']:.4f}")
print(f"W1 multiplicity:{results['w1_multiplicity']:.4f}")
print(f"FPD:            {results['fpd']:.4f}")
print(f"KPD:            {results['kpd']:.4f}")
```

---

## Key Functions

### `compute_all_metrics(generated_events, real_events, features=None) → dict`

Top-level function. Computes all available metrics and returns a flat dict.

**Parameters**

| Param | Type | Description |
|---|---|---|
| `generated_events` | `list[list[dict]]` | Untokenized generated events |
| `real_events` | `list[list[dict]]` | Untokenized real events |
| `features` | `list[str] \| None` | Which features to include (default: all in dictionary) |

**Returns**: `dict` with keys like:
```python
{
    "w1_energy": 0.0032,
    "w1_eta": 0.0018,
    "w1_phi": 0.0021,
    "w1_pt": 0.0054,
    "w1_multiplicity": 1.23,
    "fpd": 0.0041,
    "fpd_std": 0.0003,
    "kpd": 0.0029,
    "kpd_std": 0.0002,
    "n_generated": 100000,
    "n_real": 100000,
}
```

---

### `compute_w1_distances(generated_events, real_events, features) → dict`

Computes Wasserstein-1 distances for each specified feature, aggregating all values across particles in the event set (i.e. treating all particles as a flat distribution).

The Wasserstein-1 distance between two 1-D distributions P and Q is:
```
W₁(P, Q) = ∫ |F_P(x) - F_Q(x)| dx
```
where F is the CDF. Computed using `scipy.stats.wasserstein_distance`.

**Parameters**

| Param | Type | Description |
|---|---|---|
| `generated_events` | `list[list[dict]]` | Untokenized generated events |
| `real_events` | `list[list[dict]]` | Untokenized real events |
| `features` | `list[str]` | Feature names to compute W1 for |

**Returns**: `dict[str, float]` — feature name → W1 distance

---

### `compute_w1_multiplicity(generated_events, real_events) → float`

Computes the Wasserstein-1 distance on particle multiplicity distributions (particle count per event).

---

### `compute_fpd(generated_events, real_events, ...) → Tuple[float, float]`

Computes the Fréchet Physics Distance between generated and real events using JetNet's `fpd` implementation.

FPD treats each event as a variable-length set of particle feature vectors and measures the distance between the multivariate Gaussian fits to both distributions in a learned feature space.

**Returns**: `(fpd_mean, fpd_std)` — the mean and standard deviation over bootstrap resamples

---

### `compute_kpd(generated_events, real_events, ...) → Tuple[float, float]`

Computes the Kernel Physics Distance using JetNet's `kpd` implementation. KPD uses a maximum mean discrepancy (MMD) approach, which is more sensitive to distribution tails than FPD.

**Returns**: `(kpd_mean, kpd_std)`

---

### `events_to_jetnet_format(events: list[list[dict]], features: list[str], max_particles: int) → np.ndarray`

Converts the nested event/particle list structure into the 3-D array format expected by JetNet metrics:

```
shape: (N_events, max_particles, N_features)
```

Events with fewer than `max_particles` particles are zero-padded. The `mask` (1 for real particles, 0 for padding) is derived from the padding.

---

### `extract_feature_arrays(events: list[list[dict]], feature: str) → np.ndarray`

Collects all values for a single feature across all events and particles into a flat 1-D array. Used for W1 computation.

---

### `save_metrics(metrics: dict, output_path: Path) → None`

Serialises the metrics dict to `metrics.json`. Numbers are rounded to 6 significant figures for readability.

---

### `load_metrics(metrics_path: Path) → dict`

Loads a previously saved `metrics.json`. Useful for table generation without recomputing metrics.

---

## Metric Details

### Wasserstein-1 Distance

W1 is a particularly interpretable metric because it has the same units as the feature being measured. A W1 energy of 0.003 GeV means the generated and real energy distributions differ by ~3 MeV on average. Values near zero mean excellent agreement.

**Limitation**: W1 is a 1-D marginal metric. It cannot detect incorrect correlations between features (e.g. the model may get the energy distribution right but pair high-energy photons with wrong azimuthal angles). Use FPD/KPD to check correlations.

### FPD and KPD

FPD and KPD are set-level metrics that compare the distributions of **event-level feature vectors**, not individual particle features. They are sensitive to correlations and event-level structure.

Both require a reasonable minimum sample size for stable estimates (typically ≥ 10,000 events). With fewer events, variance is high and the metrics are unreliable.

---

## Gotchas

- **JetNet expects zero-padded arrays, not NaN-padded**. The `events_to_jetnet_format` function handles this conversion. Do not pass NaN-padded arrays directly to JetNet functions.

- **FPD/KPD variance**: run with `n_eval=10` (or more) bootstrap samples to get a reliable standard deviation. The default in JetNet is lower for speed; increase it for publication-quality estimates.

- **W1 is sensitive to outliers**. A single generated event with an unphysical energy value (e.g. from the model generating an out-of-range token) will inflate W1 significantly. Check the distribution tails if W1 is unexpectedly large.

- **Matching generated and real event counts matters**. If `N_generated >> N_real` or vice versa, W1 will be artificially inflated because you are comparing differently-sized empirical distributions. The analyzer matches counts by default.

- **`max_particles` for JetNet formatting** should match the value used during tokenization. Using a value that is too small will silently truncate events; using one that is too large wastes memory.

---

## Related

| Module | How it connects |
|---|---|
| `analysis/analyzer.py` | Calls `compute_all_metrics()` and `save_metrics()` |
| `analysis/plotting.py` | Uses `extract_feature_arrays()` for histogram data |
| `analysis/tables.py` | Reads `metrics.json` to populate table cells |
| `particleGPT/untokenizer.py` | Provides the event format consumed by this module |
