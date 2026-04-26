# Dictionary File #

The dictionary file contains all the information required for tokenization.
Here is breakdown of how the file is structured.


### Basics ###

The dictionary file is a `JSON` file with the following format:

```JSON
{
  "preparation_name": "",
  "dataset": "",
  "tokenization": {
    "0": "feature",
    ...
  },
  "padding": {
    "0": "padding",
    ...
  },
  "special_tokens": {
    "padding": 0,
    ...
  },
  "materials_named": {
    "G4_C": 0,
    ...
  },
  "eta_bin_data": {
    ...
  },
  "phi_bin_data": {
    ...
  },
  "pt_bin_data": {
    ...
  },
  "pdgids": {
    ...
  }
}
```

### Generic data ###

- The preparation name is important and should match the name of the preparation directory in `data/`.
- The dataset is important and should match the name of the dataset csv file in `data/`.

### Tokenization ###

Specifying the tokenization is split into two parts: 1. Tokenization and 2. Padding.
1. Tokenization.

Specifies how each particle is tokenized using the feature index and name. E.g. if a particle should be tokenized as (pdgid, pt, eta, phi), then it should be specified as such:
```JSON
"tokenization": {
    "0": "pdgid",
    "1": "pt",
    "2": "eta",
    "3": "phi"
}
```

2. Padding.

Specifies how "dummy particles" or "padding particles" should be tokenized. E.g. if a padding particle should be just padding, then it should be specified as such:
```JSON
"padding": {
    "0": "padding",
    "1": "padding",
    "2": "padding",
    "3": "padding"
}
```

### PDGIDs ###

- The PDGIDs list will be auto populated during the preparation phase.
- There are 75 particle slots and the preparation code will populate the list in the order the particles appear in the dataset.
- This should not be touched or modified except by the preparation script (`prepare.py`)

### Continuous Features ###

- Continuous features refers to, for example, energy, eta, phi, pt, etc.
- The tokenization specification for these features depends on the type of tokenization desired.

#### Linear tokenization ####

Linear tokenization is where each bin has uniform width.
```JSON
feature_bin_data": {
    "tokenization": "linear",
    "min": -4,
    "max": 4.01,
    "step_size": 0.001
}
```

`min`, `max`, and `step_size` are required for linear tokenization.
- `min` is the lowest value available. Everything in the dataset lower than this (for this feature) will be tokenized as the `min` value.
  - E.g. if the `min` is `10`, then `3`, `7`, and `10` will all be tokenized as `10`.
- `max` is the same as min, but for the largest value.
- `step_size` is the width of each bin.

#### Gaussian tokenization ####

Gaussian tokenization follows a gaussian curve for the bin widths. It is specified as such.
```JSON
"feature_bin_data": {
    "min": -4,
    "max": 4.01,
    "tokenization": "gaussian",
    "gaussian_center": 0.8,
    "gaussian_sigma": 1.2,
    "n_bins": 8000
},
```

`tokenization`, `min`, `max`, `n_bins`, `gaussian_center`, and `gaussian_sigma` are required for gaussian tokenization.
- `tokenization` defaults to linear, so it is required for gaussian.
- `min` is the lowest value available. Everything in the dataset lower than this (for this feature) will be tokenized as the `min` value.
  - E.g. if the `min` is `10`, then `3`, `7`, and `10` will all be tokenized as `10`.
- `max` is the same as min, but for the largest value.
- `n_bins` is the number of bins to split the data into.
- `gaussian_center` is the peak of the gaussian, which in this case is where the bins are the finest. They will become coarser as they move away from the specified center.
- `gaussian_sigma` is a factor specifying the "steepness" of the gaussian curve.

#### Quantile tokenization ####

Quantile tokenization uses `np.quantile` to create bins with equal frequency of occurrence. It is specified as such:
```JSON
"feature_bin_data": {
    "min": 0,
    "max": 35000,
    "tokenization": "quantile",
    "n_quantile_bins": 35000
  },
```

`tokenization`, `min`, `max`, `n_quantile_bins` are required for gaussian tokenization.
- `tokenization` defaults to linear, so it is required for quantile.
- `min` is the lowest value available. Everything in the dataset lower than this (for this feature) will be tokenized as the `min` value.
  - E.g. if the `min` is `10`, then `3`, `7`, and `10` will all be tokenized as `10`.
- `max` is the same as min, but for the largest value.
- `n_quantile_bins` is the number of bins to split the data into.