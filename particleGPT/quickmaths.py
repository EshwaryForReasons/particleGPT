
import numpy as np
import paths as paths
from numba import njit

@njit("int64(float64, float64[:])", cache=True, nogil=True)
def custom_searchsorted(value, thresholds):
    """
    Return local feature-bin token using the same convention as FeatureBins.tokenize_value.

    thresholds are interior bin edges:
        edges = [0, 1, 2, 3]
        thresholds = [1, 2]

    Then:
        value < 1      -> 0
        1 <= value < 2 -> 1
        value >= 2     -> 2

    This intentionally clips below/above the configured range into edge bins,
    matching the behavior of np.searchsorted with side='right'.
    """
    return np.searchsorted(thresholds, value, side='right')
