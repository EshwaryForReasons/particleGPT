
import numpy as np
import paths as paths
from numba import njit, prange


def pad_jagged_1d_fast(dataset: list[list], dtype=np.float64, pad_value=0.0):
    """
    dataset is a list[list] where the inner lists have non-uniform lengths.
    This functions pads them to the length of the largest inner list using
    provided pad_value
    
    This is faster to do without numba since we are dealing with regular lists.
    The preceeding statement has been tested and is not just theoretical.
    """
    n_events = len(dataset)
    if n_events == 0:
        return np.empty((0, 0), dtype=dtype)

    lengths = np.fromiter((len(e) for e in dataset), dtype=np.int64, count=n_events)
    max_event_length = int(lengths.max())

    out = np.full((n_events, max_event_length), pad_value, dtype=dtype)
    for i, e in enumerate(dataset):
        out[i, :lengths[i]] = e

    return out


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


@njit("UniTuple(float64, 4)(float64, float64, float64)", cache=True, nogil=True)
def linear_to_polar(px, py, pz):
    """
    returns (pt, eta, theta, phi) provided (px, py, pz)
    """
    r = np.sqrt(px * px + py * py + pz * pz)
    pt = np.sqrt(px * px + py * py)
    
    if r != 0.0:
        cos_theta = pz / r

        if cos_theta > 1.0:
            cos_theta = 1.0
        elif cos_theta < -1.0:
            cos_theta = -1.0

        theta = np.arccos(cos_theta)
    else:
        theta = 0.0
    
    phi = np.arctan2(py, px)
    
    if theta != 0.0:
        eta = -np.log(np.tan(theta / 2))
    else:
        eta = np.inf
        
    return pt, eta, theta, phi

@njit("float64[:,:,:](float64[:,:,:])", cache=True, nogil=True, parallel=True)
def convert_to_verbose_particles(input_data):
    """
    Convert raw-style particles into the verbose analysis layout.
    
    Input rows are expected to contain:
        pdgid, e, px, py, pz
    Output rows contain the columns expected by analysis_v2.plotting:
        pdgid, e, px, py, pz, pt, eta, theta, phi
    """
    
    n_events = input_data.shape[0]
    n_particles = input_data.shape[1]
    
    NUM_FEATURES_PER_PARTICLE_VERBOSE = 9
    verbose_data = np.empty((n_events, n_particles, NUM_FEATURES_PER_PARTICLE_VERBOSE), dtype=np.float64)
    
    for idx_e in prange(n_events):
        for idx_p in prange(n_particles):
            pdgid, e, px, py, pz = input_data[idx_e, idx_p]
            if np.isnan(pdgid):
                verbose_data[idx_e, idx_p, 0] = np.nan
                verbose_data[idx_e, idx_p, 1] = np.nan
                verbose_data[idx_e, idx_p, 2] = np.nan
                verbose_data[idx_e, idx_p, 3] = np.nan
                verbose_data[idx_e, idx_p, 4] = np.nan
                verbose_data[idx_e, idx_p, 5] = np.nan
                verbose_data[idx_e, idx_p, 6] = np.nan
                verbose_data[idx_e, idx_p, 7] = np.nan
                verbose_data[idx_e, idx_p, 8] = np.nan
                continue
            
            pt, eta, theta, phi = linear_to_polar(px, py, pz)
            verbose_data[idx_e, idx_p, 0] = pdgid
            verbose_data[idx_e, idx_p, 1] = e
            verbose_data[idx_e, idx_p, 2] = px
            verbose_data[idx_e, idx_p, 3] = py
            verbose_data[idx_e, idx_p, 4] = pz
            verbose_data[idx_e, idx_p, 5] = pt
            verbose_data[idx_e, idx_p, 6] = eta
            verbose_data[idx_e, idx_p, 7] = theta
            verbose_data[idx_e, idx_p, 8] = phi
            
    return verbose_data