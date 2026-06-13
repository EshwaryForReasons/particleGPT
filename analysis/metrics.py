
import json
from collections import Counter, defaultdict
from types import SimpleNamespace
from pathlib import Path
import pickle
import time
# Math
import math
import numpy as np
import pandas as pd
from numba import njit, float64, types
from numba.experimental import jitclass
from numba.typed import Dict
from scipy.signal import savgol_filter

from particle import Particle
import vector
import jetnet
import jetnet.evaluation
# particleGPT
import pUtil
from particleGPT.dictionary import Dictionary
import data_manager
from particleGPT import untokenizer
import particleGPT.configurator as conf
import paths

script_dir = paths.PROJECT_DIR

class metrics:
    # Wrappers for JetNet metrics
    jetnet_get_suggested_kpd_fpd_features = jetnet.evaluation.get_fpd_kpd_jet_features
    jetnet_eval_cov_mmd = jetnet.evaluation.cov_mmd
    jetnet_eval_kpd = jetnet.evaluation.kpd
    jetnet_eval_fpd = jetnet.evaluation.fpd
    jetnet_eval_w1efp = jetnet.evaluation.w1efp
    jetnet_eval_w1m = jetnet.evaluation.w1m
    jetnet_eval_w1p = jetnet.evaluation.w1p
    
    # Custom metrics implementations go here...
