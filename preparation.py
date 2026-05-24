# Utility for handling preparations--primarily binning data

import json
import numpy as np
from pathlib import Path
from particle import Particle
from enum import Enum
from types import SimpleNamespace

from scipy.stats import norm
from scipy.interpolate import interp1d

import data_manager as dm

script_dir = Path(__file__).resolve().parent


class Preparation():
    def __init__(self, preparation_filepath):
        self.preparation_filepath = preparation_filepath
        
        # Load this as a python SimpleNamespace for easier access to fields.
        with open(preparation_filepath, "r") as f:
            self.preparation_data_dict = json.load(f,)
        with open(preparation_filepath, "r") as f:
            self.preparation_data = json.load(f, object_hook=lambda d: SimpleNamespace(**d))