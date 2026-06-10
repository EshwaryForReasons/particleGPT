
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
# particleGPT
import pUtil
from dictionary import Dictionary
import data_manager
import paths

class dataset:
    
    @staticmethod
    def get_pdgid_frequency_distribution(dataset):
        """
        dataset: expected shape (num_events, num_particles_per_event, num_features_per_particle)
            where the first feature is the PDGID of the particle.
        freq: contains the frequency of each PDGID in the dataset.
        occurrences: contains the events (as row numbers) in which each PDGID occurs.
        """
        freq = Counter()
        occurrences = defaultdict(list)
        
        for event_idx, event in enumerate(dataset):
            found_ids = set()
            for particle in event:
                pdgid = particle[0]
                if pdgid == 0.0 or pdgid == np.nan:
                    continue
                freq[pdgid] += 1
                found_ids.add(pdgid)
            for pid in found_ids:
                occurrences[pid].append(event_idx)
                
        return freq, occurrences
    
    @staticmethod
    def find_events_lost_due_to_particle_removal(rows_occurring, n_least_frequent):
        """
        events_lost: If we remove all events containing the n_least_frequent particles, which events (as row numbers) do we lose?
        """
        sorted_items = sorted(rows_occurring.items(), key=lambda x: len(x[1]), reverse=False)
        sorted_items = sorted_items[:n_least_frequent]
        
        events_lost = set()
        for particle, events in sorted_items:
            events_lost.update(events)
        return events_lost

    @staticmethod
    def calculate_num_removable_particles(rows_occurring, n_allowed_event_removals):
        """
        n_most_remove_particles: number of least frequent particles we can remove and still only remove n_allowed_event_removals events.
        """
        n_most_removable_particles = 0
        for i in range(0, len(rows_occurring)):
            removed_events = dataset.find_events_lost_due_to_particle_removal(rows_occurring, i)
            n_removed_events = len(removed_events)
            if n_removed_events <= n_allowed_event_removals:
                n_most_removable_particles = i
        return n_most_removable_particles

    # ===================== 
    # Process data for various analysis 
    # =====================

    @staticmethod
    def extract_single_column_for_analysis(in_dataset, column_name, return_only_leading=False):
        """
        Processes in_dataset for analysis. This includes
        1) Removing incident particle from dataset (i.e. only keeping outgoing particles)
        2) Removing padding events (IMPORTANT: padding particles must be np.nan)
        3) Extracting the column of interest (e.g. 'pt', 'eta', etc.)
        4) Optionally, returning only the leading particle (i.e. the particle with the highest energy) for each event.
        
        @return
        processed_data: np.ndarray of shape (n_events, n_particles_per_event) or (n_events,) if return_only_leading=True
        """
        # @TODO: add verification that in_dataset is valid
        
        if column_name == 'num_particles':
            num_secondaries_across_events = []
            for event in in_dataset:
                # event has shape (num_particles (including padded), num features per particle)
                # Remove first particle (incident particle) and extract only pdgids since we only need one feature to test particle validity
                event_pdgids = event[1:, 0]
                valid_pdgids = event_pdgids[~np.isnan(event_pdgids)]
                num_secondaires = len(valid_pdgids)
                num_secondaries_across_events.append(num_secondaires)
            return np.asarray(num_secondaries_across_events)
        
        relevant_column_pos = plotting_v2.verbose_columns.index(column_name)
        # Remove incident particle
        secondaries = in_dataset[:, 1:, :]
        
        if return_only_leading:
            # Extract energies for comparison
            energies = secondaries[:, :, 1]
            # Only consider events with at least one particle
            valid_events = ~np.all(np.isnan(energies), axis=1)
            leading_idx = np.nanargmax(energies[valid_events], axis=1)
            leadings = secondaries[valid_events, :, :][np.arange(len(leading_idx)), leading_idx]
            leadings_column = leadings[:, relevant_column_pos].ravel()
            return leadings_column
        
        # Keep only column of interest
        # Flatten the array to make it easier to remove padding events
        secondaries_column = secondaries[:, :, relevant_column_pos].ravel()
        # Remove padding events (i.e. np.nan)
        secondaries_column = secondaries_column[~np.isnan(secondaries_column)]
        return secondaries_column
            
    @classmethod
    def extract_ein_eout_for_analysis(cls, model_name, in_dataset):
        MASS_CARBON = 931.5 * 12  # [MeV]
        
        e_in = np.full(len(in_dataset), np.nan, dtype=float)
        e_out = np.full(len(in_dataset), np.nan, dtype=float)

        # ===== Build PDGID mass dictionary =====
        
        dictionary_filename = pUtil.get_model_preparation_dir(model_name) / "dictionary.json"
        dictionary = Dictionary(dictionary_filename)

        pdgid_mass_dict = {}
        for _, pdgid in dictionary.pdgids.items():
            if pdgid == 0:
                continue
            part = Particle.from_pdgid(pdgid)
            pdgid_mass_dict[pdgid] = part.mass
            
        # ===== Calculate incoming and outgoing energies =====
        
        for idx, event in enumerate(in_dataset):
            # Incoming system
            in_pdgid = event[0][0]
            in_px    = event[0][2]
            in_py    = event[0][3]
            in_pz    = event[0][4]
            
            if np.isnan(in_pdgid):
                continue
            
            in_particle_vec = vector.obj(mass=pdgid_mass_dict[in_pdgid], px=in_px, py=in_py, pz=in_pz )
            in_material_vec = vector.obj(mass=MASS_CARBON, px=0.0, py=0.0, pz=0.0)
            in_vec = in_particle_vec + in_material_vec
            
            # Outgoing system
            out_vec = vector.obj(mass=0.0, px=0.0, py=0.0, pz=0.0)
            for particle in event[1:]:
                i_pdgid = particle[0]
                i_px    = particle[2]
                i_py    = particle[3]
                i_pz    = particle[4]
                
                if np.isnan(i_pdgid):
                    continue
                
                i_vec = vector.obj(mass=pdgid_mass_dict[i_pdgid], px=i_px, py=i_py, pz=i_pz)
                out_vec += i_vec
            
            e_in[idx] = in_vec.e
            e_out[idx] = out_vec.e
            
        return e_in, e_out
