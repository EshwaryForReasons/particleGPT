
import jetnet.evaluation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path

import jetnet

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
                if pdgid == 0.0:
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
    
class plotting:
    """
    All plotting functions will follow a similar API. This allows easy intuitive generation
    of various types of plots.
    
    normalized: (optional: False) bool, should the values be normalized to a area of 1 before plotting?
    use_log: (optional: False) bool, should the dependent axis be log scaled.
    juxtaposed: (optional: False) bool, if input contains multiple values (array) should all be plotted on the same
        figure should different figures be used side-by-side.
    out_file: (optional) pathlib.Path, file to save figure to. show will always be called since it
        naturally only works if there is a way to show the figures.
    """
    
    # Colors in the order they will be used for overlapping graphs.
    colors = ['blue', 'orange', 'purple', 'green', 'red']
    
    @staticmethod
    def plot_continuous_distribution(all_data, all_labels, name="unspecified", min=None, max=None, n_bins=None, normalized=False, use_log=False, juxtaposed=False, out_file=None):
        """Generates distributions (histogram) for the provided data. This works for any "continuous" data, i.e.
            energy, momentum, etc. distributions. This will not work for "discrete" data, like pdgid distributions.

        Args:
            all_data (array_like or array of array_like): data to plot distributions for
            n_bins (int): number of bins for histogram
            min (int, optional): Min value for the histogram, use min(data) if not provided. Defaults to None.
            max (int, optional): Max value for the histogram, use max(data) if not provided. Defaults to None.
        """
        if n_bins == None or n_bins == 0:
            print("analysis: plotting: plot_continuous_distribution: invalid n_bins provided. exiting.")
            return
        
        # Ensure all-data and all_labels are lists of lists
        if not isinstance(all_data, list):
            all_data = [all_data]
        if not isinstance(all_labels, list):
            all_labels = [all_labels]
                
        if min == None:
            min = min(all_data.flatten())
        if max == None:
            max = max(all_data.flatten())
        
        weights = [None] * len(all_data)
        if normalized:
            for idx, i_data in enumerate(all_data):
                weights[idx] = np.ones_like(i_data) / len(i_data)
        
        plt.figure(figsize=(21, 6), dpi=300)
        plt.subplot(1, 2, 1)
        if use_log:
            plt.yscale('log')
        plt.xlim([min, max])
        for i_weight, i_data, i_label, i_color, in zip(weights, all_data, all_labels, plotting.colors):
            plt.hist(i_data, bins=n_bins, weights=i_weight, range=(min, max), alpha=0.7, color=i_color, label=i_label)
        plt.title(f"Histogram of {name}")
        plt.xlabel(name)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        if out_file != None:
            plt.savefig(out_file, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_discrete_distribution(all_freq_dists, all_labels, name="unspecified", normalized=False, use_log=False, juxtaposed=False, out_file=None):
        """Generates distributions (histogram) for the provided data. This works for any "discrete" data, i.e.
            pdgid distributions. This will not work for "continuous" data, like energy or momentum distributions.

        Args:
            all_freq_dists (array_like or array of array_like): data to plot distributions for
            n_bins (int): number of bins for histogram
        """
        
        # Ensure all-data and all_labels are lists of lists
        if not isinstance(all_freq_dists, list):
            all_freq_dists = [all_freq_dists]
        if not isinstance(all_labels, list):
            all_labels = [all_labels]
        
        if normalized:
            all_normalized_freq_dists = []
            for i_freq_dist in all_freq_dists:
                total = sum(i_freq_dist.values())
                norm_dists = {
                    pid: count / total
                    for pid, count in i_freq_dist.items()
                }
                all_normalized_freq_dists.append(norm_dists)
            all_freq_dists = all_normalized_freq_dists
        
        # Take a union of all freq dists
        all_bins = sorted(set().union(*[freq.keys() for freq in all_freq_dists]))
        # freq dist which sums freq across all the dists in all_freq_dist.
        # This is needed because some of them might be have 0 frequency and thus wont
        # exist in the freq dist. That would break sorting so we use this fix.
        total_freqs = {p: sum(c.get(p, 0) for c in all_freq_dists) for p in all_bins}
        sorted_freqs = sorted(all_bins, key=lambda p: total_freqs[p], reverse=True)
        aligned_values = [[freq.get(f, 0) for f in sorted_freqs] for freq in all_freq_dists]
        
        x = range(len(sorted_freqs))
        plt.figure(figsize=(21, 6), dpi=300)
        if use_log:
            plt.yscale('log')
        for i_value, i_label, i_color in zip(aligned_values, all_labels, plotting.colors):
            plt.bar(x, i_value, label=i_label, color=i_color, alpha=0.7)
        plt.xticks(x, sorted_freqs, rotation=45, ha='right')
        plt.xlabel(name)
        plt.ylabel(f"{'Normalized ' if normalized else ''}Frequency")
        plt.title(f"Normalized {name} Distributions")
        plt.legend()
        plt.tight_layout()
        if out_file != None:
            plt.savefig(out_file, bbox_inches='tight')
        plt.show()