
import json
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import jetnet
import jetnet.evaluation

import pUtil
from dictionary import Dictionary
import data_manager

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
    
    normalized: (optional: False) bool, should the values be normalized to an area of 1 before plotting?
    use_log: (optional: False) bool, should the dependent axis be log scaled?
    juxtaposed: (optional: False) bool, if input contains multiple values (array) should all be plotted on the same
        axis or should different axes be used side-by-side.
    out_file: (optional) pathlib.Path, file to save figure to. plt.show will always be called since it
        naturally only works if there is a way to show the figures.
    """
    
    # Colors in the order they will be used for overlapping graphs.
    colors = ['blue', 'orange', 'purple', 'green', 'red']
    default_figsize = (21, 6)
    default_dpi = 300
    distributions_per_row = 3
    
    columns = ["num_particles", "pdgid", "e", "px", "py", "pz", "pt", "eta", "theta", "phi"]
    
    """
    Generic plotting functions.
    """
    
    @staticmethod
    def set_labels(x_label, y_label, in_fig=None):
        """
        Sets the x and y labels for the current plot.
        """
        if in_fig is None:
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        else:
            in_fig.supxlabel(x_label)
            in_fig.supylabel(y_label) 
    
    @staticmethod
    def plot_hist(all_data, n_bins=10, range_min=None, range_max=None, label='unspecified', color=None, normalized=False, use_log=False, out_file=None, in_ax=None):
        """Generates distributions (histogram) for the provided data. This works for any "continuous" data, i.e.
            energy, momentum, etc. distributions. This will not work for "discrete" data, like pdgid distributions.

        Args:
            all_data (array_like or array of array_like): data to plot distributions for
            min (int, optional): Min value for the histogram, use min(data) if not provided. Defaults to None.
            max (int, optional): Max value for the histogram, use max(data) if not provided. Defaults to None.
            n_bins (int, optional): Number of bins for histogram. Defaults to 10.
        """
        
        color = color or plotting.colors[0]
        if range_min == None:
            range_min = np.min(all_data)
        if range_max == None:
            range_max = np.max(all_data)
        
        weights = None
        if normalized:
            weights = np.ones_like(all_data) / len(all_data)
        
        fig = None
        ax = in_ax
        if not in_ax:
            fig, ax = plt.subplots(figsize=plotting.default_figsize, dpi=plotting.default_dpi)
        if use_log:
            ax.set_yscale('log')
        # ax.set_xlim([range_min, range_max])
        ax.hist(all_data, bins=n_bins, weights=weights, range=(range_min, range_max), alpha=0.7, color=color, label=label)
        ax.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        if not in_ax:
            plt.tight_layout()
            if out_file != None:
                plt.savefig(out_file, bbox_inches='tight')
            plt.show()
        
        return fig, ax
    
    @staticmethod
    def plot_bar(freq_dist, x_ticks=None, label="unspecified", color=None, normalized=False, use_log=False, out_file=None, in_ax=None):
        """Generates distributions (histogram) for the provided data. This works for any "discrete" data, i.e.
            pdgid distributions. This will not work for "continuous" data, like energy or momentum distributions.

        Args:
            all_freq_dists (array_like or Counter): data to plot distributions for
            x_ticks (array_like, optional): x-axis labels for the histogram. If freq_dist is a Counter object, item names will
                be extracted from that. Defaults to None.
        """
        
        if color is None:
            color = plotting.colors[0]
        
        # freq_dist can be a list, or a Counter object.
        if normalized and isinstance(freq_dist, list):
            freq_dist = [count / sum(freq_dist) for count in freq_dist]
        elif normalized and isinstance(freq_dist, Counter):
            freq_dist = {item: count / sum(freq_dist.values()) for item, count in freq_dist.items()}
        
        x = range(len(freq_dist))
        
        fig = None
        ax = in_ax
        if not in_ax:
            fig, ax = plt.subplots(figsize=plotting.default_figsize, dpi=plotting.default_dpi)
        if use_log:
            ax.set_yscale('log')
        ax.bar(x, freq_dist, label=label, color=color, alpha=0.7)
        if x_ticks:
            ax.set_xticks(x, x_ticks, rotation=45, ha='right')
        elif isinstance(freq_dist, Counter):
            ax.set_xticks(x, list(freq_dist.keys()), rotation=45, ha='right')
        ax.legend()
        
        if not in_ax:
            plt.tight_layout()
            if out_file != None:
                plt.savefig(out_file, bbox_inches='tight')
            plt.show()
    
        return fig, ax
    
    @staticmethod
    def plot_line(x, y, label='unspecified', color=None, type='solid', y_lim=None, use_log=False, out_file=None, in_ax=None):
        """Generates line graph of the provided data. This works for any "discrete" data, i.e.
            training data (iter vs. val loss). This will not work for "continuous" data, like energy or momentum distributions.

        Args:
            all_data (array_like or array of array_like): data to plot distributions for. Each data should have format [[label, x, y, color_override, type_override], ...]
                which should an x vs. y array_like.
            all_labels (array_like or array of array_like): labels for the data to plot distributions for.
        """
        
        if color is None:
            color = plotting.colors[0]
        
        fig = None
        ax = in_ax
        # Padding in_plt only makes since if the figure already exists.
        if not in_ax:
            fig, ax = plt.subplots(figsize=plotting.default_figsize, dpi=plotting.default_dpi)
        if use_log:
            ax.set_yscale('log')
        if y_lim:
            ax.set_ylim(y_lim)
        ax.plot(x, y, label=label, color=color, linestyle=type, linewidth=0.5)
        ax.legend()

        # If a axis object was provided, we assume we are constructing a larger figure and do not show or save yet.
        if not in_ax:
            plt.tight_layout()
            if out_file != None:
                plt.savefig(out_file, bbox_inches='tight')
            plt.show()
        
        return fig, ax

    @staticmethod
    def plot_scat(x, y, label='unspecified', color=None, type='o', y_lim=None, use_log=False, out_file=None, in_ax=None):
        """Generates a scatter plot of the provided data. This works for any "discrete" data, i.e.
            training data (iter vs. val loss). This will not work for "continuous" data, like energy or momentum distributions.

        Args:
            all_data (array_like or array of array_like): data to plot distributions for. Each data should have format [[label, x, y], ...]
                which should an x vs. y array_like.
            all_labels (array_like or array of array_like): labels for the data to plot distributions for.
        """
        
        if color is None:
            color = plotting.colors[0]
        
        fig = None
        ax = in_ax
        # Padding in_plt only makes since if the figure already exists.
        if not in_ax:
            fig, ax = plt.subplots(figsize=plotting.default_figsize, dpi=plotting.default_dpi)
        if use_log:
            ax.set_yscale('log')
        if y_lim:
            ax.set_ylim(y_lim)
        ax.scatter(x, y, label=label, color=color, marker=type, s=50)
        ax.legend()

        # If a plt object was provided, we assume we are constructing a larger figure and do not show or save yet.
        if not in_ax:
            ax.tight_layout()
            if out_file != None:
                ax.savefig(out_file, bbox_inches='tight')
            ax.show()
        
        return fig, ax

    """
    Plotting training runs and distributions of leading particles.
    """
    
    @staticmethod
    def plot_training_run(model_names, y_lim=None, use_log=False, out_file=None):
        """
        Wrapper to plot a training run. Handles plotting lines and markers for training and validation loss.
        """
        
        if not isinstance(model_names, list):
            model_names = [model_names]
        
        fig, ax = plt.subplots(figsize=plotting.default_figsize, dpi=plotting.default_dpi)
        plotting.set_labels("Iteration", "Loss", in_fig=fig)
        for idx, model_name in enumerate(model_names):
            model_data = tables.get_all_data(model_name)
            min_val_row = model_data.checkpointed_df.loc[model_data.checkpointed_df['val_loss'].idxmin()]
            
            plotting.plot_line(model_data.running_df['iter'], model_data.running_df['train_loss'], label=f'Training Loss ({model_name})', color=plotting.colors[idx], type='solid', use_log=use_log, y_lim=y_lim, in_ax=ax)
            plotting.plot_line(model_data.running_df['iter'], model_data.running_df['val_loss'], label=f'Validation Loss ({model_name})', color=plotting.colors[idx], type='dashed', use_log=use_log, y_lim=y_lim, in_ax=ax)
            plotting.plot_scat(min_val_row['iter'], min_val_row['train_loss'], label=f'Min Saved Train Loss ({model_name}; {min_val_row["train_loss"]:.4f})', color=plotting.colors[idx], type='s', use_log=use_log, y_lim=y_lim, in_ax=ax)
            plotting.plot_scat(min_val_row['iter'], min_val_row['val_loss'], label=f'Min Saved Val Loss ({model_name}; {min_val_row["val_loss"]:.4f})', color=plotting.colors[idx], type='o', use_log=use_log, y_lim=y_lim, in_ax=ax)
            final_row = model_data.running_df.iloc[-1]
            ax.annotate(model_name, xy=(final_row['iter'], final_row['val_loss']), xytext=(final_row['iter'] * 1.005, final_row['val_loss'] - 0.02), fontsize=9, color=plotting.colors[idx])
        fig.suptitle(f"Training progress for {model_names}")
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, ax

    @staticmethod
    def _get_common_data(model_name):
        """
        Retrieves bin widths and ranges for each feature and the real and sampled leading particles dataframes.
        """
        dictionary_filename = pUtil.get_model_preparation_dir(model_name) / 'dictionary.json'
        real_leading_test_particles_filename = pUtil.get_model_preparation_dir(model_name) / 'real_leading_test_particles.csv'
        sampled_leading_particles_filename = pUtil.get_latest_sampling_dir(model_name) / 'sampled_leading_particles.csv'
        
        dictionary = Dictionary(dictionary_filename)
        
        def get_bin_count(type_str):
            step_size = dictionary.token_step_size(type_str)
            if step_size == 0:
                return 0
            if type_str in ['eta', 'theta', 'phi']:
                step_size = 0.05
            return int(dictionary.token_range(type_str) // step_size)
        
        columns = ["num_particles", "pdgid", "e", "px", "py", "pz", "pt", "eta", "theta", "phi"]
        bin_settings = {
            "num_particles": { "min": 0,                             "max": 50,                            "bins": 50 },
            "e":             { "min": 0,                             "max": 35000,                         "bins": 350 },
            "px":            { "min": -5000,                         "max": 35000,                         "bins": 400 },
            "py":            { "min": -5000,                         "max": 35000,                         "bins": 400 },
            "pz":            { "min": -5000,                         "max": 35000,                         "bins": 400 },
            "eta":           { "min": dictionary.token_min('eta'),   "max": dictionary.token_max('eta'),   "bins": get_bin_count('eta') },
            "theta":         { "min": dictionary.token_min('theta'), "max": dictionary.token_max('theta'), "bins": get_bin_count('theta') },
            "phi":           { "min": dictionary.token_min('phi'),   "max": dictionary.token_max('phi'),   "bins": get_bin_count('phi') },
            "pt":            { "min": dictionary.token_min('pt'),    "max": dictionary.token_max('pt'),    "bins": get_bin_count('pt') },
        }

        real_df = pd.read_csv(real_leading_test_particles_filename, sep=" ", names=columns, engine="c", header=None)
        sampled_df = pd.read_csv(sampled_leading_particles_filename, sep=" ", names=columns, engine="c", header=None)
        return bin_settings, real_df, sampled_df

    @staticmethod
    def plot_distribution_leading(model_names, column_name=None, normalized=False, use_log=False, out_file=None):
        unit = ''
        if column_name in ['e', 'pt', 'px', 'py', 'pz']:
            unit = '(MeV)'
        elif column_name in ['eta', 'theta', 'phi']:
            unit = '(angular)'
        
        num_horizontal, num_vertical = min(len(model_names), plotting.distributions_per_row), (math.ceil(1 / plotting.distributions_per_row))
        fig, axes = plt.subplots(num_vertical, num_horizontal, figsize=(8 * num_horizontal, 6 * num_vertical), sharex=False, sharey=True, dpi=plotting.default_dpi)
        fig.suptitle(f'{column_name} Distribution for Leading Particles')
        plotting.set_labels(f'{column_name} {unit}', 'Frequency', in_fig=fig)
        for ax, model_name in zip([axes], model_names):
            bin_settings, real_df, sampled_df = plotting._get_common_data(model_name)
            range_min = bin_settings[column_name]['min']
            range_max = bin_settings[column_name]['max']
            n_bins = bin_settings[column_name]['bins']
            
            plotting.plot_hist(real_df[column_name], range_min=range_min, range_max=range_max, n_bins=n_bins, normalized=normalized, label=f'Input ({model_name})', color=plotting.colors[0], use_log=use_log, in_ax=ax)
            plotting.plot_hist(sampled_df[column_name], range_min=range_min, range_max=range_max, n_bins=n_bins, normalized=normalized, label=f'Sampled ({model_name})', color=plotting.colors[1], use_log=use_log, in_ax=ax)
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, ax
    
    @staticmethod
    def plot_pdgid_distribution_leading(model_names, normalized=False, use_log=False, out_file=None):
        model_name = model_names[0]
        
        _, real_df, sampled_df = plotting._get_common_data(model_name)
        real_pdgids, sampled_pdgids = real_df['pdgid'], sampled_df['pdgid']
        sampled_pdgids = sampled_pdgids
        real_freq, sampled_freq = Counter(real_pdgids), Counter(sampled_pdgids)
        
        # Union of all particle labels from both histograms
        all_particles = sorted(set(real_freq.keys()).union(sampled_freq.keys()))
        # Sorting them by frequency in real leading particles to ensure a legible plot
        sorted_particles = sorted(all_particles, key=lambda p: real_freq[p], reverse=True)
        # Build aligned values for both histograms
        real_values = [real_freq.get(p, 0) for p in sorted_particles]
        sampeld_values = [sampled_freq.get(p, 0) for p in sorted_particles]
        
        fig, ax = plt.subplots(figsize=(21, 8), dpi=plotting.default_dpi)
        fig.suptitle("Normalized Particle Type Distributions")
        plotting.set_labels("Particle Type", "Frequency", in_fig=fig)
        plotting.plot_bar(real_values, label=f'Input ({model_name})', color=plotting.colors[0], normalized=normalized, use_log=use_log, in_ax=ax)
        plotting.plot_bar(sampeld_values, label=f'Sampled ({model_name})', color=plotting.colors[1], normalized=normalized, use_log=use_log, in_ax=ax)
        x = range(len(sorted_particles))
        ax.set_xticks(x, sorted_particles, rotation=45, ha='right')
        ax.legend()
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, ax

    @staticmethod
    def plot_pdgid_distribution_all(model_names, normalized=False, use_log=False, out_file=None):
        model_name = model_names[0]
        
        meta_data = tables.get_meta_data(model_name)
        
        testing_bin_filename = pUtil.get_model_preparation_dir(model_name) / 'test_real.bin'
        generated_samples_filename = pUtil.get_latest_sampling_dir(model_name) / 'untokenized_samples.csv'
        
        num_tokens_per_particle_raw = 5
        testing_real_data = np.memmap(testing_bin_filename, dtype=np.float64, mode='r')
        testing_real_data = testing_real_data.reshape(-1, int((meta_data.max_sequence_length - 2) / num_tokens_per_particle_raw), num_tokens_per_particle_raw)
        generated_sample_data = data_manager.load_geant4_dataset(generated_samples_filename, pad_token=0.0)
        
        real_pdgid_freq_dist, _ = dataset.get_pdgid_frequency_distribution(testing_real_data)
        sampled_pdgid_freq_dist, _ = dataset.get_pdgid_frequency_distribution(generated_sample_data)
        real_freq, sampled_freq = real_pdgid_freq_dist, sampled_pdgid_freq_dist
        
        # Union of all particle labels from both histograms
        all_particles = sorted(set(real_freq.keys()).union(sampled_freq.keys()))
        # Sorting them by frequency in real leading particles to ensure a legible plot
        sorted_particles = sorted(all_particles, key=lambda p: real_freq[p], reverse=True)
        # Build aligned values for both histograms
        real_values = [real_freq.get(p, 0) for p in sorted_particles]
        sampled_values = [sampled_freq.get(p, 0) for p in sorted_particles]
        
        fig, ax = plt.subplots(figsize=(21, 8), dpi=plotting.default_dpi)
        fig.suptitle("Normalized Particle Type Distributions")
        plotting.set_labels("Particle Type", "Frequency", in_fig=fig)
        plotting.plot_bar(real_values, label=f'Input ({model_name})', color=plotting.colors[0], normalized=normalized, use_log=use_log, in_ax=ax)
        plotting.plot_bar(sampled_values, label=f'Sampled ({model_name})', color=plotting.colors[1], normalized=normalized, use_log=use_log, in_ax=ax)
        x = range(len(sorted_particles))
        ax.set_xticks(x, sorted_particles, rotation=45, ha='right')
        ax.legend()
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, ax

class tables:
    """
    This class primary aggregates all the data from the model meta, config, training and metrics files into various useful formats.
    """
    
    model_metadata_columns          = ['vocab_size', 'max_sequence_length', 'num_train_tokens', 'num_val_tokens']
    model_config_columns            = ['batch_size', 'block_size', 'learning_rate', 'min_lr', 'lr_decay_iters', 'n_layer', 'n_head', 'n_embd', 'scheme', 'preparation_name']
    model_training_columns          = ['iters_trained', 'min_saved_train_loss', 'min_saved_val_loss']
    model_metrics_columns           = ['coverage', 'mmd', 'kpd_median', 'fpd_value', 'w1m_score', 'w1p_avg_eta', 'w1p_avg_phi', 'w1p_avg_pt']
    model_metrics_columns_verbose   = ['kpd_error', 'fpd_error', 'w1m_score_std', 'w1p_avg_eta_std', 'w1p_avg_phi_std', 'w1p_avg_pt_std']
    model_all_columns               = ['model_name'] + model_metadata_columns + model_config_columns + model_training_columns + model_metrics_columns
    model_all_columns_verbose       = ['model_name'] + model_metadata_columns + model_config_columns + model_training_columns + model_metrics_columns + model_metrics_columns_verbose
    
    @staticmethod
    def get_meta_data(model_name):
        meta_filename = pUtil.get_model_meta_filename(model_name)
        if not meta_filename.exists():
            return None
        
        with open(meta_filename, 'rb') as meta_file:
            meta_data = pickle.load(meta_file)
        
        return SimpleNamespace(
            vocab_size              = meta_data.get('vocab_size', np.nan),
            max_sequence_length     = meta_data.get('max_sequence_length', np.nan),
            num_train_tokens        = meta_data.get('num_train_tokens', np.nan),
            num_val_tokens          = meta_data.get('num_val_tokens', np.nan)
        )
    
    @staticmethod
    def get_config_data(model_name):
        config_filename = pUtil.get_model_config_filename(model_name)
        if not config_filename.exists():
            return None
        
        meta_data = tables.get_meta_data(model_name)
        
        with open(config_filename, 'r') as config_file:
            config_data = json.load(config_file)
        
        training_config = config_data.get('training_config', {})
        block_size = training_config.get('block_size', np.nan)
        context_events = training_config.get('context_events', np.nan)
        
        if block_size == np.nan:
            block_size = context_events * meta_data.max_sequence_length
        
        return SimpleNamespace(
            batch_size              = training_config.get('batch_size', np.nan),
            block_size              = training_config.get('block_size', np.nan),
            learning_rate           = training_config.get('learning_rate', np.nan),
            min_lr                  = training_config.get('min_lr', np.nan),
            lr_decay_iters          = training_config.get('lr_decay_iters', np.nan),
            n_layer                 = training_config.get('n_layer', np.nan),
            n_head                  = training_config.get('n_head', np.nan),
            n_embd                  = training_config.get('n_embd', np.nan),
            scheme                  = training_config.get('scheme', 'unknown'),
            preparation_name        = training_config.get('preparation_name', 'unknown')
        )

    @staticmethod
    def get_training_run_data(model_name, iterations_per_epoch=-1):
        training_log_filename = pUtil.get_training_dir(model_name) / "train_log_1.jsonl"
        
        running_data, checkpointed_data = [], []
        with open(training_log_filename) as training_log_file:
            for jline in training_log_file:
                jdata = json.loads(jline)
                if jdata.get("message") == "Training progress" and "iter" in jdata:
                    current_epochs_trained = 0 if jdata['iter'] == 0 else (jdata['iter'] / iterations_per_epoch)
                    running_data.append({'iter': jdata["iter"], 'epoch': current_epochs_trained, 'train_loss': jdata["train_loss"], 'val_loss': jdata["val_loss"]})
                elif jdata.get("message") == "Training progress: checking checkpoint conditions":
                    current_epochs_trained = 0 if jdata['step'] == 0 else (jdata['step'] / iterations_per_epoch)
                    checkpointed_data.append({'iter': jdata["step"], 'epoch': current_epochs_trained, 'train_loss': jdata["train_loss"], 'val_loss': jdata["val_loss"]})
        
        return SimpleNamespace(
            running_df = pd.DataFrame(running_data),
            checkpointed_df = pd.DataFrame(checkpointed_data)
        )
    
    @staticmethod
    def get_metrics(model_name):
        ret_dict = SimpleNamespace(**{
            'coverage': np.nan,
            'mmd': np.nan,
            'kpd_median': np.nan,
            'kpd_error': np.nan,
            'fpd_value': np.nan,
            'fpd_error': np.nan,
            'w1m_score': np.nan,
            'w1m_score_std': np.nan,
            'w1p_avg_eta': np.nan,
            'w1p_avg_phi': np.nan,
            'w1p_avg_pt': np.nan,
            'w1p_avg_eta_std': np.nan,
            'w1p_avg_phi_std': np.nan,
            'w1p_avg_pt_std': np.nan,
        })
        
        latest_sampling_dir = pUtil.get_latest_sampling_dir(model_name)
        metrics_results_filename = latest_sampling_dir / "metrics_results.json"
        
        if not latest_sampling_dir.exists():
            return ret_dict
        if not metrics_results_filename.exists():
            return ret_dict
        
        with open(metrics_results_filename, 'r') as metrics_file:
            metrics_data = json.load(metrics_file)
        
        return SimpleNamespace(**metrics_data)
    
    # Returns all important data for a model in a dictionary
    @staticmethod
    def get_all_data(model_name):
        meta_data = tables.get_meta_data(model_name)
        config_data = tables.get_config_data(model_name)
        metrics_data = tables.get_metrics(model_name)

        iterations_per_epoch = meta_data.num_train_tokens // (config_data.batch_size * config_data.block_size)
        
        # Training information
        training_run_data = tables.get_training_run_data(model_name, iterations_per_epoch)
        iters_trained = training_run_data.running_df['iter'].max()
        min_saved_val_loss_row = training_run_data.checkpointed_df.loc[training_run_data.checkpointed_df['val_loss'].idxmin()]
        
        training_run_data = SimpleNamespace(
            iters_trained           = iters_trained,
            min_saved_train_loss    = min_saved_val_loss_row['train_loss'],
            min_saved_val_loss      = min_saved_val_loss_row['val_loss'],
            running_df              = training_run_data.running_df,
            checkpointed_df         = training_run_data.checkpointed_df
        )
        
        return SimpleNamespace(**{'model_name': model_name, **vars(meta_data), **vars(config_data), **vars(metrics_data), **vars(training_run_data)})

    # Returns a DataFrame with all the important data for all models
    @staticmethod
    def get_default_df(model_names):
        columns = tables.model_all_columns
        model_data_list = [row for name in model_names if (row := tables.get_all_data(name)) is not None]
        model_data_df = pd.DataFrame(model_data_list, columns=columns)
        return model_data_df