
import json
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from particle import Particle
import vector

import jetnet
import jetnet.evaluation

import pUtil
from dictionary import Dictionary
import data_manager

# Custom formatter to force scientific notation for small numbers
def sci_notation(x, pos):
    return f'{x:.0e}'  # or '{:.1e}' for 1 decimal

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
    colors = ['blue', 'orange', 'purple', 'green', 'red', 'yellow', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    default_figsize = (21, 6)
    default_dpi = 300
    distributions_per_row = 3
    
    verbose_columns = ["pdgid", "e", "px", "py", "pz", "pt", "eta", "theta", "phi"]
    columns = ["num_particles", "pdgid", "e", "px", "py", "pz", "pt", "eta", "theta", "phi"]
    
    """
    Plotting training runs and distributions of leading particles.
    """
    
    # learning rate decay scheduler
    @staticmethod
    def _get_lr(it, lr_scheduler, warmup_iters, lr_decay_iters, min_lr, learning_rate, cycle_steps_mult, base_lr_decay_mult):
        if lr_scheduler == 'cosine_annealing_with_warmup':
            # 1) linear warmup for warmup_iters steps
            if it < warmup_iters:
                return learning_rate * it / warmup_iters
            # 2) if it > lr_decay_iters, return min learning rate
            if it >= lr_decay_iters:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return min_lr + coeff * (learning_rate - min_lr)
        elif lr_scheduler == 'cosine_with_warmup':
            # 1) linear warmup for warmup_iters steps
            if it < warmup_iters:
                return learning_rate * it / warmup_iters
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return min_lr + coeff * (learning_rate - min_lr)
        elif lr_scheduler == 'cosine_annealing_with_warm_restarts':
            # 1) linear warmup for warmup_iters steps
            if it < warmup_iters:
                return learning_rate * (it / warmup_iters)
            # Adjust iteration to account for warmup
            it -= warmup_iters
            # 2) Find current cycle and position in the cycle
            cycle = 0
            curr_cycle_len = lr_decay_iters
            iter_in_cycle = it
            while iter_in_cycle >= curr_cycle_len:
                iter_in_cycle -= curr_cycle_len
                cycle += 1
                curr_cycle_len = int(curr_cycle_len * cycle_steps_mult)
            # 3) Decay the base learning rate for the current cycle
            curr_base_lr = learning_rate * (base_lr_decay_mult ** cycle)
            # 4) Normalized progress within the cycle
            t = iter_in_cycle / curr_cycle_len
            # 5) Cosine annealing
            lr = min_lr + 0.5 * (curr_base_lr - min_lr) * (1 + math.cos(math.pi * t))
            return lr
        raise ValueError(f"Unknown lr_scheduler {lr_scheduler}")
    
    @staticmethod
    def plot_training_run(model_names, y_lim=None, x_lim=None, use_log=False, out_file=None, plot_lr_schedule=False):
        """
        Wrapper to plot a training run. Handles plotting lines and markers for training and validation loss.
        """
        
        if not isinstance(model_names, list):
            model_names = [model_names]
        
        # Set up plot
        fig, ax = plt.subplots(figsize=plotting.default_figsize, dpi=plotting.default_dpi)
        fig.suptitle(f'Training Progress for {model_names}')
        fig.supxlabel("Iteration")
        fig.supylabel("Loss")
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if use_log:
            ax.set_yscale('log')
            
        for idx, model_name in enumerate(model_names):
            # Parse model data
            model_data = tables.get_all_data(model_name)
            min_val_row = model_data.checkpointed_df.loc[model_data.checkpointed_df['val_loss'].idxmin()]
            final_row = model_data.running_df.iloc[-1]
            
            if plot_lr_schedule:
                config_filepath = pUtil.get_model_config_filepath(model_name)
                with open(config_filepath, 'r') as f:
                    model_config = json.load(f)
                    training_conf = model_config.get('training_config', {})
                    lr_scheduler = training_conf.get('lr_scheduler', 'cosine_annealing_with_warmup')
                    warmup_iters = training_conf .get('warmup_iters', 0)
                    lr_decay_iters = training_conf .get('lr_decay_iters', 0)
                    learning_rate = training_conf .get('learning_rate', 0)
                    min_lr = training_conf .get('min_lr', 0)
                    base_lr_decay_mult = training_conf .get('base_lr_decay_mult', 1)
                    cycle_steps_mult = training_conf .get('cycle_steps_mult', 1)
                iters = [i for i in range(model_data.running_df['iter'].max())]
                lrs = [plotting._get_lr(it, lr_scheduler, warmup_iters, lr_decay_iters, min_lr, learning_rate, cycle_steps_mult, base_lr_decay_mult) for it in iters]
                
                # Another y-axis for lr as otherwise it would be too small to see 
                lrax = ax.twinx()
                lrax.plot(iters, lrs, label=f'Learning rate', color="magenta", linestyle='solid', linewidth=1)
                lrax.tick_params(axis='y', labelcolor="magenta")
                # Decimal can be confusing so we switch to scientific
                lrax.yaxis.set_major_formatter(FuncFormatter(sci_notation))

            # Do plot
            ax.plot(model_data.running_df['iter'], model_data.running_df['train_loss'], label=f'Training Loss ({model_name})', color=plotting.colors[idx], linestyle='solid', linewidth=0.5)
            ax.plot(model_data.running_df['iter'], model_data.running_df['val_loss'], label=f'Validation Loss ({model_name})', color=plotting.colors[idx], linestyle='dashed', linewidth=0.5)
            ax.scatter(min_val_row['iter'], min_val_row['train_loss'], label=f'Min Saved Train Loss ({model_name}; {min_val_row["train_loss"]:.4f})', color=plotting.colors[idx], marker='s')
            ax.scatter(min_val_row['iter'], min_val_row['val_loss'], label=f'Min Saved Val Loss ({model_name}; {min_val_row["val_loss"]:.4f})', color=plotting.colors[idx], marker='o')
            ax.annotate(model_name, xy=(final_row['iter'], final_row['val_loss']), xytext=(final_row['iter'] * 1.005, final_row['val_loss'] - 0.02), fontsize=9, color=plotting.colors[idx])

        # Final touches and show and/or save
        fig.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
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
        
        # Legacy NO NOT DELETE. Using the bin counts as specified in the tokenization is ideal but not always possible.
        
        # def get_bin_count(type_str):
        #     step_size = dictionary.token_step_size(type_str)
        #     if step_size == 0:
        #         return 0
        #     if type_str in ['eta', 'theta', 'phi']:
        #         step_size = 0.05
        #     return int(dictionary.token_range(type_str) // step_size)
        
        # For now, I have replaced those with hard coded values.
        # bin_settings = {
        #     "num_particles": { "min": -0.5,                          "max": 50.5,                          "bins": 51 },
        #     "e":             { "min": 0,                             "max": 35000,                         "bins": 350 },
        #     "px":            { "min": -5000,                         "max": 35000,                         "bins": 400 },
        #     "py":            { "min": -5000,                         "max": 35000,                         "bins": 400 },
        #     "pz":            { "min": -5000,                         "max": 35000,                         "bins": 400 },
        #     "eta":           { "min": dictionary.token_min('eta'),   "max": dictionary.token_max('eta'),   "bins": get_bin_count('eta') },
        #     "theta":         { "min": dictionary.token_min('theta'), "max": dictionary.token_max('theta'), "bins": get_bin_count('theta') },
        #     "phi":           { "min": dictionary.token_min('phi'),   "max": dictionary.token_max('phi'),   "bins": get_bin_count('phi') },
        #     "pt":            { "min": dictionary.token_min('pt'),    "max": dictionary.token_max('pt'),    "bins": get_bin_count('pt') },
        # }
        
        theta_min = 0 if dictionary.token_min('theta') == 0 else dictionary.token_min('theta')
        theta_max = np.pi if dictionary.token_max('theta') == 0 else dictionary.token_max('theta')
        phi_min = -np.pi if dictionary.token_min('phi') == 0 else dictionary.token_min('phi')
        phi_max = np.pi if dictionary.token_max('phi') == 0 else dictionary.token_max('phi')
        
        columns = ["num_particles", "pdgid", "e", "px", "py", "pz", "pt", "eta", "theta", "phi"]
        bin_settings = {
            "num_particles": { "min": -0.5,                          "max": 50.5,                          "bins": 51 },
            "e":             { "min": 0,                             "max": 35000,                         "bins": 350 },
            "px":            { "min": -5000,                         "max": 35000,                         "bins": 400 },
            "py":            { "min": -5000,                         "max": 35000,                         "bins": 400 },
            "pz":            { "min": -5000,                         "max": 35000,                         "bins": 400 },
            "eta":           { "min": dictionary.token_min('eta'),   "max": dictionary.token_max('eta'),   "bins": 400 },
            "theta":         { "min": theta_min,                     "max": theta_max,                     "bins": 400 },
            "phi":           { "min": phi_min,                       "max": phi_max,                       "bins": 400 },
            "pt":            { "min": dictionary.token_min('pt'),    "max": dictionary.token_max('pt'),    "bins": 400 },
        }

        # real_df = pd.read_csv(real_leading_test_particles_filename, sep=" ", names=columns, engine="c", header=None)
        # sampled_df = pd.read_csv(sampled_leading_particles_filename, sep=" ", names=columns, engine="c", header=None)
        return bin_settings #, real_df, sampled_df

    @staticmethod
    def plot_distribution_leading(model_names, column_name=None, normalized=False, use_log=False, out_file=None):
        assert column_name in plotting.verbose_columns, f"Invalid column name: {column_name}. Must be one of {plotting.verbose_columns}."
        unit = ''
        if column_name in ['e', 'pt', 'px', 'py', 'pz']:
            unit = '(MeV)'
        elif column_name in ['eta', 'theta', 'phi']:
            unit = '(angular)'
        
        # Set up plot
        num_horizontal, num_vertical = min(len(model_names), plotting.distributions_per_row), (math.ceil(1 / plotting.distributions_per_row))
        fig, axes = plt.subplots(num_vertical, num_horizontal, figsize=(8 * num_horizontal, 6 * num_vertical), sharex=False, sharey=True, dpi=plotting.default_dpi)
        fig.suptitle(f'{column_name} Distribution for Leading Particles {"(Normalized)" if normalized else ""}')
        fig.supxlabel(f'{column_name} {unit}')
        fig.supylabel('log(Frequency)' if use_log else 'Frequency')
        
        for ax, model_name in zip([axes], model_names):
            # Parse data
            bin_settings = plotting._get_common_data(model_name)
            range = (bin_settings[column_name]['min'], bin_settings[column_name]['max'])
            n_bins = bin_settings[column_name]['bins']
            
            relevant_column_pos = plotting.verbose_columns.index(column_name)
            real_verbose_data = data_manager.load_verbose_dataset(pUtil.get_model_preparation_dir(model_name) / 'real_verbose_test_particles.csv', pad_token = np.nan)
            all_instances_of_this_column_real = []
            for event in real_verbose_data:
                secondaries = event[1:]
                # Find index of particle with the highest energy
                leading_particle_idx = np.nanargmax(secondaries[:, 1])
                leading_particle = secondaries[leading_particle_idx]
                all_instances_of_this_column_real.append(leading_particle[relevant_column_pos])
            
            sampled_verbose_data = data_manager.load_verbose_dataset(pUtil.get_latest_sampling_dir(model_name) / 'untokenized_samples_verbose.csv', pad_token = np.nan)
            all_instances_of_this_column_sampled = []
            for event in sampled_verbose_data:
                secondaries = event[1:]
                # Find index of particle with the highest energy
                leading_particle_idx = np.nanargmax(secondaries[:, 1])
                leading_particle = secondaries[leading_particle_idx]
                all_instances_of_this_column_sampled.append(leading_particle[relevant_column_pos])
            
            # Do plot
            ax.set_yscale('log' if use_log else 'linear')
            ax.hist(all_instances_of_this_column_real, range=range, bins=n_bins, density=normalized, label=f'Input ({model_name})', color=plotting.colors[0], alpha=0.7)
            ax.hist(all_instances_of_this_column_sampled, range=range, bins=n_bins, density=normalized, label=f'Sampled ({model_name})', color=plotting.colors[1], alpha=0.7)
            
        # Finishing touches and show and/or save
        fig.legend()
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, ax
    
    @staticmethod
    def plot_distribution_all(model_names, column_name=None, normalized=False, use_log=False, out_file=None):
        unit = ''
        if column_name in ['e', 'pt', 'px', 'py', 'pz']:
            unit = '(MeV)'
        elif column_name in ['eta', 'theta', 'phi']:
            unit = '(angular)'
        
        # Set up plot
        num_horizontal, num_vertical = min(len(model_names), plotting.distributions_per_row), (math.ceil(1 / plotting.distributions_per_row))
        fig, axes = plt.subplots(num_vertical, num_horizontal, figsize=(8 * num_horizontal, 6 * num_vertical), sharex=False, sharey=True, dpi=plotting.default_dpi)
        fig.suptitle(f'{column_name} Distribution for All Outgoing Particles {"(Normalized)" if normalized else ""}')
        fig.supxlabel(f'{column_name} {unit}')
        fig.supylabel('log(Frequency)' if use_log else 'Frequency')
        
        for ax, model_name in zip([axes], model_names):
            # Parse data
            bin_settings = plotting._get_common_data(model_name)
            range = (bin_settings[column_name]['min'], bin_settings[column_name]['max'])
            n_bins = bin_settings[column_name]['bins']
            
            relevant_column_pos = plotting.verbose_columns.index(column_name)
            real_verbose_data = data_manager.load_verbose_dataset(pUtil.get_model_preparation_dir(model_name) / 'real_verbose_test_particles.csv', pad_token = np.nan)
            all_instances_of_this_column_real = []
            for event in real_verbose_data:
                secondaries = event[1:]
                for particle in secondaries:
                    if not np.isnan(particle[relevant_column_pos]):
                        all_instances_of_this_column_real.append(particle[relevant_column_pos])
            
            sampled_verbose_data = data_manager.load_verbose_dataset(pUtil.get_latest_sampling_dir(model_name) / 'untokenized_samples_verbose.csv', pad_token = np.nan)
            all_instances_of_this_column_sampled = []
            for event in sampled_verbose_data:
                secondaries = event[1:]
                for particle in secondaries:
                    if not np.isnan(particle[relevant_column_pos]):
                        all_instances_of_this_column_sampled.append(particle[relevant_column_pos])
            
            # Do plot
            ax.set_yscale('log' if use_log else 'linear')
            ax.hist(all_instances_of_this_column_real, range=range, bins=n_bins, density=normalized, label=f'Input ({model_name})', color=plotting.colors[0], alpha=0.7)
            ax.hist(all_instances_of_this_column_sampled, range=range, bins=n_bins, density=normalized, label=f'Sampled ({model_name})', color=plotting.colors[1], alpha=0.7)
            
        # Finishing touches and show and/or save
        fig.legend()
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, ax
    
    @staticmethod
    def plot_pdgid_distribution_leading(model_names, normalized=False, use_log=False, out_file=None):
        model_name = model_names[0]
        
        relevant_column_pos = plotting.verbose_columns.index('pdgid')
        real_verbose_data = data_manager.load_verbose_dataset(pUtil.get_model_preparation_dir(model_name) / 'real_verbose_test_particles.csv', pad_token = np.nan)
        all_instances_of_this_column_real = []
        for event in real_verbose_data:
            secondaries = event[1:]
            # Find index of particle with the highest energy
            leading_particle_idx = np.nanargmax(secondaries[:, 1])
            leading_particle = secondaries[leading_particle_idx]
            all_instances_of_this_column_real.append(leading_particle[relevant_column_pos])
        
        sampled_verbose_data = data_manager.load_verbose_dataset(pUtil.get_latest_sampling_dir(model_name) / 'untokenized_samples_verbose.csv', pad_token = np.nan)
        all_instances_of_this_column_sampled = []
        for event in sampled_verbose_data:
            secondaries = event[1:]
            # Find index of particle with the highest energy
            leading_particle_idx = np.nanargmax(secondaries[:, 1])
            leading_particle = secondaries[leading_particle_idx]
            all_instances_of_this_column_sampled.append(leading_particle[relevant_column_pos])
            
        # real_pdgids, sampled_pdgids = real_df['pdgid'], sampled_df['pdgid']
        real_freq, sampled_freq = Counter(all_instances_of_this_column_real), Counter(all_instances_of_this_column_sampled)
        
        # Union of all particle labels from both histograms
        all_particles = sorted(set(real_freq.keys()).union(sampled_freq.keys()))
        # Sorting them by frequency in real leading particles to ensure a legible plot
        sorted_particles = sorted(all_particles, key=lambda p: real_freq[p], reverse=True)
        # Build aligned values for both histograms
        real_values = [real_freq.get(p, 0) for p in sorted_particles]
        sampled_values = [sampled_freq.get(p, 0) for p in sorted_particles]
        
        # Set up plot
        fig, ax = plt.subplots(figsize=plotting.default_figsize, dpi=plotting.default_dpi)
        fig.suptitle(f'Particle Type Distributions {"(Normalized)" if normalized else ""}')
        fig.supxlabel('Particle Type')
        fig.supylabel('log(Frequency)' if use_log else 'Frequency')
        if use_log:
            ax.set_yscale('log')
        if normalized:
            total_real = sum(real_values)
            total_sampled = sum(sampled_values)
            real_values = [v / total_real for v in real_values]
            sampled_values = [v / total_sampled for v in sampled_values]
        
        # Do plot
        ax.bar(range(len(sorted_particles)), real_values, label=f'Input ({model_name})', color=plotting.colors[0], alpha=0.7, width=0.9, align='center')
        ax.bar(range(len(sorted_particles)), sampled_values, label=f'Sampled ({model_name})', color=plotting.colors[1], alpha=0.7, width=0.9, align='center')
        ax.set_xticks(range(len(sorted_particles)), sorted_particles, rotation=45, ha='right')
        
        # Finishing touches and show and/or save
        fig.legend()
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
        
        # Set up plot
        fig, ax = plt.subplots(figsize=plotting.default_figsize, dpi=plotting.default_dpi)
        fig.suptitle(f'Particle Type Distributions {"(Normalized)" if normalized else ""}')
        fig.supxlabel('Particle Type')
        fig.supylabel('log(Frequency)' if use_log else 'Frequency')
        if use_log:
            ax.set_yscale('log')
        if normalized:
            total_real = sum(real_values)
            total_sampled = sum(sampled_values)
            real_values = [v / total_real for v in real_values]
            sampled_values = [v / total_sampled for v in sampled_values]
        
        # Do plot
        ax.bar(range(len(sorted_particles)), real_values, label=f'Input ({model_name})', color=plotting.colors[0], alpha=0.7, width=0.9, align='center')
        ax.bar(range(len(sorted_particles)), sampled_values, label=f'Sampled ({model_name})', color=plotting.colors[1], alpha=0.7, width=0.9, align='center')
        ax.set_xticks(range(len(sorted_particles)), sorted_particles, rotation=45, ha='right')
        
        # Finishing touches and show and/or save
        fig.legend()
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, ax

    def plot_energy_conservation(model_names, normalized=False, use_log=False, out_file=None):
        MASS_CARBON = 931.5 * 12 # MeV
        generated_samples_data = data_manager.load_geant4_dataset(pUtil.get_latest_sampling_dir(model_names[0]) / 'untokenized_samples.csv', pad_token=0.0)

        # Perform computation
        computed_data = np.full(shape=(len(generated_samples_data), 2), fill_value=np.nan)
        for idx, event in enumerate(generated_samples_data):
            # Input vector
            primary_particle = Particle.from_pdgid(event[0][0])
            in_particle_vec = vector.obj(mass=primary_particle.mass, px=event[0][2], py=event[0][3], pz=event[0][4])
            in_material_vec = vector.obj(mass=MASS_CARBON, px=0.0, py=0.0, pz=0.0)
            in_vec = in_particle_vec + in_material_vec
            
            # Output vector
            out_vec = vector.obj(mass=0.0, px=0.0, py=0.0, pz=0.0)
            for particle in event[1:]:
                if particle[0] == 0.0:
                    continue
                i_particle = Particle.from_pdgid(int(particle[0]))
                i_vec = vector.obj(mass=i_particle.mass, px=particle[2], py=particle[3], pz=particle[4])
                out_vec += i_vec
            
            computed_data[idx] = in_vec.e, out_vec.e
        
        # Set up plot
        num_horizontal, num_vertical = min(len(model_names), plotting.distributions_per_row), (math.ceil(1 / plotting.distributions_per_row))
        fig, axes = plt.subplots(num_vertical, num_horizontal, figsize=(8 * num_horizontal, 6 * num_vertical), sharex=False, sharey=True, dpi=plotting.default_dpi)
        fig.suptitle(f'Energy conservation for {model_names}')
        fig.supxlabel(f'Delta Energy (MeV)')
        fig.supylabel('Frequency')
        
        for ax, model_name in zip([axes], model_names):
            # Do plot
            ax.set_yscale('log' if use_log else 'linear')
            ax.hist(computed_data[:,0], bins=50, density=normalized, label=f'Input ({model_name})', color=plotting.colors[0], alpha=0.7)
            ax.hist(computed_data[:,1], bins=50, density=normalized, label=f'Sampled ({model_name})', color=plotting.colors[1], alpha=0.7)

        # Finishing touches and show and/or save
        fig.legend()
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
    def plot_num_particles(model_names, normalized=False, use_log=False, out_file=None):
        model_name = model_names[0]
        
        MASS_CARBON = 931.5 * 12 # MeV
        
        real_verbose_data = data_manager.load_verbose_dataset(pUtil.get_model_preparation_dir(model_name) / 'real_verbose_test_particles.csv', pad_token = np.nan)
        real_num_particle_data = np.full(shape=(len(real_verbose_data)), fill_value=np.nan)
        for idx, event in enumerate(real_verbose_data):
            secondaries = event[1:]
            secondaries = [secondary for secondary in secondaries if not np.isnan(secondary[0])]
            num_secondaries = len(secondaries)
            real_num_particle_data[idx] = num_secondaries
        
        generated_samples_data = data_manager.load_geant4_dataset(pUtil.get_latest_sampling_dir(model_names[0]) / 'untokenized_samples.csv', pad_token = np.nan)
        sampled_num_particle_data = np.full(shape=(len(generated_samples_data)), fill_value=np.nan)
        for idx, event in enumerate(generated_samples_data):
            secondaries = event[1:]
            secondaries = [secondary for secondary in secondaries if not np.isnan(secondary[0])]
            num_secondaries = len(secondaries)
            sampled_num_particle_data[idx] = num_secondaries

        # Set up plot
        num_horizontal, num_vertical = min(len(model_names), plotting.distributions_per_row), (math.ceil(1 / plotting.distributions_per_row))
        fig, axes = plt.subplots(num_vertical, num_horizontal, figsize=(8 * num_horizontal, 6 * num_vertical), sharex=False, sharey=True, dpi=plotting.default_dpi)
        fig.suptitle(f'Num particles per event for {model_names}')
        fig.supxlabel(f'Number of particles')
        fig.supylabel('Frequency')
        
        for ax, model_name in zip([axes], model_names):
            # Do plot
            ax.set_yscale('log' if use_log else 'linear')
            ax.hist(real_num_particle_data, range=(0, 50), bins=50, density=normalized, label=f'Input ({model_name})', color=plotting.colors[0], alpha=0.7)
            ax.hist(sampled_num_particle_data, range=(0, 50), bins=50, density=normalized, label=f'Sampled ({model_name})', color=plotting.colors[1], alpha=0.7)
            
        # Finishing touches and show and/or save
        fig.legend()
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()

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
        meta_filename = pUtil.get_model_meta_filepath(model_name)
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
        config_filename = pUtil.get_model_config_filepath(model_name)
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
        model_data_list = [row for name in model_names if (row := vars(tables.get_all_data(name))) is not None]
        model_data_df = pd.DataFrame(model_data_list, columns=columns)
        return model_data_df