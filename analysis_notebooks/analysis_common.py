
import json
import math
import pickle
import numpy as np
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import Counter

parent_dir = Path().resolve().parent
sys.path.insert(0, str(parent_dir))

from dictionary import Dictionary
import pUtil

def get_model_metadata_columns():
    return ['vocab_size', 'max_sequence_length', 'num_train_tokens', 'num_val_tokens']

def get_model_config_columns():
    return ['batch_size', 'block_size', 'learning_rate', 'min_lr', 'lr_decay_iters', 'n_layer', 'n_head', 'n_embd', 'scheme', 'preparation_name']

def get_model_training_columns():
    return ['iters_trained', 'min_saved_train_loss', 'min_saved_val_loss']

def get_model_metrics_columns():
    return ['coverage', 'mmd', 'kpd_median', 'kpd_error', 'fpd_value', 'fpd_error', 'w1m_score', 'w1m_score_std', 'w1p_avg_eta', 'w1p_avg_phi', 'w1p_avg_pt', 'w1p_avg_eta_std', 'w1p_avg_phi_std', 'w1p_avg_pt_std']

def get_model_all_columns():
    return ['model_name'] + get_model_metadata_columns() + get_model_config_columns() + get_model_training_columns() + get_model_metrics_columns()

# iterations_per_epoch must be provided for epoch calculations
def get_training_data_for_model(model_name, iterations_per_epoch=-1):
    training_log_filename = pUtil.get_training_dir(model_name) / "train_log_1.jsonl"
    
    running_data, saved_data = [], []
    with open(training_log_filename) as training_log_file:
        for jline in training_log_file:
            jdata = json.loads(jline)
            if jdata.get("message") == "Training progress" and "iter" in jdata:
                current_epochs_trained = 0 if jdata['iter'] == 0 else (jdata['iter'] / iterations_per_epoch)
                running_data.append({'iter': jdata["iter"], 'epoch': current_epochs_trained, 'train_loss': jdata["train_loss"], 'val_loss': jdata["val_loss"]})
            elif jdata.get("message") == "Training progress: checking checkpoint conditions":
                current_epochs_trained = 0 if jdata['step'] == 0 else (jdata['step'] / iterations_per_epoch)
                saved_data.append({'iter': jdata["step"], 'epoch': current_epochs_trained, 'train_loss': jdata["train_loss"], 'val_loss': jdata["val_loss"]})
    
    return pd.DataFrame(running_data), pd.DataFrame(saved_data)

def get_metrics_for_model(model_name):
    ret_dict = {
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
    }
    
    latest_sampling_dir = pUtil.get_latest_sampling_dir(model_name)
    if not latest_sampling_dir.exists():
        return ret_dict
    
    metrics_results_filename = latest_sampling_dir / "metrics_results.json"
    if not metrics_results_filename.exists():
        return ret_dict
    
    with open(metrics_results_filename, 'r') as metrics_file:
        metrics_data = json.load(metrics_file)
    
    return metrics_data

# Returns all important data for a model in a dictionary
def get_all_data_for_model(model_name):
    meta_filename = pUtil.get_model_meta_filename(model_name)
    config_filename = pUtil.get_model_config_filename(model_name)
    
    if not meta_filename.exists():
        return
    
    # Metadata (dataset information)
    with open(meta_filename, 'rb') as meta_file:
        meta_data = pickle.load(meta_file)
        vocab_size = meta_data["vocab_size"]
        max_sequence_length = meta_data['max_sequence_length']
        num_train_tokens = meta_data['num_train_tokens']
        num_val_tokens = meta_data['num_val_tokens']

    # Configuration information
    with open(config_filename, 'r') as config_file:
        config = json.load(config_file)
        training_config = config.get('training_config', {})
        batch_size = training_config.get('batch_size', -1)
        block_size = training_config.get('block_size', -1)
        context_events = training_config.get('context_events', -1)
        learning_rate = training_config.get('learning_rate', -1)
        min_lr = training_config.get('min_lr', -1)
        lr_decay_iters = training_config.get('lr_decay_iters', -1)
        n_layer = training_config.get('n_layer', -1)
        n_head = training_config.get('n_head', -1)
        n_embd = training_config.get('n_embd', -1)
        scheme = training_config.get('scheme', 'standard')
        preparation_name = training_config.get('preparation_name', 'unknown')
        
        if block_size == -1:
            block_size = context_events * max_sequence_length
        
        iterations_per_epoch = num_train_tokens // (batch_size * block_size)
        
    # Metrics information
    metrics_data = get_metrics_for_model(model_name)
    
    # Training information
    running_df, saved_df = get_training_data_for_model(model_name, iterations_per_epoch)      
    
    iters_trained = running_df['iter'].max()
    min_saved_val_loss_row = saved_df.loc[saved_df['val_loss'].idxmin()]
    
    return {
        'model_name': model_name,
        # Metadata (dataset information)
        'vocab_size': vocab_size,
        'max_sequence_length': max_sequence_length,
        'num_train_tokens': num_train_tokens,
        'num_val_tokens': num_val_tokens,
        # Configuration information
        'batch_size': batch_size,
        'block_size': block_size,
        'learning_rate': learning_rate,
        'min_lr': min_lr,
        'lr_decay_iters': lr_decay_iters,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'scheme': scheme,
        'preparation_name': preparation_name,
        # Training information
        "iters_trained": iters_trained,
        'min_saved_train_loss': min_saved_val_loss_row['train_loss'],
        'min_saved_val_loss': min_saved_val_loss_row['val_loss'],
        # Metrics
        'coverage': metrics_data.get('coverage', np.nan),
        'mmd': metrics_data.get('mmd', np.nan),
        'kpd_median': metrics_data.get('kpd_median', np.nan),
        'kpd_error': metrics_data.get('kpd_error', np.nan),
        'fpd_value': metrics_data.get('fpd_value', np.nan),
        'fpd_error': metrics_data.get('fpd_error', np.nan),
        'w1m_score': metrics_data.get('w1m_score', np.nan),
        'w1m_score_std': metrics_data.get('w1m_score_std', np.nan),
        'w1p_avg_eta': metrics_data.get('w1p_avg_eta', np.nan),
        'w1p_avg_phi': metrics_data.get('w1p_avg_phi', np.nan),
        'w1p_avg_pt': metrics_data.get('w1p_avg_pt', np.nan),
        'w1p_avg_eta_std': metrics_data.get('w1p_avg_eta_std', np.nan),
        'w1p_avg_phi_std': metrics_data.get('w1p_avg_phi_std', np.nan),
        'w1p_avg_pt_std': metrics_data.get('w1p_avg_pt_std', np.nan)
    }

# Returns a DataFrame with all the important data for all models
def get_default_df(model_names):
    columns = get_model_all_columns()
    model_data_list = [row for name in model_names if (row := get_all_data_for_model(name)) is not None]
    model_data_df = pd.DataFrame(model_data_list, columns=columns)
    return model_data_df

# ---------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------

def plot_model_train_data(model_name, ax=None, use_epochs=True, y_lim=None, use_log_scale=True):
    """
    Plots training and validation loss for a single model.
    """
    model_data = get_all_data_for_model(model_name)
    iterations_per_epoch = model_data['num_train_tokens'] // (model_data['batch_size'] * model_data['block_size'])

    running_df, saved_df = get_training_data_for_model(model_name, iterations_per_epoch)
    metric = 'epoch' if use_epochs else 'iter'

    min_val_row = saved_df.loc[saved_df['val_loss'].idxmin()]

    ax = ax or plt
    
    # Plot training and validation loss
    train_line, = ax.plot(running_df[metric], running_df['train_loss'], label=f'Training Loss ({model_name})', linewidth=0.5)
    val_line, = ax.plot(running_df[metric], running_df['val_loss'], label=f'Validation Loss ({model_name})', color=train_line.get_color(), linestyle='--', linewidth=0.5)

    # Mark the point of minimum validation loss
    ax.scatter(min_val_row[metric], min_val_row['train_loss'], label=f'Min Train Loss ({model_name}; {min_val_row["train_loss"]:.4f})', color=train_line.get_color(), edgecolors='black')
    ax.scatter(min_val_row[metric], min_val_row['val_loss'], label=f'Min Val Loss ({model_name}; {min_val_row["val_loss"]:.4f})', color=train_line.get_color(), edgecolors='black', marker='s', s=50)

    # Annotate the line with the model name
    final_row = running_df.iloc[-1]
    ax.annotate(model_name, xy=(final_row['iter'], final_row['val_loss']), xytext=(final_row['iter'] * 1.005, final_row['val_loss'] - 0.02), fontsize=9, color=val_line.get_color())

    # Format axes if it's a subplot
    if ax is not plt:
        ax.set_xlabel(metric.capitalize())
        ax.set_ylabel('Loss')
        ax.set_title(model_name)
        if y_lim:
            ax.set_ylim(y_lim)
        if use_log_scale:
            ax.set_yscale('log', base=10)
        ax.legend()
    
def plot_train_graphs(models_to_compare, juxtaposed=True, use_epochs=True, y_lim=None, use_log_scale=True):
    """
    Plots training graphs for multiple models, either side-by-side or overlaid.
    """
    num_models = len(models_to_compare)
    share_axes = juxtaposed and num_models > 1

    if share_axes:
        cols = min(num_models, 3)
        rows = math.ceil(num_models / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), sharex=False, sharey=True)
        axes = np.ravel(axes)
    else:
        fig, axes = plt.figure(figsize=(15, 6)), [None] * num_models

    for model_name, ax in zip(models_to_compare, axes):
        plot_model_train_data(model_name, ax=ax, use_epochs=use_epochs, y_lim=y_lim, use_log_scale=use_log_scale)

    # Add title and format global figure
    full_title = f"Training Progress for {', '.join(models_to_compare)}"
    if share_axes:
        fig.suptitle(full_title)
        plt.tight_layout()
    else:
        plt.title("\n".join(textwrap.wrap(full_title, width=60)))
        if use_log_scale:
            plt.yscale('log', base=10)
        plt.xlabel('Iteration' if not use_epochs else 'Epoch')
        plt.ylabel('Loss')
        if y_lim:
            plt.ylim(y_lim)
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

columns = ["num_particles", "pdgid", "e", "px", "py", "pz", "pt", "eta", "theta", "phi"]

def get_common_data(model_name):
    dictionary_filename = pUtil.get_model_preparation_dir(model_name) / 'dictionary.json'
    real_leading_test_particles_filename = pUtil.get_model_preparation_dir(model_name) / 'real_leading_test_particles.csv'
    sampled_leading_particles_filename = pUtil.get_latest_sampling_dir(model_name) / 'sampled_leading_particles.csv'
    
    dictionary = Dictionary(dictionary_filename)
        
    def get_bin_count(type_str):
        step_size = dictionary.token_step_size(type_str)
        if step_size == 0:
            return 0
        return int(dictionary.token_range(type_str) // step_size)
    
    # Convenience dictionary definitions
    p_bin_count = int(dictionary.token_range('e') // 1000)
    e_bin_count = get_bin_count('e')
    eta_bin_count = get_bin_count('eta')
    theta_bin_count = get_bin_count('theta')
    phi_bin_count = get_bin_count('phi')
    pt_bin_count = get_bin_count('pt')
    
    columns = ["num_particles", "pdgid", "e", "px", "py", "pz", "pt", "eta", "theta", "phi"]
    bin_settings = {
        "num_particles": { "min": 0,                             "max": 50,                            "bins": 50 },
        "e":             { "min": dictionary.token_min('e'),     "max": dictionary.token_max('e'),     "bins": e_bin_count },
        "px":            { "min": dictionary.token_min('e'),     "max": dictionary.token_max('e'),     "bins": p_bin_count },
        "py":            { "min": dictionary.token_min('e'),     "max": dictionary.token_max('e'),     "bins": p_bin_count },
        "pz":            { "min": dictionary.token_min('e'),     "max": dictionary.token_max('e'),     "bins": p_bin_count },
        "eta":           { "min": dictionary.token_min('eta'),   "max": dictionary.token_max('eta'),   "bins": eta_bin_count },
        "theta":         { "min": dictionary.token_min('theta'), "max": dictionary.token_max('theta'), "bins": theta_bin_count },
        "phi":           { "min": dictionary.token_min('phi'),   "max": dictionary.token_max('phi'),   "bins": phi_bin_count },
        "pt":            { "min": dictionary.token_min('pt'),    "max": dictionary.token_max('pt'),    "bins": pt_bin_count },
    }

    real_df = pd.read_csv(real_leading_test_particles_filename, sep=" ", names=columns, engine="c", header=None)
    sampled_df = pd.read_csv(sampled_leading_particles_filename, sep=" ", names=columns, engine="c", header=None)
    return bin_settings, real_df, sampled_df

def plot_pdgid_distributions(model_name, ax=None, use_log_scale=False):
    dictionary_filename = pUtil.get_model_preparation_dir(model_name) / 'dictionary.json'
    dictionary = Dictionary(dictionary_filename)
    
    _, real_df, sampled_df = get_common_data(model_name)
    
    # PDGID is a fundamentally different plot, so we do it first here.
    read_pdgids = real_df['pdgid'].astype(str)
    sampled_pdgids = sampled_df['pdgid'].astype(str)
    real_freq = Counter(read_pdgids)
    sampled_freq = Counter(sampled_pdgids)
    real_total = sum(real_freq.values())
    sampled_total = sum(sampled_freq.values())
    real_normalized = {dictionary.particles_id.get(pid, pid): count / real_total for pid, count in real_freq.items()}
    sampled_normalized = {dictionary.particles_id.get(pid, pid): count / sampled_total for pid, count in sampled_freq.items()}
    
    # Union of all particle labels from both histograms
    all_particles = sorted(set(real_normalized.keys()).union(sampled_normalized.keys()))
    total_freq = {p: real_normalized.get(p, 0) + sampled_normalized.get(p, 0) for p in all_particles}
    
    sorted_particles = sorted(all_particles, key=lambda p: total_freq[p], reverse=True)
    
    # Build aligned values for both histograms
    real_values = [real_normalized.get(p, 0) for p in sorted_particles]
    sampeld_values = [sampled_normalized.get(p, 0) for p in sorted_particles]
    
    # Plotting
    ax = ax or plt
    x = range(len(sorted_particles))
    ax.bar(x, real_values, label=f'Input ({model_name})', color='blue', alpha=0.7)
    ax.bar(x, sampeld_values, label=f'Sampled ({model_name})', color='orange', alpha=0.7)
    if ax is not plt:
        if use_log_scale:
            ax.set_yscale('log', base=10)
        ax.set_xticks(x, sorted_particles, rotation=45, ha='right')
        ax.set_xlabel("Particle Type")
        ax.set_ylabel("Normalized Frequency")
        ax.set_title("Normalized Particle Type Distributions")
    ax.legend()

def plot_model_distributions(model_name, column_name, ax=None, use_log_scale=False):
    bin_settings, real_df, sampled_df = get_common_data(model_name)
    
    min_val = bin_settings[column_name]['min']
    max_val = bin_settings[column_name]['max']
    bins = bin_settings[column_name]['bins']
    
    if bins == 0:
        return
    
    real_df_weights = np.ones_like(real_df[column_name]) / len(real_df[column_name])
    sampled_df_weights = np.ones_like(sampled_df[column_name]) / len(sampled_df[column_name])
    
    ax = ax or plt
    ax.hist(real_df[column_name], bins=bins, weights=real_df_weights, range=(min_val, max_val), alpha=0.7, color="blue", label=f'Input ({model_name})')
    ax.hist(sampled_df[column_name], bins=bins, weights=sampled_df_weights, range=(min_val, max_val), alpha=0.7, color="orange", label=f'Sampled ({model_name})')
    if ax is not plt:
        unit = ''
        if column_name in ['e', 'px', 'py', 'pz', 'pt']:
            unit = '(MeV)'
        elif column_name in ['eta', 'theta', 'phi']:
            unit = '(angular)'
        if use_log_scale:
            ax.set_yscale('log', base=10)
        ax.set_xlabel(f'{column_name} {unit}')
        ax.set_ylabel('Frequency (Normalized)')
        ax.set_title(f'{model_name}')
        ax.legend()
    
def compare_pdgid_distributions(models_to_compare, juxtaposed=True, dists_per_row=3, use_log_scale=False):
    if juxtaposed:
        num_horizontal, num_vertical = min(len(models_to_compare), dists_per_row), (math.ceil(len(models_to_compare) / dists_per_row))
        figure, axes = plt.subplots(num_vertical, num_horizontal, figsize=(8 * num_horizontal, 6 * num_vertical), sharex=False, sharey=True)
        if len(models_to_compare) == 1:
            axes = [axes]
        axes = np.atleast_1d(axes).flatten()
        for model_name, ax in zip(models_to_compare, axes):
            plot_pdgid_distributions(model_name, ax=ax, use_log_scale=use_log_scale)
        figure.suptitle(f'Particle Type Distributions for {", ".join(models_to_compare)}')
        plt.tight_layout()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
    else:
        plt.figure(figsize=(15, 6))
        if use_log_scale:
            plt.yscale('log', base=10)
        for model_name in models_to_compare:
            plot_pdgid_distributions(model_name, use_log_scale=use_log_scale)
        plt.title(f'Particle Type Distributions for {", ".join(models_to_compare)}')
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

def compare_distributions(models_to_compare, column_name, juxtaposed=True, dists_per_row=3, use_log_scale=False):
    if column_name == 'pdgid':
        compare_pdgid_distributions(models_to_compare, juxtaposed=juxtaposed, dists_per_row=dists_per_row, use_log_scale=use_log_scale)
        return
    
    if juxtaposed:
        num_horizontal, num_vertical = min(len(models_to_compare), dists_per_row), (math.ceil(len(models_to_compare) / dists_per_row))
        figure, axes = plt.subplots(num_vertical, num_horizontal, figsize=(8 * num_horizontal, 6 * num_vertical), sharex=False, sharey=True)
        if len(models_to_compare) == 1:
            axes = [axes]
        axes = np.atleast_1d(axes).flatten()
        for model_name, ax in zip(models_to_compare, axes):
            plot_model_distributions(model_name, column_name=column_name, ax=ax, use_log_scale=use_log_scale)
        figure.suptitle(f'Distributions for {", ".join(models_to_compare)}')
        plt.tight_layout()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
    else:
        plt.figure(figsize=(15, 6))
        if use_log_scale:
            plt.yscale('log', base=10)
        for model_name in models_to_compare:
            plot_model_distributions(model_name, column_name=column_name, use_log_scale=use_log_scale)
        plt.title(f'Distributions for {", ".join(models_to_compare)}')
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()