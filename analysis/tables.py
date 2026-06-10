
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

import pUtil
from dictionary import Dictionary
import paths

class tables:
    """
    This class primarily aggregates all the data from the model meta, config, training and metrics files into various useful formats.
    """
    
    model_metadata_columns          = ['vocab_size', 'max_sequence_length', 'num_train_tokens', 'num_val_tokens']
    model_config_columns            = ['batch_size', 'gradient_accumulation_steps', 'block_size', 'learning_rate', 'min_lr', 'lr_warmup_iters', 'lr_decay_iters', 'lr_scheduler', 'n_layer', 'n_head', 'n_embd', 'num_params']
    model_training_columns          = ['iters_trained', 'iters_saved', 'min_saved_train_loss', 'min_saved_val_loss', 'compute_time_trained', 'compute_time_saved']
    model_metrics_columns           = ['coverage', 'mmd', 'kpd_median', 'fpd_value', 'w1m_score', 'w1p_avg_eta', 'w1p_avg_phi', 'w1p_avg_pt']
    model_metrics_columns_verbose   = ['kpd_error', 'fpd_error', 'w1m_score_std', 'w1p_avg_eta_std', 'w1p_avg_phi_std', 'w1p_avg_pt_std']
    model_all_columns               = ['model_name'] + model_metadata_columns + model_config_columns + model_training_columns + model_metrics_columns + ['running_df', 'checkpointed_df']
    model_all_columns_verbose       = ['model_name'] + model_metadata_columns + model_config_columns + model_training_columns + model_metrics_columns + model_metrics_columns_verbose
    
    @staticmethod
    def get_meta_data(model_name):            
        preparation_dir = pUtil.get_model_preparation_dir(model_name)
        preparation_name = pUtil.get_model_preparation_name(model_name)
        prep_data_filepath = preparation_dir / 'preparation.json'
        prep_info_filepath = preparation_dir / 'preparation_info.json'
        if not prep_data_filepath.exists() or not prep_info_filepath.exists():
            raise Exception(f"Preparation data or info file for found for preparation name {preparation_name}")
        
        with open(prep_data_filepath, 'r') as pdf:
            prep_data = json.load(pdf)
            if not 'train_bin' in prep_data:
                raise Exception(f"Preparation data file does not have a 'train_bin' key!")
            if not 'validation_bin' in prep_data:
                raise Exception(f"Preparation data file does not have a 'validation_bin' key!")
            
        with open(prep_info_filepath, 'r') as pif:
            prep_info = json.load(pif)
        
        max_sequence_length = prep_info.get('max_sequence_length', np.nan)
        return SimpleNamespace(
            vocab_size              = prep_info.get('vocab_size', np.nan),
            max_sequence_length     = max_sequence_length,
            num_train_tokens        = max_sequence_length * prep_data['train_bin'].get('num_events', np.nan),
            num_val_tokens          = max_sequence_length * prep_data['validation_bin'].get('num_events', np.nan),
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
        
        if np.isnan(block_size):
            block_size = context_events * meta_data.max_sequence_length
        
        # Get the number of trainable parameters. Even though we use the log file this is technically defined by the configuration.
        num_params = np.nan
        training_log_filename = pUtil.get_training_dir(model_name) / "train_log_1.jsonl"
        with open(training_log_filename) as training_log_file:
            for jline in training_log_file:
                jdata = json.loads(jline)
                if jdata.get("message") == "Model info" and "num_params" in jdata:
                    num_params = jdata['num_params']
                    num_params = int(float(num_params.replace("M", "")) * 1e6)
        
        return SimpleNamespace(
            batch_size                  = training_config.get('batch_size', np.nan),
            gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', np.nan),
            block_size                  = block_size,
            learning_rate               = training_config.get('learning_rate', np.nan),
            min_lr                      = training_config.get('min_lr', np.nan),
            lr_warmup_iters             = training_config.get('warmup_iters', np.nan),
            lr_decay_iters              = training_config.get('lr_decay_iters', np.nan),
            lr_scheduler                = training_config.get('lr_scheduler', 'cosine_annealing_with_warmup'),
            n_layer                     = training_config.get('n_layer', np.nan),
            n_head                      = training_config.get('n_head', np.nan),
            n_embd                      = training_config.get('n_embd', np.nan),
            scheme                      = training_config.get('scheme', 'unknown'),
            preparation_name            = training_config.get('preparation_name', 'unknown'),
            num_params                  = num_params
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
                    running_data.append({
                        'iter': jdata["iter"],
                        'epoch': current_epochs_trained,
                        'train_loss': jdata["train_loss"],
                        'val_loss': jdata["val_loss"],
                        'time': jdata['time']
                    })
                elif jdata.get("message") == "Training progress: checking checkpoint conditions":
                    current_epochs_trained = 0 if jdata['step'] == 0 else (jdata['step'] / iterations_per_epoch)
                    checkpointed_data.append({
                        'iter': jdata["step"],
                        'epoch': current_epochs_trained,
                        'train_loss': jdata["train_loss"],
                        'val_loss': jdata["val_loss"]
                    })
        
        return SimpleNamespace(
            running_df = pd.DataFrame(running_data),
            checkpointed_df = pd.DataFrame(checkpointed_data),
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
        # Trained iters and compute time
        iters_trained = training_run_data.running_df['iter'].max()
        compute_time_trained = training_run_data.running_df['time'].sum()        
        # Saved minimum validation loss
        min_saved_val_loss_row_idx = training_run_data.checkpointed_df['val_loss'].idxmin()
        min_saved_val_loss_row = training_run_data.checkpointed_df.loc[min_saved_val_loss_row_idx]
        iters_saved = int(min_saved_val_loss_row['iter'])
        # Saved compute time; we use the iteration closest to iters_saved since the exact might not exist.
        closest_to_min_saved_val_loss_row_idx = (training_run_data.running_df['iter'] - iters_saved).abs().idxmin()
        compute_time_saved = (training_run_data.running_df.iloc[:closest_to_min_saved_val_loss_row_idx])['time'].sum()
        
        training_run_data = SimpleNamespace(
            iters_trained           = iters_trained,
            iters_saved             = iters_saved,
            min_saved_train_loss    = min_saved_val_loss_row['train_loss'],
            min_saved_val_loss      = min_saved_val_loss_row['val_loss'],
            compute_time_trained    = compute_time_trained,
            compute_time_saved      = compute_time_saved,
            running_df              = training_run_data.running_df,
            checkpointed_df         = training_run_data.checkpointed_df,
        )
        
        return SimpleNamespace(**{'model_name': model_name, **vars(meta_data), **vars(config_data), **vars(metrics_data), **vars(training_run_data)})

    # Returns a DataFrame with all the important data for all models
    @staticmethod
    def get_default_df(model_names):
        model_names = np.atleast_1d(model_names)
        columns = tables.model_all_columns
        model_data_list = [row for name in model_names if (row := vars(tables.get_all_data(name))) is not None]
        model_data_df = pd.DataFrame(model_data_list, columns=columns)
        return model_data_df