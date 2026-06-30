
import json
import os
import sys
import paths
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass, field

import torch

@dataclass
class GenericConfiguration:
    config_file_path:           str = ''
    model_name:                 str = ''
    data_mode:                  str = 'particle' # 'particle' or 'generic'
    mlp_type:                   str = "swiglu"   # 'swiglu', 'relu2', 'gelu'
    mlp_ratio:                  float = 4.0
    qk_norm:                    bool = True
    embd_norm_type:             str = "none" # 'none' or 'rmsnorm'
    embd_norm_init_scale:       float = 0.02
    use_particle_index_embd:    bool = False
    use_bin_value_embd:         bool = False
    bin_value_embd_init_scale:  float = 0.0
    
    # Important; must have
    preparation_config_file:    Path | None = None

@dataclass
class TrainingConfiguration:
    # 'scratch', 'resume'
    # 'scratch' will force a restart, 'resume' will ensure no accidental restart
    init_from: str = ''
    
    # evaluation
    eval_interval: int = 2000 # Set to -1 to disable (e.g, if eval_every_epoch is True or if we just do not want any)
    eval_iters: int = 200
    eval_only: bool = False
    eval_every_epoch: bool = True
    log_interval: int = 10
    max_iters: int = 600000
    max_epochs: int = int(10e9)
    max_num_failed_checkpoint_checks: int = 4
    min_val_loss_improvement_criteria: float = 0.002 # Minimum relative improvement in validation loss to reset the failed checkpoint counter

    # data
    gradient_accumulation_steps: int = 32
    batch_size: int = 12
    block_size: int = -1
    # context_events is left for legacy purposes
    context_events: int = -1  # Used to compute block_size dynamically
    # used to compute block_size dynamically;
    # number of sequences to concatenate together in a block. Only used if block_size is -1.
    context_sequences: int = -1
    # If True, then PADDING token will be used when constructing Y tensors for training.
    # If False, then a real token from the next block will be used as the final target token.
    use_self_contained_blocks: bool = False

    # model
    n_layer: int = 12
    n_head: int = 12
    n_kv_heads: int = 0   # 0 means same as n_head (standard MHA)
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    loss_function: str = 'cross_entropy'

    # adamw optimizer
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    cycle_steps_mult: float = 1
    base_lr_decay_mult: float = 1
    loss_sigma = 1.0

    # learning rate
    lr_scheduler: str = "cosine_annealing_with_warmup" # cosine_annealing_with_warmup, cosine_annealing_with_warm_restarts
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    learning_rate: float = 6e-4
    min_lr: float = 6e-5

    backend: str = 'nccl'
    
    # DDP settings
    auto_ddp = True
    auto_ddp_world_size = 4
    auto_ddp_master_addr = "127.0.0.1"
    auto_ddp_master_port = 0

    # system
    device: str = 'cuda'
    dtype: str = field(init=False)
    compile: bool = True
    seed: int = 1337
    
    meta_benchmarking: bool = False
    
    # @TODO: put this somewhere else, this is not a config variable, it should be derived from the dataset and training configuration.
    # It makes sense to put this in TokenBlockDataset
    iterations_per_epoch: int = 0

    def __post_init__(self):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = 'bfloat16'
        else:
            self.dtype = 'float16'

@dataclass
class SamplingConfiguration:
    samples_storage_dir:    str = ''
    batch_size:             int = 128
    max_new_tokens:         int | None = 500 # If None will be auto calculated as sequence_length - starting_tokens
    temperature:            float = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k:                  int = 200
    seed:                   int = 1337
    device:                 str = 'cuda'
    dtype:                  str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile:                bool = True
    num_sample_sequences:   int | None = None # If None, will use all test sequences
    keep_shards:            bool = False
    sampling_idx_override:  int | None = None
    force_single_gpu:       bool = False
    log_interval:           int = 1000
    
    # Used for untokenization
    stop_at_event_end:      bool = True
    stop_at_padding:        bool = True
    float_precision:        float = 5

generic = None
training = None
sampling = None

def should_print_import_configuration_message() -> bool:
    """
    Return True only for the top-level process that should emit import-time
    configurator messages.

    This preserves configurator side effects on import, but prevents repeated
    stdout spam from DDP ranks, mp.spawn children, and spawned worker processes.
    """
    if mp.current_process().name != "MainProcess":
        return False

    rank = os.environ.get("RANK")
    if rank is not None and int(rank) != 0:
        return False

    return True

def perform_configuration(config_file_path):
    generic = GenericConfiguration()
    training = TrainingConfiguration()
    sampling = SamplingConfiguration()
    
    if should_print_import_configuration_message():
        print(f'Configurator found file {config_file_path}.')
    
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    generic.config_file_path = paths.PROJECT_DIR / config_file_path

    for key, value in config.items():
        if key in ['training_config', 'sampling_config']:
            continue
        if hasattr(generic, key):
            setattr(generic, key, value)
        else:
            print(f'Warning! Key {key} not found in GenericConfiguration. Skipping.')
    
    if 'training_config' in config:
        for key, value in config['training_config'].items():
            if hasattr(training, key):
                setattr(training, key, value)
            else:
                print(f'Warning! Key {key} not found in TrainingConfiguration. Skipping.')
            
    if 'sampling_config' in config:
        for key, value in config['sampling_config'].items():
            if hasattr(sampling, key):
                setattr(sampling, key, value)
            else:
                print(f'Warning! Key {key} not found in SamplingConfiguration. Skipping.')

    # If no model_name is set, then use the config file name.
    # This saves a lot of pain with accidentally overwriting files.
    if (generic.model_name == ''):
        config_file_name = Path(config_file_path).name
        config_file_name_stripped = Path(config_file_name).stem
        generic.model_name = config_file_name_stripped
    
    return generic, training, sampling



def main(config_filepath):
    global generic
    global training
    global sampling
    arg_one_exceptions = [ '--help', '-h', 'all', 'single_threaded', '--benchmark']
    # Assume first argument is config file path
    if config_filepath not in arg_one_exceptions:
        config_file_path = config_filepath
        generic, training, sampling = perform_configuration(config_file_path)
        
        # Handle legacy context_events alias
        if training.context_sequences == -1 and training.context_events != -1:
            training.context_sequences = training.context_events
    
    if '--benchmark' in sys.argv:
        training.meta_benchmarking = True

if __name__ == "__main__":
    # Configurator should only run if a config file is provided as an argument.
    # All expected exceptions will be handled here.
    if not len(sys.argv) > 1:
        print("WARNING: No config file or options provided.")
    else:
        main(sys.argv[1])