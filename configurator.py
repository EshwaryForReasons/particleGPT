import json
import sys
import os
import torch

from dataclasses import dataclass, field

config_file_path = sys.argv[1]

@dataclass
class GenericConfiguration:
    config_file_path:      str = ''
    preparation_name:      str = ''
    model_name:            str = ''
    dataset:               str = ''

@dataclass
class TrainingConfiguration:
    # I/O
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    init_from: str = ''  # 'scratch', 'resume', or 'gpt2*'

    # data
    gradient_accumulation_steps: int = field(default=5 * 8)
    batch_size: int = 12
    block_size: int = -1
    context_events: int = 1  # Used to compute block_size dynamically

    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    # adamw optimizer
    learning_rate: float = 6e-4
    max_iters: int = 600000
    max_num_failed_checkpoint_checks: int = 4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    lr_scheduler: str = "cosine_annealing_with_warmup" # cosine_annealing_with_warmup, cosine_annealing_with_warm_restarts

    # learning rate decay
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # DDP settings
    backend: str = 'nccl'

    # system
    device: str = 'cuda'
    dtype: str = field(init=False)
    compile: bool = True
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
    max_new_tokens:         int = 500
    temperature:            float = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k:                  int = 200
    seed:                   int = 1337
    device:                 str = 'cuda'
    dtype:                  str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile:                bool = True

generic = GenericConfiguration()
training = TrainingConfiguration()
sampling = SamplingConfiguration()

def perform_configuration():
    global generic
    global training
    global sampling
    
    print(f'Configurator found file {config_file_path}.')
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    generic.config_file_path = config_file_path

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
        config_file_name = os.path.basename(config_file_path)
        config_file_name_stripped = os.path.splitext(config_file_name)
        generic.model_name = config_file_name_stripped[0]

# Configurator should only run if a config file is provided as an argument.
# All expected exceptions will be handled here.
arg_one_exceptions = [ '--help', '-h', 'all', 'single_threaded']
if len(sys.argv) > 1 and sys.argv[1] not in arg_one_exceptions:
    perform_configuration()