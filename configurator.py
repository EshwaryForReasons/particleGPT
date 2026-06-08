import json
import sys
import os
import torch

from dataclasses import dataclass, field

@dataclass
class GenericConfiguration:
    config_file_path:           str = ''
    preparation_name:           str = ''
    model_name:                 str = ''
    dataset:                    str = ''
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
    preparation_config_file:    str | None = None

@dataclass
class TrainingConfiguration:
    # I/O
    eval_interval: int = 2000 # Set negative to disable (e.g, if eval_every_epoch is True or if we just do not want any)
    log_interval: int = 10
    eval_iters: int = 200
    eval_only: bool = False
    init_from: str = ''  # 'scratch', 'resume', or 'gpt2*'
    eval_every_epoch: bool = True

    # data
    gradient_accumulation_steps: int = field(default=5 * 8)
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
    learning_rate: float = 6e-4
    max_iters: int = 600000
    max_epochs: int = int(10e9)
    max_num_failed_checkpoint_checks: int = 4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    cycle_steps_mult: float = 1
    base_lr_decay_mult: float = 1
    lr_scheduler: str = "cosine_annealing_with_warmup" # cosine_annealing_with_warmup, cosine_annealing_with_warm_restarts
    
    loss_sigma = 1.0

    # learning rate decay
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # DDP settings
    backend: str = 'nccl'
    
    auto_ddp = True
    auto_ddp_world_size = 4
    auto_ddp_master_addr = "127.0.0.1"
    auto_ddp_master_port = 0

    # system
    device: str = 'cuda'
    dtype: str = field(init=False)
    compile: bool = True
    iterations_per_epoch: int = 0
    seed: int = 1337
    
    meta_benchmarking: bool = False

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
    max_test_sequences:     int | None = None

generic = None
training = None
sampling = None

def perform_configuration(config_file_path):
    generic = GenericConfiguration()
    training = TrainingConfiguration()
    sampling = SamplingConfiguration()
    
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
    
    return generic, training, sampling

# Configurator should only run if a config file is provided as an argument.
# All expected exceptions will be handled here.
arg_one_exceptions = [ '--help', '-h', 'all', 'single_threaded', '--benchmark']
if not len(sys.argv) > 1:
    print("WARNING: No config file or options provided.")
else:
    # Assume first argument is config file path
    if sys.argv[1] not in arg_one_exceptions:
        config_file_path = sys.argv[1]
        generic, training, sampling = perform_configuration(config_file_path)
    
    if '--benchmark' in sys.argv:
        training.meta_benchmarking = True
    