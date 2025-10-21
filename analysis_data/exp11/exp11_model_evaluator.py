"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import json

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import model
from model import GPTConfig, GPT
import pLogging
import configurator as conf

script_dir = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Ensure the output directory exists. Need to do this before creating the logger.
model_output_dir = script_dir / "trained_models" / conf.generic.model_name
model_output_dir.mkdir(parents=True, exist_ok=True)

logger_idx = -1

data_dir = Path('data', conf.generic.preparation_name)
meta_path = Path(data_dir, 'meta.pkl')

if not meta_path.exists():
    print("No meta file found, ensure data is prepared!")
    
# attempt to derive vocab_size from the dataset
meta_vocab_size = None
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
meta_vocab_size = meta['vocab_size']

# If block_size is not set, then use context_particles to set it
if conf.training.block_size == -1 and conf.training.context_events != -1:
    conf.training.block_size = conf.training.context_events * meta['max_sequence_length']

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=conf.training.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    conf.training.device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(conf.training.device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    # if not master_process:
    #     logger_idx = -1 # disable logging for non-master processes
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert conf.training.gradient_accumulation_steps % ddp_world_size == 0
    conf.training.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = conf.training.gradient_accumulation_steps * ddp_world_size * conf.training.batch_size * conf.training.block_size

if master_process:
    os.makedirs(model_output_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in conf.training.device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[conf.training.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

class pDataset(Dataset):
    def __init__(self, split):
        assert split in ['train', 'val'], "Split must be either 'train' or 'val'"
        self.split = split
        self.data_filepath = data_dir / ('train.bin' if split == 'train' else 'val.bin')
        data = np.memmap(self.data_filepath, dtype=np.uint16, mode='r')
        assert len(data) % conf.training.block_size == 0, "Data length is not a multiple of block_size!"
        self.block_size = conf.training.block_size
        self.num_samples = len(data) // self.block_size
    
    def __getitem__(self, index):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        data = np.memmap(self.data_filepath, dtype=np.uint16, mode='r')
        
        # Our data is guaranteed to be a multiple of block_size, so we can reshape it.
        assert len(data) % conf.training.block_size == 0, "Data length is not a multiple of block_size!"
        data = data.reshape(-1, conf.training.block_size)
        
        x = torch.from_numpy(data[index].astype(np.int64)) # shape (batch_size, block_size).
        y = torch.roll(x.clone(), shifts=-1, dims=0) # shape (batch_size, block_size), shifted by 1.
        y[-1] = 0 # 0 is our padding token and it is ignored in the loss calculations.
        
        # We do this here because Dataloader's pin_memory=True does not automatically move to GPU.
        if device_type == 'cuda':
            x, y = x.pin_memory().to(conf.training.device, non_blocking=True), y.pin_memory().to(conf.training.device, non_blocking=True)
        else:
            x, y = x.to(conf.training.device), y.to(conf.training.device)

        return x, y
    
    def __len__(self):
        return self.num_samples
    
train_dataset = pDataset('train')
val_dataset = pDataset('val')
# For loss estimation
le_train_dataloader = DataLoader(
    train_dataset,
    batch_size=conf.training.batch_size,
    shuffle=False if ddp else True,
    drop_last=True,
    sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True) if ddp else None
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=conf.training.batch_size,
    shuffle=False,
    drop_last=True,
    sampler=DistributedSampler(val_dataset, drop_last=True) if ddp else None
)

conf.training.iterations_per_epoch = int(len(le_train_dataloader) // conf.training.gradient_accumulation_steps)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
num_failed_checkpoint_checks = 0

# model init; start with model_args from command line
model_args = dict(
    n_layer=conf.training.n_layer, 
    n_head=conf.training.n_head, 
    n_embd=conf.training.n_embd, 
    block_size=conf.training.block_size,
    bias=conf.training.bias, 
    vocab_size=None, 
    dropout=conf.training.dropout
)

def get_all_checkpoint_filepaths():
    ckpt_dict = {}
    for path in model_output_dir.glob("ckpt_running_*.pt"):
        idx_str = path.stem.split("_")[-1]
        try:
            idx = int(idx_str)
            ckpt_dict[idx] = path
        except ValueError:
            pass
    return dict(sorted(ckpt_dict.items()))

# Returns the epoch number from the current iteration number.
def get_epoch_from_iter(iter):
    assert conf.training.iterations_per_epoch > 0, "iterations_per_epoch must be greater than 0"
    epoch = int(iter // conf.training.iterations_per_epoch)
    remaining_iters = iter - (epoch * conf.training.iterations_per_epoch)
    return epoch, remaining_iters

@torch.no_grad()
def estimate_loss():
    epochs_trained_thus_far, iters_trained_thus_far = get_epoch_from_iter(iter_num)
    dataloader = val_dataloader
    
    if ddp:
        dataloader.sampler.set_epoch(epochs_trained_thus_far)
        
    data_loader_iter = iter(dataloader)
    losses = torch.zeros(conf.training.eval_iters)
    for k in range(conf.training.eval_iters):
        try:
            # Skip any iterations already trained in this epoch
            skip_iters_range = range(iters_trained_thus_far)
            for iter_idx in skip_iters_range:
                next(data_loader_iter)
                iters_trained_thus_far -= 1
            x, y = next(data_loader_iter)
        except StopIteration:
            # If our dataset size is small for the batch size, then eval_iters may be larger than how many we can
            # do in this epoch. In this case we increase the epoch and continue.
            epochs_trained_thus_far += 1
            if ddp:
                dataloader.sampler.set_epoch(epochs_trained_thus_far)
            data_loader_iter = iter(dataloader)
            x, y = next(data_loader_iter)
            continue
        
        with ctx:
            logits, loss, _ = model(x, y)
        losses[k] = loss.item()
        
    return losses.mean()

all_model_checkpoints = get_all_checkpoint_filepaths()
losses_estimated = {}
for ckpt_idx, ckpt_path in all_model_checkpoints.items():
    print(f"Found checkpoint: {ckpt_path} with index {ckpt_idx}")
    
    checkpoint = torch.load(ckpt_path, map_location=conf.training.device)
    checkpoint_model_args = checkpoint['model_args']
    
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']

    # crop down the model block size if desired, using model surgery
    if conf.training.block_size < model.config.block_size:
        model.crop_block_size(conf.training.block_size)
        model_args['block_size'] = conf.training.block_size # so that the checkpoint will have the right value
    model.to(conf.training.device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler('cuda', enabled=(conf.training.dtype == 'float16'))

    # conf.training.optimizer
    optimizer = model.configure_optimizers(conf.training.weight_decay, conf.training.learning_rate, (conf.training.beta1, conf.training.beta2), device_type)
    optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        
    model.eval()
    val_loss_for_this_model = estimate_loss()
    losses_estimated[ckpt_idx * conf.training.eval_interval] = val_loss_for_this_model.item()

with open(f'estimated_losses_{conf.generic.model_name}.json', 'w') as f:
    json.dump(losses_estimated, f, indent=4)

print(losses_estimated)