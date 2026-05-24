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

import json
import os
import time
import math
import pickle
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import itertools
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import model_fixes1_v1 as model
from model_fixes1_v1 import GPTConfig, GPT
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

logger_idx = pLogging.create_training_logger(conf.generic.model_name, 1)
model.set_logger(logger_idx)
pLogging.info(logger_idx, 'Training started.')

# ===== Define all directories and filepaths and ensure all required files exist =====

# Not a required file
best_ckpt_path = model_output_dir / 'ckpt.pt'

prep_dir = script_dir / 'preparations' / conf.generic.preparation_name
if not prep_dir.exists():
    pLogging.info(logger_idx, "No preparation directory found, ensure it is set correctly in the configuration!")
    raise Exception("No preparation directory found, ensure it is set correctly in the configuration!")

prep_info_filepath = prep_dir / 'preparation_info.json'
if not prep_info_filepath.exists():
    pLogging.info(logger_idx, "No preparation info file found, ensure it is set correctly in the configuration!")
    raise Exception("No preparation info file found, ensure it is set correctly in the configuration!")

# ===== Get useful data from prep info =====

try:
    with open(prep_info_filepath, 'r') as f:
        prep_info = json.load(f)
    prep_vocab_size = prep_info['vocab_size']
    prep_max_sequence_length = prep_info['max_sequence_length']
except Exception as e:
    pLogging.info(logger_idx, f"Error occurred while reading preparation info: {e}")
    raise Exception("Error occurred while reading preparation info!")

# If block_size is not set, then use context_particles to set it
if conf.training.block_size == -1 and conf.training.context_events != -1:
    conf.training.block_size = conf.training.context_events * prep_max_sequence_length

# ===== Useful functions =====

# Checkpoints are stored as ckpt.pt (best val loss), ckpt_running_idx.pt (in order of saving, if eval_iters is set), and cktp_epoch_idx.pt (per epoch, if enabled).
# This function returns the latest running checkpoint filepath or None if there are none.
def get_latest_running_ckpt_filepath():
    running_ckpt_map = {}
    for path in model_output_dir.glob("ckpt_running_*.pt"):
        idx_str = path.stem.split("_")[-1]
        try:
            idx = int(idx_str)
            running_ckpt_map[idx] = path
        except ValueError:
            pass
        
    latest_idx = -1
    latest_running_checkpoint = None
    if running_ckpt_map:
        latest_idx = max(running_ckpt_map)
        latest_running_checkpoint = running_ckpt_map[latest_idx]
    
    return latest_idx, latest_running_checkpoint

# This function returns the latest epoch checkpoint filepath or None if there are none.
def get_latest_epoch_ckpt_filepath():
    epoch_ckpt_map = {}
    for path in model_output_dir.glob("ckpt_epoch_*.pt"):
        idx_str = path.stem.split("_")[-1]
        try:
            idx = int(idx_str)
            epoch_ckpt_map[idx] = path
        except ValueError:
            pass
    
    latest_idx = -1
    latest_epoch_checkpoint = None
    if epoch_ckpt_map:
        latest_idx = max(epoch_ckpt_map)
        latest_epoch_checkpoint = epoch_ckpt_map[latest_idx]
    
    return latest_idx, latest_epoch_checkpoint

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
    
def ddp_broadcast_bool(value, src=0):
    if not ddp:
        return value

    flag = torch.tensor([1 if value else 0], device=conf.training.device)
    dist.broadcast(flag, src=src)
    return bool(flag.item())

if master_process:
    os.makedirs(model_output_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in conf.training.device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[conf.training.dtype]

# Default to bfloat16 if GPU supports it and no explicit dtype is set
if device_type == 'cuda' and conf.training.dtype == 'float16':
    # keep float16 + scaler
    ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
elif device_type == 'cuda' and conf.training.dtype == 'bfloat16':
    ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    scaler = torch.amp.GradScaler('cuda', enabled=False)   # no-op scaler
else:
    ctx = nullcontext()
    scaler = torch.amp.GradScaler('cuda', enabled=False)

class pDataset(Dataset):
    def __init__(self, split):
        assert split in ["train", "val"]
        self.split = split
        self.data_filepath = prep_dir / ("train.bin" if split == "train" else "val.bin")

        data = np.memmap(self.data_filepath, dtype=np.uint16, mode="r")
        assert len(data) % conf.training.block_size == 0
        self.block_size = conf.training.block_size
        self.num_samples = len(data) // self.block_size

        # Each dataloader worker will lazily open its own memmap.
        self._data = None

    def _get_data(self):
        if self._data is None:
            data = np.memmap(self.data_filepath, dtype=np.uint16, mode="r")
            assert len(data) % self.block_size == 0
            self._data = data.reshape(-1, self.block_size)
        return self._data

    def __getitem__(self, index):
        data = self._get_data()

        # Embedding inputs must be long.
        x = torch.from_numpy(data[index].astype(np.int64, copy=True))

        # Faster than torch.roll for this exact task.
        y = torch.empty_like(x)
        y[:-1] = x[1:]
        # @TODO: incorporate dictionary later on; not ideal to have these hard-coded.
        # something like the following might be ok for now?
        # y[-1] = dictionary.padding_token if "dictionary" in globals() else 0
        y[-1] = 0 # 0 is our padding token and it is ignored in the loss calculations.

        return x, y
    
    def __len__(self):
        return self.num_samples
    
class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.iterator = None
        self.next_batch = None

    def __iter__(self):
        self.iterator = iter(self.loader)
        self.preload()
        return self

    def preload(self):
        try:
            X, Y = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)

        self.next_batch = (X, Y)

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        torch.cuda.current_stream().wait_stream(self.stream)

        X, Y = self.next_batch

        # Important: make sure tensors stay alive until current stream is done.
        X.record_stream(torch.cuda.current_stream())
        Y.record_stream(torch.cuda.current_stream())

        self.preload()
        return X, Y
    
train_dataset = pDataset('train')
val_dataset = pDataset('val')

NUM_WORKERS = 4  # try 2, 4, 8 per GPU and benchmark

loader_kwargs = dict(
    batch_size=conf.training.batch_size,
    drop_last=True,
    pin_memory=(device_type == "cuda"),
    num_workers=NUM_WORKERS,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
)

# Remove None-valued args because DataLoader dislikes prefetch_factor with num_workers=0.
loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}

train_dataloader = DataLoader(
    train_dataset,
    shuffle=False if ddp else True,
    sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True) if ddp else None,
    **loader_kwargs,
)

le_train_dataloader = DataLoader(
    train_dataset,
    shuffle=False if ddp else True,
    sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True) if ddp else None,
    **loader_kwargs,
)

val_dataloader = DataLoader(
    val_dataset,
    shuffle=False,
    sampler=DistributedSampler(val_dataset, drop_last=True) if ddp else None,
    **loader_kwargs,
)

conf.training.iterations_per_epoch = int(len(train_dataloader) // conf.training.gradient_accumulation_steps)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
num_failed_checkpoint_checks = 0

pLogging.info(logger_idx, "Training configuration", {
    "config_file_path": conf.generic.config_file_path,
    "preparation": conf.generic.preparation_name,
    "model_name": conf.generic.model_name,
    "ddp": ddp,
    "ddp_world_size": ddp_world_size,
    "device": conf.training.device,
    "dtype": conf.training.dtype,
    "compile": conf.training.compile,
    "prep_vocab_size": prep_vocab_size,
    "prep_max_sequence_length": prep_max_sequence_length,
    "eval_interval": conf.training.eval_interval,
    "log_interval": conf.training.log_interval,
    "eval_iters": conf.training.eval_iters,
    "eval_only": conf.training.eval_only,
    "init_from": conf.training.init_from,
    "gradient_accumulation_steps": conf.training.gradient_accumulation_steps,
    "batch_size": conf.training.batch_size,
    "block_size": conf.training.block_size,
    "n_layer": conf.training.n_layer,
    "n_head": conf.training.n_head,
    "n_kv_heads": conf.training.n_kv_heads,
    "n_embd": conf.training.n_embd,
    "dropout": conf.training.dropout,
    "bias": conf.training.bias,
    "scheduler": conf.training.lr_scheduler,
    "learning_rate": conf.training.learning_rate,
    "max_iters": conf.training.max_iters,
    "weight_decay": conf.training.weight_decay,
    "beta1": conf.training.beta1,
    "beta2": conf.training.beta2,
    "grad_clip": conf.training.grad_clip,
    "warmup_iters": conf.training.warmup_iters,
    "lr_decay_iters": conf.training.lr_decay_iters,
    "min_lr": conf.training.min_lr,
    "backend": conf.training.backend,
    "device_type": device_type
})

# model init; start with model_args from command line
model_args = dict(
    n_layer=conf.training.n_layer, 
    n_head=conf.training.n_head, 
    n_embd=conf.training.n_embd, 
    n_kv_heads=conf.training.n_kv_heads if conf.training.n_kv_heads else conf.training.n_head,
    block_size=conf.training.block_size,
    bias=conf.training.bias, 
    # this will be overridden if we resume from a checkpoint and the checkpoint has a different vocab size, but that is necessary to ensure we can even resume from the checkpoint
    vocab_size=prep_vocab_size,
    dropout=conf.training.dropout
)

# ===== Determine if we should resume or start from scratch =====

# This is based on if a running/epoch cktp file exists and if the resume behavior is overridden.
init_training_from = 'scratch' # or 'resume'
latest_running_cktp_idx, latest_running_ckpt = get_latest_running_ckpt_filepath()
if latest_running_ckpt and conf.training.init_from != 'scratch':
    init_training_from = 'resume'
latest_epoch_cktp_idx, latest_epoch_ckpt = get_latest_epoch_ckpt_filepath()
if latest_epoch_ckpt and conf.training.init_from != 'scratch':
    init_training_from = 'resume'

if init_training_from == 'scratch':
    # Init a new model from scratch
    pLogging.info(logger_idx, "Initializing a new model from scratch")
    pLogging.info(logger_idx, "Training progress", {"info": "Initializing a new model from scratch"})
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_training_from == 'resume':
    # First we need to determine if we start from the latest epoch ckpt or the latest saved running cktp
    resume_ckpt = None
    
    running_iter_num = 0
    epoch_iter_num = 0
    if latest_running_ckpt is not None:
        temp_checkpoint = torch.load(latest_running_ckpt, map_location=conf.training.device)
        running_iter_num = temp_checkpoint['iter_num']
    if latest_epoch_ckpt is not None:
        temp_checkpoint = torch.load(latest_epoch_ckpt, map_location=conf.training.device)
        epoch_iter_num = temp_checkpoint['iter_num']
    
    assert (running_iter_num != 0 or epoch_iter_num != 0), "If resume pathway was picked, one of these must exist!"
    if running_iter_num > epoch_iter_num:
        resume_ckpt = latest_running_ckpt
    elif running_iter_num < epoch_iter_num:
        resume_ckpt = latest_epoch_ckpt
    elif running_iter_num == epoch_iter_num:
        resume_ckpt = latest_running_ckpt
    
    # Resume training from the chosen
    pLogging.info(logger_idx, f"Resuming training from {resume_ckpt}")
    pLogging.info(logger_idx, "Training progress", {"info": f"Resuming training from {resume_ckpt}"})
    checkpoint = torch.load(resume_ckpt, map_location=conf.training.device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'n_kv_heads']:
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
    best_val_loss = checkpoint['best_val_loss']

# crop down the model block size if desired, using model surgery
if conf.training.block_size < model.config.block_size:
    model.crop_block_size(conf.training.block_size)
    model_args['block_size'] = conf.training.block_size # so that the checkpoint will have the right value
model.to(conf.training.device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(conf.training.dtype == 'float16'))

# conf.training.optimizer
optimizer = model.configure_optimizers(conf.training.weight_decay, conf.training.learning_rate, (conf.training.beta1, conf.training.beta2), device_type)
if init_training_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if conf.training.compile:
    pLogging.info(logger_idx, "Training progress", {"info": "Compiling the model (reduce-overhead, dynamic=False)"})
    unoptimized_model = model
    model = torch.compile(
        model,
        # mode="reduce-overhead",
        mode="default",
        dynamic=False,      # all shapes are static (block_size fixed)
        fullgraph=False     # @IMPORTANT: set to True only after verifying no graph breaks
    )

# wrap model into DDP container
# if ddp:
    # model = DDP(model, device_ids=[ddp_local_rank])
if ddp:
    model = DDP(
        model,
        device_ids=[ddp_local_rank],
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )

# Returns the epoch number from the current iteration number.
def get_epoch_from_iter(iter):
    assert conf.training.iterations_per_epoch > 0, "iterations_per_epoch must be greater than 0"
    epoch = int(iter // conf.training.iterations_per_epoch)
    remaining_iters = iter - (epoch * conf.training.iterations_per_epoch)
    return epoch, remaining_iters

@torch.no_grad()
def estimate_loss():
    """
    Fast, reliable estimation of train/val loss.
    - Uses `eval_iters` batches from each split.
    - In DDP, losses are all-reduced to give the global mean.
    - Handles CPU/CUDA automatically.
    """
    model.eval()
    
    out = {}
    # For each split, we collect all local losses, then average across ranks
    for split, dataloader in zip(["train", "val"], [le_train_dataloader, val_dataloader]):
        if ddp:
            # Set the epoch so each rank gets a consistent view
            # For train we use the current epoch; for val we always start from epoch 0
            epoch = get_epoch_from_iter(iter_num)[0] if split == "train" else 0
            dataloader.sampler.set_epoch(epoch)

        losses = []
        # Take exactly `eval_iters` batches from the dataloader
        for batch_idx, (x, y) in enumerate(itertools.islice(dataloader, conf.training.eval_iters)):
            # Move data to the training device
            x = x.to(conf.training.device, non_blocking=True)
            y = y.to(conf.training.device, non_blocking=True)

            # Forward pass with the appropriate autocast context
            with ctx:
                _, loss, _ = model(x, y)
            losses.append(loss.detach())

        # Stack losses from this rank
        local_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=conf.training.device)

        # If DDP, all-reduce the mean loss across all ranks
        if ddp:
            # Sum of local means, then divide by world size
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            local_loss /= ddp_world_size

        out[split] = local_loss.cpu()

    model.train()
    return out

# learning rate decay scheduler
def get_lr(it):
    if conf.training.lr_scheduler == 'cosine_annealing_with_warmup':
        # 1) linear warmup for warmup_iters steps
        if it < conf.training.warmup_iters:
            return conf.training.learning_rate * it / conf.training.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it >= conf.training.lr_decay_iters:
            return conf.training.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - conf.training.warmup_iters) / (conf.training.lr_decay_iters - conf.training.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return conf.training.min_lr + coeff * (conf.training.learning_rate - conf.training.min_lr)
    elif conf.training.lr_scheduler == 'cosine_with_warmup':
        # 1) linear warmup for warmup_iters steps
        if it < conf.training.warmup_iters:
            return conf.training.learning_rate * it / conf.training.warmup_iters
        # 2) in between, use cosine decay down to min learning rate
        decay_ratio = (it - conf.training.warmup_iters) / (conf.training.lr_decay_iters - conf.training.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return conf.training.min_lr + coeff * (conf.training.learning_rate - conf.training.min_lr)
    elif conf.training.lr_scheduler == 'cosine_annealing_with_warm_restarts':
        # 1) linear warmup for warmup_iters steps
        if it < conf.training.warmup_iters:
            return conf.training.learning_rate * (it / conf.training.warmup_iters)
        # Adjust iteration to account for warmup
        it -= conf.training.warmup_iters
        # 2) Find current cycle and position in the cycle
        cycle = 0
        curr_cycle_len = conf.training.lr_decay_iters
        iter_in_cycle = it
        while iter_in_cycle >= curr_cycle_len:
            iter_in_cycle -= curr_cycle_len
            cycle += 1
            curr_cycle_len = int(curr_cycle_len * conf.training.cycle_steps_mult)
        # 3) Decay the base learning rate for the current cycle
        curr_base_lr = conf.training.learning_rate * (conf.training.base_lr_decay_mult ** cycle)
        # 4) Normalized progress within the cycle
        t = iter_in_cycle / curr_cycle_len
        # 5) Cosine annealing
        lr = conf.training.min_lr + 0.5 * (curr_base_lr - conf.training.min_lr) * (1 + math.cos(math.pi * t))
        return lr
    elif conf.training.lr_scheduler == 'constant':
        return conf.training.learning_rate
    raise ValueError(f"Unknown lr_scheduler {conf.training.lr_scheduler}")

# training loop
pLogging.info(logger_idx, f'Iterations per epoch is {conf.training.iterations_per_epoch}')
pLogging.info(logger_idx, 'iterations_per_epoch', {"iterations_per_epoch": conf.training.iterations_per_epoch})
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
trained_epochs = 0 # number of epochs trained in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

meta_benchmarking = conf.training.meta_benchmarking

# ---- Benchmarking / metrics ----
if meta_benchmarking == True:
    pLogging.info(logger_idx, 
        "Meta benchmarking is enabled. This will log micro step times in milliseconds to the training log, "
        "which can be used to get an accurate measure of time per step excluding evaluation and checkpointing."
    )
    
    # total global batch size across all GPUs and grad accum steps
    global_batch_size = conf.training.batch_size * conf.training.gradient_accumulation_steps * ddp_world_size
    step_times = []                  # store micro‑step times for averaging
    
    if device_type == 'cuda' and master_process:
        torch.cuda.reset_peak_memory_stats()

# Used for when we are resuming training from a checkpoint.
epochs_trained_thus_far, iters_trained_thus_far = get_epoch_from_iter(iter_num)
data_retrievals_so_far = iters_trained_thus_far * conf.training.gradient_accumulation_steps
continue_training = True
for epoch_num in range(epochs_trained_thus_far, conf.training.max_epochs + 1):
    if not continue_training:
        break
    
    # Update epoch so the correct seed is used for shuffling.
    if ddp:
        train_dataloader.sampler.set_epoch(epoch_num)

    # One run of this loop is essentially a micro step (grad_accum micro steps form an iteration).
    # for num_micro_steps, (X, Y) in enumerate(train_dataloader): # old, doesn't work with new prefetcher
    # CPU runs cannot use CUDA prefetcher; this allows CPU/debug runs to still work
    epoch_loader = train_dataloader
    if device_type == "cuda":
        epoch_loader = CUDAPrefetcher(train_dataloader, conf.training.device)
    
    for num_micro_steps, (X, Y) in enumerate(epoch_loader):
        if meta_benchmarking == True:
            step_start = time.time()
        
        # Skip over data_retrievals_so_far so we are at the correct point in the shuffle.
        # This is for the case where we resume training from a checkpoint.
        if data_retrievals_so_far > 0:
            data_retrievals_so_far -= 1
            continue
            
        # Termination conditions
        if iter_num >= conf.training.max_iters or (iter_num == 0 and conf.training.eval_only):
            continue_training = False
            break
        
        # Get learning rate for this iteration
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(iter_num)
        
        # ----------------------------------------------------------------------
        # Perform training.
        # ----------------------------------------------------------------------
        
        b_grad_accum_steps_completed = (num_micro_steps + 1) % conf.training.gradient_accumulation_steps == 0
        
        # in DDP training we only need to sync gradients at the last micro step.
        # the official way to do this is with model.no_sync() context manager, but
        # I really dislike that this bloats the code and forces us to repeat code
        # looking at the source of that context manager, it just toggles this variable
        if ddp:
            model.require_backward_grad_sync = b_grad_accum_steps_completed
            
        with ctx:
            logits, loss, _ = model(X, Y)
            del logits  # reduce VRAM usage (hopefully)
            # scale the loss to account for gradient accumulation
            loss = loss / conf.training.gradient_accumulation_steps 
        
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        
        if meta_benchmarking == True:
            if torch.cuda.is_available():
                # The synchronize() is crucial to get an accurate GPU step time.
                # Learned that the hard way..
                torch.cuda.synchronize()
            step_time = time.time() - step_start
            step_times.append(step_time * 1000)   # milliseconds
        
        # ----------------------------------------------------------------------
        # Update gradients and optimizer every gradient_accumulation_steps micro steps.
        # ----------------------------------------------------------------------
        
        if b_grad_accum_steps_completed:
            # ===== Optimizer step =====
            
            # clip the gradient
            if conf.training.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf.training.grad_clip)
            
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            
            losses = None
            should_stop = False
            
            # ===== Determine if we need eval/checkpointing/logging =====
            
            # Computed on all ranks so they follow the same control-flow process
            eval_cond_1 = conf.training.eval_interval > 0 and (iter_num % conf.training.eval_interval == 0)
            eval_cond_2 = conf.training.eval_every_epoch and (iter_num % conf.training.iterations_per_epoch == 0)
            log_cond = conf.training.log_interval > 0 and (iter_num % conf.training.log_interval == 0)
            
            # ===== Evaluation Section =====
            
            if eval_cond_1 or eval_cond_2 or log_cond:
                losses = estimate_loss()
                    
            # ===== Checkpointing/Early stopping =====
            # Only the master process writes logs and handles checkpoints
            
            # Save checkpoint
            if (eval_cond_1 or eval_cond_2) and master_process:
                # Evaluate the loss on train/val sets and write checkpoints. We save checkpoints every eval_interval and/or epoch. 
                # ckpt.pt is the checkpoint with the best val_loss; ckpt_running.pt is saved regardless of val_loss; ckpt_epoch.pt is
                # saved every epoch also regardless of val_loss. 
                # This is to ensure we can resume training from the running checkpoint, without having to redo many of them.
                b_save_best = False
                if losses['val'] < best_val_loss:
                    b_save_best = True
                    best_val_loss = losses['val']
                
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                
                pLogging.info(logger_idx, "Training progress: checking checkpoint conditions", {
                    "step": iter_num,
                    "train_loss": losses['train'].item(),
                    "val_loss": losses['val'].item()
                })
                
                if b_save_best and iter_num > 0:
                    num_failed_checkpoint_checks = 0
                    pLogging.info(logger_idx, f"Training progress: saving best checkpoint @ val_loss {losses['val'].item()}")
                    torch.save(checkpoint, best_ckpt_path)
                elif conf.training.max_num_failed_checkpoint_checks > 0:
                    # Negative max_num_failed_checkpoint_checks means we are using a different stopping method
                    # On the max_num_failed_checkpoint_checks-th failure, we end training
                    num_failed_checkpoint_checks += 1
                    if num_failed_checkpoint_checks >= conf.training.max_num_failed_checkpoint_checks:
                        should_stop = True
                
                if eval_cond_1:
                    # We save a current checkpoint every eval_interval, regardless of val_loss to ensure we can resume training
                    # without having to redo iterations.
                    pLogging.info(logger_idx, f"Training progress: saving current checkpoint @ val_loss {losses['val'].item()}")
                    latest_running_cktp_idx, latest_running_ckpt_filepath = get_latest_running_ckpt_filepath()
                    new_running_ckpt_idx = latest_running_cktp_idx + 1
                    new_running_ckpt_path = model_output_dir / f'ckpt_running_{new_running_ckpt_idx}.pt'
                    torch.save(checkpoint, new_running_ckpt_path)
                if eval_cond_2:
                    # We save a current checkpoint every epoch
                    pLogging.info(logger_idx, f"Training progress: saving epoch {epoch_num} checkpoint @ val_loss {losses['val'].item()}")
                    new_epoch_ckpt_path = model_output_dir / f'ckpt_epoch_{epoch_num}.pt'
                    torch.save(checkpoint, new_epoch_ckpt_path)
                    
            # ===== Broadcast early stop decision =====
            # Rank 0 is the only rank to decide whether training should stop but all ranked must receive the same decision.
            
            if ddp:
                stop_tensor = torch.tensor([1 if should_stop else 0], device=conf.training.device, dtype=torch.int32)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
            
            if should_stop:
                continue_training = False
                break
            
            # ===== Logging =====
            # Only master logs
            
            # if log_cond and master_process:
            #     t1 = time.time()
            #     dt = t1 - t0
            #     t0 = t1
                
            #     if losses is None:
            #         losses = estimate_loss()
                
            #     # Get loss as float. Note: this is a CPU-GPU sync point
            #     # Scale up to undo the division in loss from the training loop, approximating the true
            #     # total loss (exact would have been a sum)
            #     lossf = loss.item() * conf.training.gradient_accumulation_steps
            #     # Let the training loop settle a bit before estimating MFU
            #     if local_iter_num >= 5:
            #         mfu = raw_model.estimate_mfu(conf.training.batch_size * conf.training.gradient_accumulation_steps, dt)
            #         running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                
            #     pLogging.info(logger_idx, "Training progress", {
            #         "iter": iter_num,
            #         "loss": lossf,
            #         "train_loss": losses['train'].item(),
            #         "val_loss": losses['val'].item(),
            #         "time": dt * 1000,
            #         "mfu": running_mfu
            #     })
                
            if log_cond and master_process:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1

                # Get loss as float. Note: this is a CPU-GPU sync point
                # Scale up to undo the division in loss from the training loop, approximating the true
                # total loss (exact would be a sum)
                lossf = loss.item() * conf.training.gradient_accumulation_steps
                # Let the training loop settle a bit before estimating MFU
                if local_iter_num >= 5:
                    mfu = raw_model.estimate_mfu(conf.training.batch_size * conf.training.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

                # ===== Benchmarking gets special metrics =====
                if meta_benchmarking == True:
                    # Average step time over the last log_interval micro‑steps
                    avg_step_ms = sum(step_times[-conf.training.log_interval:]) / len(step_times[-conf.training.log_interval:]) if step_times else 0.0
                    # Tokens processed in one log interval (all micro‑steps)
                    tokens_per_interval = global_batch_size * conf.training.block_size * conf.training.log_interval
                    tokens_per_sec = tokens_per_interval / dt if dt > 0 else 0.0
                    # Peak GPU memory
                    mem_allocated = torch.cuda.max_memory_allocated(device=conf.training.device) / (1024**3)  # GiB
                    mem_reserved  = torch.cuda.max_memory_reserved(device=conf.training.device) / (1024**3)
                    # Reset memory stats for next interval
                    torch.cuda.reset_peak_memory_stats(device=conf.training.device)

                    pLogging.info(logger_idx, "Benchmarking progress", {
                        "iter": iter_num,
                        "loss": lossf,
                        "train_loss": losses['train'].item(),
                        "val_loss": losses['val'].item(),
                        "time_ms": dt * 1000,                 # wall clock since last log
                        "step_ms": avg_step_ms,               # average micro‑step time
                        "tok_per_sec": tokens_per_sec,        # throughput
                        "mem_alloc_gb": mem_allocated,        # peak allocated memory
                        "mem_res_gb": mem_reserved,           # peak reserved memory
                        "mfu": running_mfu,
                    })
                
                pLogging.info(logger_idx, "Training progress", {
                    "iter": iter_num,
                    "loss": lossf,
                    "train_loss": losses['train'].item(),
                    "val_loss": losses['val'].item(),
                    "time": dt * 1000,
                    "mfu": running_mfu
                })
            
            iter_num += 1
            local_iter_num += 1

if ddp:
    dist.barrier() # ensure all processes have finished training before master prints final message and/or cleanup
if master_process:
    pLogging.info(logger_idx, "Training finished.")
if ddp:
    destroy_process_group()