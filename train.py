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

best_ckpt_path = model_output_dir / 'ckpt.pt'
running_ckpt_path = model_output_dir / 'ckpt_running.pt'

logger_idx = pLogging.create_training_logger(conf.generic.model_name, 1)
model.set_logger(logger_idx)
pLogging.info(logger_idx, 'Training started.')

data_dir = Path('data', conf.generic.preparation_name)
meta_path = Path(data_dir, 'meta.pkl')

if not meta_path.exists():
    pLogging.info(logger_idx, "No meta file found, ensure data is prepared!")
    
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
# note: float16 data type will automatically use a GradScaler
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
train_dataloader = DataLoader(
    train_dataset,
    batch_size=conf.training.batch_size,
    shuffle=False if ddp else True,
    drop_last=True,
    sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True) if ddp else None
)
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

conf.training.iterations_per_epoch = int(len(train_dataloader) // conf.training.gradient_accumulation_steps)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
num_failed_checkpoint_checks = 0

pLogging.info(logger_idx, "Training configuration", {
    "config_file_path": conf.generic.config_file_path,
    "preparation": conf.generic.preparation_name,
    "model_name": conf.generic.model_name,
    "tokens_per_iter": tokens_per_iter,
    "ddp": ddp,
    "ddp_world_size": ddp_world_size,
    "device": conf.training.device,
    "dtype": conf.training.dtype,
    "compile": conf.training.compile,
    "meta_vocab_size": meta_vocab_size,
    "meta_path": meta_path,
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
    "decay_lr": conf.training.decay_lr,
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
    block_size=conf.training.block_size,
    bias=conf.training.bias, 
    vocab_size=None, 
    dropout=conf.training.dropout
)

# Determine if we should resume or start from scratch
# This is based on if a ckpt_running.pt file exists and if the resume behavior is overridden.
init_training_from = 'scratch' # or 'resume'
if running_ckpt_path.exists() and conf.training.init_from != 'scratch':
    init_training_from = 'resume'

if init_training_from == 'scratch':
    # Init a new model from scratch
    pLogging.info(logger_idx, "Initializing a new model from scratch")
    pLogging.info(logger_idx, "Training progress", {"info": "Initializing a new model from scratch"})
    model_args['vocab_size'] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_training_from == 'resume':
    # Resume training from a checkpoint
    pLogging.info(logger_idx, f"Resuming training from {model_output_dir}")
    pLogging.info(logger_idx, "Training progress", {"info": f"Resuming training from {model_output_dir}"})
    checkpoint = torch.load(running_ckpt_path, map_location=conf.training.device)
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
if conf.training.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if conf.training.compile:
    pLogging.info(logger_idx, "Training progress", {"info": "Compiling the model"})
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Returns the epoch number from the current iteration number.
def get_epoch_from_iter(iter):
    assert conf.training.iterations_per_epoch > 0, "iterations_per_epoch must be greater than 0"
    epoch = int(iter // conf.training.iterations_per_epoch)
    remaining_iters = iter - (epoch * conf.training.iterations_per_epoch)
    return epoch, remaining_iters

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    epochs_trained_thus_far, iters_trained_thus_far = get_epoch_from_iter(iter_num)
    
    out = {}
    model.eval()
    for split, dataloader in zip(['train', 'val'], [le_train_dataloader, val_dataloader]):
        if ddp:
            dataloader.sampler.set_epoch(epochs_trained_thus_far)
        data_loader_iter = iter(dataloader)
        losses = torch.zeros(conf.training.eval_iters)
        for k in range(conf.training.eval_iters):
            x, y = next(data_loader_iter)
            with ctx:
                logits, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
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
        # Edge case: if lr_decay_iters == warmup_iters we get a division by zero error
        # I have patched this so far changing the second if to it >= lr_decay_iters (opposed so it < lr_decay_iters)
        # That not being the case was probably a bug anyway, but I'll keep track of notable changes to that.
        decay_ratio = (it - conf.training.warmup_iters) / (conf.training.lr_decay_iters - conf.training.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return conf.training.min_lr + coeff * (conf.training.learning_rate - conf.training.min_lr)
    elif conf.training.lr_scheduler == 'cosine_annealing_with_warm_restarts':
        pass

# training loop
pLogging.info(logger_idx, f'Iterations per epoch is {conf.training.iterations_per_epoch}')
pLogging.info(logger_idx, 'iterations_per_epoch', {"iterations_per_epoch": conf.training.iterations_per_epoch})
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
trained_epochs = 0 # number of epochs trained in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# Used for when we are resuming training from a checkpoint.
epochs_trained_thus_far, iters_trained_thus_far = get_epoch_from_iter(iter_num)
data_retrievals_so_far = iters_trained_thus_far * conf.training.gradient_accumulation_steps
ARBITRARY_LARGE_NUMBER = int(10e9)
for epoch_num in range(epochs_trained_thus_far, ARBITRARY_LARGE_NUMBER):
    # Update epoch so the correct seed is used for shuffling.
    if ddp:
        train_dataloader.sampler.set_epoch(epoch_num)

    # One run of this loop is essentially a micro step.
    for num_micro_steps, (X, Y) in enumerate(train_dataloader):
        # Skip over data_retrievals_so_far so we are at the correct point in the shuffle.
        if data_retrievals_so_far > 0:
            data_retrievals_so_far -= 1
            continue
            
        # Termination conditions
        if iter_num > conf.training.max_iters or (iter_num == 0 and conf.training.eval_only):
            break
        
        # Get learning rate for this iteration
        lr = get_lr(iter_num) if conf.training.decay_lr else conf.training.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
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
            # scale the loss to account for gradient accumulation
            loss = loss / conf.training.gradient_accumulation_steps 
        
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        
        # ----------------------------------------------------------------------
        # Update gradients and optimizer every gradient_accumulation_steps micro steps.
        # ----------------------------------------------------------------------
        
        if b_grad_accum_steps_completed:
            # clip the gradient
            if conf.training.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf.training.grad_clip)
            
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            
            # Save checkpoint
            if iter_num % conf.training.eval_interval == 0 and master_process:
                # Evaluate the loss on train/val sets and write checkpoints
                # We save checkpoints every eval_interval. Once in ckpt.pt with the best val_loss and again in ckpt_running.pt regardless of val_loss.
                # This is to ensure we can resume training from the running checkpoint, without having to redo many of them.
                b_save_best = False
                losses = estimate_loss()
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
                else:
                    # On the max_num_failed_checkpoint_checks-th failure, we end training
                    num_failed_checkpoint_checks += 1
                    if num_failed_checkpoint_checks >= conf.training.max_num_failed_checkpoint_checks:
                        break
                
                # We save a current checkpoint every eval_interval, regardless of val_loss to ensure we can resume training
                # without having to redo iterations.
                pLogging.info(logger_idx, f"Training progress: saving current checkpoint @ val_loss {losses['val'].item()}")
                torch.save(checkpoint, running_ckpt_path)
            
            # Log training progress
            if iter_num % conf.training.log_interval == 0 and master_process:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                # Get loss as float. Note: this is a CPU-GPU sync point
                # Scale up to undo the division in loss from the training loop, approximating the true
                # total loss (exact would have been a sum)
                lossf = loss.item() * conf.training.gradient_accumulation_steps
                # Let the training loop settle a bit before estimating MFU
                if local_iter_num >= 5:
                    mfu = raw_model.estimate_mfu(conf.training.batch_size * conf.training.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                
                losses = estimate_loss()
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

pLogging.info(logger_idx, "Training finished.")

if ddp:
    destroy_process_group()