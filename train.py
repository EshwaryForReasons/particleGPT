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

import model
from model import GPTConfig, GPT
import pLogging
# import prepare

script_dir = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# data
preparation_name = ''
dataset = ''
model_name = ""
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = -1
context_particles = 1 # Use this instead of block_size. This updates the block_size to be context_particles * max_sequence_length from the meta file
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
max_num_failed_checkpoint_checks = 4 # number of failed checkpoint checks before ending training
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
iterations_per_epoch = 0 # This is needed to calculate epochs trained
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Ensure the output directory exists. Need to do this before creating the logger.
out_dir = os.path.join(script_dir, "trained_models", model_name)
os.makedirs(out_dir, exist_ok=True)

logger_idx = pLogging.create_training_logger(model_name, 1)
model.set_logger(logger_idx)
pLogging.info(logger_idx, 'Training started.')

data_dir = Path('data', preparation_name)
meta_path = Path(data_dir, 'meta.pkl')

if not meta_path.exists():
    pLogging.info(logger_idx, "No meta file found, ensure data is prepared!")
    
# attempt to derive vocab_size from the dataset
meta_vocab_size = None
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
meta_vocab_size = meta['vocab_size']

# If block_size is not set, then use context_particles to set it
if block_size == -1 and context_particles != -1:
    block_size = context_particles * meta['max_sequence_length']

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    if not master_process:
        logger_idx = -1 # disable logging for non-master processes
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
def get_batch(split):
    global iterations_per_epoch
    
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        num_train_samples = len(data)
        iterations_per_epoch = num_train_samples // (batch_size * block_size)
        # pLogging.info(logger_idx, f'Iterations per epoch is now {iterations_per_epoch}')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
num_failed_checkpoint_checks = 0
    
pLogging.info(logger_idx, "Training configuration", {
    "config_file_path": config_file_path,
    "preparation": preparation_name,
    "model_name": model_name,
    "tokens_per_iter": tokens_per_iter,
    "ddp": ddp,
    "ddp_world_size": ddp_world_size,
    "device": device,
    "dtype": dtype,
    "compile": compile,
    "meta_vocab_size": meta_vocab_size,
    "meta_path": meta_path,
    "eval_interval": eval_interval,
    "log_interval": log_interval,
    "eval_iters": eval_iters,
    "eval_only": eval_only,
    "always_save_checkpoint": always_save_checkpoint,
    "init_from": init_from,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "batch_size": batch_size,
    "block_size": block_size,
    "n_layer": n_layer,
    "n_head": n_head,
    "n_embd": n_embd,
    "dropout": dropout,
    "bias": bias,
    "learning_rate": learning_rate,
    "max_iters": max_iters,
    "weight_decay": weight_decay,
    "beta1": beta1,
    "beta2": beta2,
    "grad_clip": grad_clip,
    "decay_lr": decay_lr,
    "warmup_iters": warmup_iters,
    "lr_decay_iters": lr_decay_iters,
    "min_lr": min_lr,
    "backend": backend,
    "device_type": device_type
})

best_ckpt_path = Path(out_dir, 'ckpt.pt')
running_ckpt_path = Path(out_dir, 'ckpt_running.pt')

# model init; start with model_args from command line
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    # Init a new model from scratch
    pLogging.info(logger_idx, "Training progress", {"info": "Initializing a new model from scratch"})
    model_args['vocab_size'] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # Resume training from a checkpoint
    pLogging.info(logger_idx, "Training progress", {"info": f"Resuming training from {out_dir}"})
    checkpoint = torch.load(running_ckpt_path, map_location=device)
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
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    pLogging.info(logger_idx, "Training progress", {"info": "Compiling the model"})
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it >= lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    # Edge case: if lr_decay_iters == warmup_iters we get a division by zero error
    # I have patched this so far changing the second if to it >= lr_decay_iters (opposed so it < lr_decay_iters)
    # That not being the case was probably a bug anyway, but I'll keep track of notable changes to that.
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
X, Y = get_batch('train') # fetch the very first batch
pLogging.info(logger_idx, f'Iterations per epoch is {iterations_per_epoch}')
pLogging.info(logger_idx, 'iterations_per_epoch', {"iterations_per_epoch": iterations_per_epoch})
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
trained_epochs = 0 # number of epochs trained in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    # We save checkpoints every eval_interval. Once in ckpt.pt with the best val_loss and again in ckpt_running.pt regardless of val_loss.
    # This is to ensure we can resume training from the running checkpoint, without having to redo many of them.
    if iter_num % eval_interval == 0 and master_process:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        losses = estimate_loss()
        pLogging.info(logger_idx, "Training progress: checking checkpoint conditions", {
            "step": iter_num,
            "train_loss": losses['train'].item(),
            "val_loss": losses['val'].item()
        })
        
        if losses['val'] < best_val_loss:
            num_failed_checkpoint_checks = 0
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                pLogging.info(logger_idx, f"Training progress: saving best checkpoint @ val_loss {losses['val'].item()}")
                torch.save(checkpoint, best_ckpt_path)
        else:
            # On the max_num_failed_checkpoint_checks failure, we end training
            num_failed_checkpoint_checks += 1
            if num_failed_checkpoint_checks >= max_num_failed_checkpoint_checks:
                break
        
        pLogging.info(logger_idx, f"Training progress: saving current checkpoint @ val_loss {losses['val'].item()}")
        torch.save(checkpoint, running_ckpt_path)

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    if iter_num % log_interval == 0 and master_process:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        # Get loss as float. Note: this is a CPU-GPU sync point
        # Scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # Let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
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
    
    if iter_num % iterations_per_epoch == 0:
        trained_epochs += 1
        pLogging.info(logger_idx, f"Training progress: Finished training epoch", {"epochs_trained": trained_epochs})

    # termination conditions
    if iter_num > max_iters:
        break
    
pLogging.info(logger_idx, "Training finished.")

if ddp:
    destroy_process_group()