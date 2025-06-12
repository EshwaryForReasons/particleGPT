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

script_dir = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Ensure the output directory exists. Need to do this before creating the logger.
out_dir = os.path.join(script_dir, "trained_models", conf.generic.model_name)
os.makedirs(out_dir, exist_ok=True)

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
    
def setup_ddp():
    global master_process, seed_offset, ddp_rank, ddp_local_rank, ddp_world_size
    
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
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in conf.training.device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[conf.training.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

class pDataset(Dataset):
    def __init__(self, dataset_filepath):
        self.data_filepath = dataset_filepath
        assert dataset_filepath.exists(), f"Dataset file {dataset_filepath} does not exist!"
        data = np.memmap(self.data_filepath, dtype=np.uint16, mode='r')
        assert len(data) % conf.training.block_size == 0, "Data length is not a multiple of block_size!"
        self.num_samples = len(data) // conf.training.block_size
    
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

train_dataset = pDataset(data_dir / 'train.bin')
val_dataset = pDataset(data_dir / 'val.bin')
# Data loader used for training.
train_dataloader = DataLoader(
    train_dataset,
    batch_size=conf.training.batch_size,
    shuffle=False,
    drop_last=True,
    sampler=DistributedSampler(train_dataset) if ddp else None
)
# Data loaders for loss estimations so the training loaders remain unaffected.
le_train_dataloader = DataLoader(
    train_dataset,
    batch_size=conf.training.batch_size,
    shuffle=False,
    drop_last=True,
    sampler=DistributedSampler(train_dataset) if ddp else None
)
le_val_dataloader = DataLoader(
    val_dataset,
    batch_size=conf.training.batch_size,
    shuffle=False,
    drop_last=True,
    sampler=DistributedSampler(val_dataset) if ddp else None
)

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

best_ckpt_path = Path(out_dir, 'ckpt.pt')
running_ckpt_path = Path(out_dir, 'ckpt_running.pt')

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
if conf.training.init_from == 'scratch':
    # Init a new model from scratch
    pLogging.info(logger_idx, "Training progress", {"info": "Initializing a new model from scratch"})
    model_args['vocab_size'] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif conf.training.init_from == 'resume':
    # Resume training from a checkpoint
    pLogging.info(logger_idx, "Training progress", {"info": f"Resuming training from {out_dir}"})
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



# training loop
pLogging.info(logger_idx, f'Iterations per epoch is {conf.training.iterations_per_epoch}')
pLogging.info(logger_idx, 'iterations_per_epoch', {"iterations_per_epoch": conf.training.iterations_per_epoch})
 
raw_model = model.module if ddp else model # unwrap DDP container if needed

class Trainer:
    def __init__(self):
        self.running_mfu = -1.0
        self.t0 = time.time()
        self.local_iter_num = 0 # number of iterations in the lifetime of this process
        self.trained_epochs = 0 # number of epochs trained in the lifetime of this process
        self.best_val_loss = 1e9 # best validation loss seen so far
        self.loss = None # last loss value
        self.iter_num = iter_num # global iteration number, shared across processes
        self.num_failed_checkpoint_checks = 0
        self.b_early_stop_flag = False
        
    @torch.no_grad()
    def estimate_loss():
        """
        Estimate an arbitrarily accurate loss over either split using many batches.
        """
        out = {}
        model.eval()
        for split, dataloader in zip(['train', 'val'], [le_train_dataloader, le_val_dataloader]):
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
        
    def log_training_progress(self):
        # Ensure logging is restricted to the master process
        if not master_process:
            return
        
        t1 = time.time()
        dt = t1 - self.t0
        self.t0 = t1
        # Get loss as float. Note: this is a CPU-GPU sync point
        # Scale up to undo the division in loss from the training loop, approximating the true
        # total loss (exact would have been a sum)
        lossf = self.loss.item() * conf.training.gradient_accumulation_steps
        # Let the training loop settle a bit before estimating MFU
        if self.local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(conf.training.batch_size * conf.training.gradient_accumulation_steps, dt)
            self.running_mfu = mfu if self.running_mfu == -1.0 else 0.9 * self.running_mfu + 0.1 * mfu
        
        losses = self.estimate_loss()
        pLogging.info(logger_idx, "Training progress", {
            "iter": iter_num,
            "loss": lossf,
            "train_loss": losses['train'].item(),
            "val_loss": losses['val'].item(),
            "time": dt * 1000,
            "mfu": self.running_mfu
        })

    def save_checkpoint(self):
        """
        Evaluate the loss on train/val sets and write checkpoints
        We save checkpoints every eval_interval. Once in ckpt.pt with the best val_loss and again in ckpt_running.pt regardless of val_loss.
        This is to ensure we can resume training from the running checkpoint, without having to redo many of them.
        """
        
        # Ensure checkpointing is restricted to the master process
        if not master_process:
            return
        
        b_save_best = False
        losses = self.estimate_loss()
        if losses['val'] < self.best_val_loss:
            b_save_best = True
            self.best_val_loss = losses['val']
        
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': self.best_val_loss,
            'config': config
        }
        
        pLogging.info(logger_idx, "Training progress: checking checkpoint conditions", {
            "step": iter_num,
            "train_loss": losses['train'].item(),
            "val_loss": losses['val'].item()
        })
        
        if b_save_best and iter_num > 0:
            self.num_failed_checkpoint_checks = 0
            pLogging.info(logger_idx, f"Training progress: saving best checkpoint @ val_loss {losses['val'].item()}")
            torch.save(checkpoint, best_ckpt_path)
        else:
            # On the max_num_failed_checkpoint_checks-th failure, we end training
            self.num_failed_checkpoint_checks += 1
            if self.num_failed_checkpoint_checks >= conf.training.max_num_failed_checkpoint_checks:
                self.b_early_stop_flag = True
        
        # We save a current checkpoint every eval_interval, regardless of val_loss to ensure we can resume training
        # without having to redo iterations.
        pLogging.info(logger_idx, f"Training progress: saving current checkpoint @ val_loss {losses['val'].item()}")
        torch.save(checkpoint, running_ckpt_path)
        
    def do_training(self):
        # One run of this loop is essentially a micro step
        for num_micro_steps, (X, Y) in enumerate(train_dataloader):
            # Termination conditions
            if self.b_early_stop_flag or iter_num > conf.training.max_iters or (iter_num == 0 and conf.training.eval_only):
                break
            
            # Get learning rate for this iteration
            lr = self.get_lr(iter_num) if conf.training.decay_lr else conf.training.learning_rate
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
                
                if iter_num % conf.training.eval_interval == 0 and master_process:
                    self.save_checkpoint()
                
                if iter_num % conf.training.log_interval == 0 and master_process:
                    self.log_training_progress()
                
                iter_num += 1
                local_iter_num += 1

pLogging.info(logger_idx, "Training finished.")

if ddp:
    destroy_process_group()