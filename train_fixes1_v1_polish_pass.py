"""
particleGPT training entrypoint.

This script supports:
  - single-GPU/debug training
  - DistributedDataParallel (DDP) training launched with torchrun
  - resumable checkpoints from running or epoch checkpoints
  - optional torch.compile
  - CUDA prefetching for the main training loader

Examples
--------
Single GPU/debug:
    python train.py --batch_size=32 --compile=False

Single-node DDP, 4 GPUs:
    torchrun --standalone --nproc_per_node=4 train.py

Multi-node DDP, 2 nodes:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
        --master_addr=<MASTER_IP> --master_port=1234 train.py
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
        --master_addr=<MASTER_IP> --master_port=1234 train.py
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import time
import socket
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import configurator as conf
import model_fixes1_v1_polish_pass as model_module
import pLogging
from model_fixes1_v1_polish_pass import GPT, GPTConfig

SCRIPT_DIR = Path(__file__).resolve().parent

PADDING_TOKEN = 0
CHECKPOINT_VERSION = 2
RUNNING_CKPT_RE = re.compile(r"^ckpt_running_(\d+)\.pt$")
EPOCH_CKPT_RE = re.compile(r"^ckpt_epoch_(\d+)\.pt$")

@dataclass(frozen=True)
class DDPState:
    """Small container for distributed-training metadata."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    master_process: bool
    seed_offset: int
    device: str

@dataclass(frozen=True)
class PrecisionState:
    """Autocast/scaler settings derived from conf.training.dtype."""

    device_type: str
    dtype: torch.dtype
    scaler: torch.amp.GradScaler

    def autocast_context(self):
        """Return a fresh context manager for the current precision mode."""
        if self.device_type == "cuda" and self.dtype == torch.float16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if self.device_type == "cuda" and self.dtype == torch.bfloat16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

class TokenBlockDataset(Dataset):
    """
    Memmap-backed dataset of fixed-length token blocks.

    Each item returns:
        x: input tokens, shape (block_size,)
        y: next-token targets, shape (block_size,)

    The final target token is padding because there is no next token inside the
    current fixed block. The model loss ignores this padding token.
    """

    def __init__(self, data_filepath: Path, block_size: int, padding_token: int = PADDING_TOKEN):
        self.data_filepath = Path(data_filepath)
        self.block_size = int(block_size)
        self.padding_token = int(padding_token)
        self._data: Optional[np.memmap] = None

        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if not self.data_filepath.exists():
            raise FileNotFoundError(f"Missing token file: {self.data_filepath}")

        data = np.memmap(self.data_filepath, dtype=np.uint16, mode="r")
        if len(data) % self.block_size != 0:
            raise ValueError(
                f"{self.data_filepath} contains {len(data):,} tokens, which is not divisible "
                f"by block_size={self.block_size}. Regenerate the prepared data or choose a "
                "compatible block_size."
            )
        self.num_samples = len(data) // self.block_size

    def _get_data(self) -> np.memmap:
        """Open the memmap lazily so each DataLoader worker owns its handle."""
        if self._data is None:
            data = np.memmap(self.data_filepath, dtype=np.uint16, mode="r")
            self._data = data.reshape(-1, self.block_size)
        return self._data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self._get_data()[index]

        # Embedding inputs must be int64/long. copy=True avoids PyTorch warnings
        # about non-writable numpy memmap buffers.
        x = torch.from_numpy(row.astype(np.int64, copy=True))

        y = torch.empty_like(x)
        y[:-1] = x[1:]
        y[-1] = self.padding_token
        return x, y

class CUDAPrefetcher:
    """
    Move the next CPU batch to GPU on a side stream while the current batch runs.

    This wrapper is only used for the training loader. Evaluation keeps the
    ordinary DataLoader path so loss estimation remains simple and deterministic.
    """

    def __init__(self, loader: DataLoader, device: str):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAPrefetcher requires CUDA.")
        self.loader = loader
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.iterator: Optional[Iterable] = None
        self.next_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def __iter__(self) -> "CUDAPrefetcher":
        self.iterator = iter(self.loader)
        self._preload()
        return self

    def _preload(self) -> None:
        assert self.iterator is not None
        try:
            x, y = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        self.next_batch = (x, y)

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.next_batch is None:
            raise StopIteration

        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        x, y = self.next_batch

        # Keep tensors alive on the stream that actually consumes them.
        x.record_stream(torch.cuda.current_stream(device=self.device))
        y.record_stream(torch.cuda.current_stream(device=self.device))

        self._preload()
        return x, y

def log_info(logger_idx: int, message: str, payload: Optional[Mapping[str, Any]] = None) -> None:
    """Thin wrapper so logging intent is obvious at call sites."""
    if payload is None:
        pLogging.info(logger_idx, message)
    else:
        pLogging.info(logger_idx, message, dict(payload))

def init_distributed() -> DDPState:
    """Initialize DDP if torchrun provided RANK/LOCAL_RANK/WORLD_SIZE."""
    ddp_enabled = int(os.environ.get("RANK", -1)) != -1

    if not ddp_enabled:
        return DDPState(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            master_process=True,
            seed_offset=0,
            device=conf.training.device,
        )

    dist.init_process_group(backend=conf.training.backend)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.cuda.is_available():
        raise RuntimeError("This DDP script expects CUDA devices when launched with torchrun.")

    device = f"cuda:{local_rank}"
    conf.training.device = device
    torch.cuda.set_device(device)

    requested_grad_accum = int(conf.training.gradient_accumulation_steps)
    if requested_grad_accum % world_size != 0:
        raise ValueError(
            "gradient_accumulation_steps must be divisible by WORLD_SIZE. "
            f"Got gradient_accumulation_steps={requested_grad_accum}, WORLD_SIZE={world_size}."
        )
    conf.training.gradient_accumulation_steps = requested_grad_accum // world_size

    return DDPState(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        master_process=(rank == 0),
        seed_offset=rank,
        device=device,
    )

def destroy_distributed(ddp_state: DDPState) -> None:
    """Cleanly tear down the process group."""
    if ddp_state.enabled and dist.is_initialized():
        dist.destroy_process_group()

def barrier(ddp_state: DDPState) -> None:
    if ddp_state.enabled:
        dist.barrier()

def broadcast_bool(value: bool, ddp_state: DDPState, src: int = 0) -> bool:
    """Broadcast a boolean decision from rank 0 to every rank."""
    if not ddp_state.enabled:
        return value
    flag = torch.tensor([1 if value else 0], device=ddp_state.device, dtype=torch.int32)
    dist.broadcast(flag, src=src)
    return bool(flag.item())

def configure_precision(device: str) -> PrecisionState:
    """Configure autocast and GradScaler from conf.training.dtype."""
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if conf.training.dtype not in dtype_map:
        raise ValueError(f"Unknown dtype {conf.training.dtype!r}; expected one of {sorted(dtype_map)}")

    dtype = dtype_map[conf.training.dtype]
    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda" and dtype == torch.float16))
    return PrecisionState(device_type=device_type, dtype=dtype, scaler=scaler)

def setup_reproducibility(seed_offset: int) -> None:
    """Set deterministic-ish seeds and allow TF32 for speed on Ampere+ GPUs."""
    torch.manual_seed(1337 + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def resolve_paths(logger_idx: int) -> Tuple[Path, Path, Path]:
    """Create/check output and preparation paths."""
    model_output_dir = SCRIPT_DIR / "trained_models" / conf.generic.model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    prep_dir = SCRIPT_DIR / "preparations" / conf.generic.preparation_name
    if not prep_dir.exists():
        log_info(logger_idx, "No preparation directory found; check conf.generic.preparation_name.")
        raise FileNotFoundError(f"Missing preparation directory: {prep_dir}")

    prep_info_filepath = prep_dir / "preparation_info.json"
    if not prep_info_filepath.exists():
        log_info(logger_idx, "No preparation_info.json found in preparation directory.")
        raise FileNotFoundError(f"Missing preparation info file: {prep_info_filepath}")

    return model_output_dir, prep_dir, prep_info_filepath

def apply_derived_config(prep_info: Mapping[str, Any]) -> Tuple[int, int]:
    """Derive block_size/vocab_size from prepared data and config."""
    prep_vocab_size = int(prep_info["vocab_size"])
    prep_max_sequence_length = int(prep_info["max_sequence_length"])

    if prep_vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(
            "Prepared data are read as uint16, but vocab_size exceeds uint16 capacity. "
            f"Got vocab_size={prep_vocab_size}. Use uint32 data or reduce the vocabulary."
        )

    if conf.training.block_size == -1 and conf.training.context_events != -1:
        conf.training.block_size = int(conf.training.context_events) * prep_max_sequence_length

    if conf.training.block_size <= 0:
        raise ValueError(
            "conf.training.block_size must be positive, or set block_size=-1 with "
            "context_events > 0 so it can be derived from the preparation."
        )

    return prep_vocab_size, prep_max_sequence_length

def namespace_to_plain_dict(namespace: Any) -> Dict[str, Any]:
    """Serialize simple public config attributes for checkpoint metadata."""
    out: Dict[str, Any] = {}
    for name in dir(namespace):
        if name.startswith("_"):
            continue
        value = getattr(namespace, name)
        if isinstance(value, (str, int, float, bool, type(None))):
            out[name] = value
    return out

def snapshot_run_config() -> Dict[str, Dict[str, Any]]:
    return {
        "generic": namespace_to_plain_dict(conf.generic),
        "training": namespace_to_plain_dict(conf.training),
    }

def find_latest_indexed_checkpoint(directory: Path, pattern: re.Pattern[str]) -> Tuple[int, Optional[Path]]:
    """Return the highest-index checkpoint matching ckpt_running_N.pt or ckpt_epoch_N.pt."""
    best_idx = -1
    best_path: Optional[Path] = None

    for path in directory.glob("*.pt"):
        match = pattern.match(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        if idx > best_idx:
            best_idx = idx
            best_path = path

    return best_idx, best_path

def get_latest_running_ckpt_filepath(model_output_dir: Path) -> Tuple[int, Optional[Path]]:
    return find_latest_indexed_checkpoint(model_output_dir, RUNNING_CKPT_RE)

def get_latest_epoch_ckpt_filepath(model_output_dir: Path) -> Tuple[int, Optional[Path]]:
    return find_latest_indexed_checkpoint(model_output_dir, EPOCH_CKPT_RE)

def torch_load_checkpoint(path: Path, device: str) -> Dict[str, Any]:
    """Load a training checkpoint with compatibility across PyTorch versions."""
    kwargs: Dict[str, Any] = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        # Optimizer checkpoints contain more than raw tensor weights.
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)

def checkpoint_resume_iter(checkpoint: Mapping[str, Any]) -> int:
    """
    Return the next optimizer iteration to run.

    Version-2 checkpoints store iter_num as the next iteration directly. Older
    versions from the original script stored the just-completed iteration, so we
    advance by one to avoid retraining that optimizer step after resume.
    """
    iter_num = int(checkpoint.get("iter_num", 0))
    version = int(checkpoint.get("checkpoint_version", 1))
    if version < CHECKPOINT_VERSION:
        return iter_num + 1
    return iter_num

def choose_resume_checkpoint(model_output_dir: Path, device: str) -> Tuple[Optional[Path], int]:
    """Pick the newest resumable checkpoint among running and epoch checkpoints."""
    if conf.training.init_from == "scratch":
        return None, 0

    _, running_path = get_latest_running_ckpt_filepath(model_output_dir)
    _, epoch_path = get_latest_epoch_ckpt_filepath(model_output_dir)

    candidates = [path for path in (running_path, epoch_path) if path is not None]
    if not candidates:
        return None, 0

    best_path: Optional[Path] = None
    best_iter = -1
    for path in candidates:
        checkpoint = torch_load_checkpoint(path, device)
        resume_iter = checkpoint_resume_iter(checkpoint)
        if resume_iter > best_iter:
            best_iter = resume_iter
            best_path = path

    return best_path, max(best_iter, 0)

def clean_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove common wrappers added by torch.compile/DDP before loading/saving."""
    cleaned: Dict[str, torch.Tensor] = {}
    prefixes = ("module.", "_orig_mod.")
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned

def unwrap_model(train_model: torch.nn.Module) -> torch.nn.Module:
    """Return the real GPT module underneath DDP and/or torch.compile wrappers."""
    module = train_model.module if isinstance(train_model, DDP) else train_model
    return getattr(module, "_orig_mod", module)

def build_model_args(prep_vocab_size: int) -> Dict[str, Any]:
    """Build GPTConfig kwargs from the training config."""
    model_args: Dict[str, Any] = {
        "n_layer": conf.training.n_layer,
        "n_head": conf.training.n_head,
        "n_embd": conf.training.n_embd,
        "n_kv_heads": conf.training.n_kv_heads if conf.training.n_kv_heads else conf.training.n_head,
        "block_size": conf.training.block_size,
        "bias": conf.training.bias,
        "vocab_size": prep_vocab_size,
        "dropout": conf.training.dropout,
    }

    optional_config_fields = (
        "mlp_type",
        "mlp_ratio",
        "qk_norm",
        "embedding_norm_type",
        "embedding_norm_init_scale",
        "use_particle_index_embeddings",
        "num_features_per_particle",
        "max_particles_per_event",
        "use_bin_value_embeddings",
        "bin_embedding_init_scale",
    )
    for field in optional_config_fields:
        if hasattr(conf.training, field):
            model_args[field] = getattr(conf.training, field)

    return model_args

def architecture_keys() -> Tuple[str, ...]:
    """Config keys that must match a checkpoint for state_dict loading to work."""
    return (
        "n_layer",
        "n_head",
        "n_embd",
        "n_kv_heads",
        "block_size",
        "bias",
        "vocab_size",
        "mlp_type",
        "mlp_ratio",
        "qk_norm",
        "embedding_norm_type",
        "use_particle_index_embeddings",
        "num_features_per_particle",
        "max_particles_per_event",
        "use_bin_value_embeddings",
    )

def initialize_or_resume_model(
    model_output_dir: Path,
    model_args: Dict[str, Any],
    logger_idx: int,
    device: str,
) -> Tuple[GPT, Dict[str, Any], Optional[Dict[str, Any]], int, float, str]:
    """
    Create a new GPT or resume one from the newest running/epoch checkpoint.

    Returns:
        base_model: unwrapped GPT instance
        model_args: final args used to build the model
        checkpoint: loaded checkpoint if resuming, else None
        iter_num: next optimizer iteration to run
        best_val_loss: best validation loss carried from checkpoint or inf
        init_source: "scratch" or "resume"
    """
    resume_path, resume_iter = choose_resume_checkpoint(model_output_dir, device)

    if resume_path is None:
        log_info(logger_idx, "Initializing a new model from scratch")
        log_info(logger_idx, "Training progress", {"info": "Initializing a new model from scratch"})
        return GPT(GPTConfig(**model_args)), model_args, None, 0, float("inf"), "scratch"

    log_info(logger_idx, f"Resuming training from {resume_path}")
    log_info(logger_idx, "Training progress", {"info": f"Resuming training from {resume_path}"})

    checkpoint = torch_load_checkpoint(resume_path, device)
    checkpoint_model_args = dict(checkpoint["model_args"])

    # Keep architecture fixed to the checkpoint. Non-architectural training choices
    # such as dropout may still come from the current config.
    for key in architecture_keys():
        if key in checkpoint_model_args:
            model_args[key] = checkpoint_model_args[key]

    base_model = GPT(GPTConfig(**model_args))
    base_model.load_state_dict(clean_state_dict_keys(checkpoint["model"]))

    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return base_model, model_args, checkpoint, resume_iter, best_val_loss, "resume"

def maybe_crop_block_size(base_model: GPT, model_args: Dict[str, Any]) -> None:
    """Crop RoPE/cache buffers if the current config asks for a shorter context."""
    if conf.training.block_size < base_model.config.block_size:
        base_model.crop_block_size(conf.training.block_size)
        model_args["block_size"] = conf.training.block_size

def build_dataloader(
    dataset: Dataset,
    ddp_state: DDPState,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Create a DataLoader with correct sampler/shuffle behavior for DDP or single GPU."""
    sampler = None
    if ddp_state.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_state.world_size,
            rank=ddp_state.rank,
            shuffle=shuffle,
            drop_last=True,
        )

    loader_kwargs: Dict[str, Any] = {
        "batch_size": conf.training.batch_size,
        "drop_last": True,
        "pin_memory": ("cuda" in conf.training.device),
        "num_workers": num_workers,
        "persistent_workers": (num_workers > 0),
        "sampler": sampler,
        "shuffle": (shuffle and sampler is None),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(getattr(conf.training, "prefetch_factor", 4))

    return DataLoader(dataset, **loader_kwargs)

def set_sampler_epoch(loader: DataLoader, epoch: int) -> None:
    """Set DistributedSampler epoch when present; no-op otherwise."""
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)

def get_epoch_from_iter(iter_num: int) -> Tuple[int, int]:
    """Map global optimizer iteration -> (epoch_index, optimizer_iter_inside_epoch)."""
    if conf.training.iterations_per_epoch <= 0:
        raise ValueError("iterations_per_epoch must be positive.")
    epoch = iter_num // conf.training.iterations_per_epoch
    iter_in_epoch = iter_num - epoch * conf.training.iterations_per_epoch
    return int(epoch), int(iter_in_epoch)

def validate_epoch_shape(train_loader: DataLoader) -> int:
    """
    Return how many micro-batches form one full epoch of optimizer steps.

    We intentionally ignore leftover micro-batches that do not complete a full
    gradient-accumulation group. This prevents partial gradients from leaking
    into the next epoch.
    """
    grad_accum = int(conf.training.gradient_accumulation_steps)
    if grad_accum <= 0:
        raise ValueError(f"gradient_accumulation_steps must be positive, got {grad_accum}")

    iterations_per_epoch = len(train_loader) // grad_accum
    if iterations_per_epoch <= 0:
        raise ValueError(
            "Not enough batches for one optimizer step. Need at least "
            f"gradient_accumulation_steps={grad_accum} batches, got len(train_loader)={len(train_loader)}."
        )
    conf.training.iterations_per_epoch = int(iterations_per_epoch)
    return int(iterations_per_epoch * grad_accum)

def configure_optimizer(base_model: GPT, precision: PrecisionState) -> torch.optim.Optimizer:
    return base_model.configure_optimizers(
        conf.training.weight_decay,
        conf.training.learning_rate,
        (conf.training.beta1, conf.training.beta2),
        precision.device_type,
    )

def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    """Ensure optimizer state tensors are on the training device after resume."""
    target_device = torch.device(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target_device)

def maybe_compile_and_wrap_ddp(base_model: GPT, ddp_state: DDPState, logger_idx: int) -> torch.nn.Module:
    """Apply torch.compile and DDP wrapping while keeping base_model available."""
    train_model: torch.nn.Module = base_model

    if conf.training.compile:
        log_info(
            logger_idx,
            "Training progress",
            {"info": "Compiling the model with torch.compile(mode='default', dynamic=False)"},
        )
        train_model = torch.compile(train_model, mode="default", dynamic=False, fullgraph=False)

    if ddp_state.enabled:
        train_model = DDP(
            train_model,
            device_ids=[ddp_state.local_rank],
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    return train_model

@torch.no_grad()
def estimate_loss(
    train_model: torch.nn.Module,
    loaders: Mapping[str, DataLoader],
    precision: PrecisionState,
    ddp_state: DDPState,
    iter_num: int,
) -> Dict[str, torch.Tensor]:
    """
    Estimate train/val loss over eval_iters batches.

    In DDP, each rank evaluates its sampler shard and we all-reduce the summed
    loss and count. This gives a true global mean rather than a fragile mean of
    per-rank means.
    """
    was_training = train_model.training
    train_model.eval()

    out: Dict[str, torch.Tensor] = {}
    current_epoch, _ = get_epoch_from_iter(iter_num)

    for split, loader in loaders.items():
        # Train eval uses current epoch for a representative shuffled sample.
        # Val eval stays deterministic by using epoch 0 and shuffle=False sampler.
        set_sampler_epoch(loader, current_epoch if split == "train" else 0)

        loss_sum = torch.zeros((), device=ddp_state.device)
        loss_count = torch.zeros((), device=ddp_state.device)

        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= conf.training.eval_iters:
                break
            x = x.to(ddp_state.device, non_blocking=True)
            y = y.to(ddp_state.device, non_blocking=True)
            with precision.autocast_context():
                _, loss, _ = train_model(x, y)
            loss_sum += loss.detach()
            loss_count += 1

        if ddp_state.enabled:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)

        if loss_count.item() == 0:
            raise RuntimeError(f"estimate_loss saw zero batches for split={split!r}.")

        out[split] = (loss_sum / loss_count).detach().cpu()

    train_model.train(was_training)
    return out

def get_lr(iter_num: int) -> float:
    """Learning-rate schedule selected by conf.training.lr_scheduler."""
    scheduler = conf.training.lr_scheduler
    learning_rate = float(conf.training.learning_rate)
    min_lr = float(conf.training.min_lr)
    warmup_iters = int(conf.training.warmup_iters)
    lr_decay_iters = int(conf.training.lr_decay_iters)

    if scheduler == "constant":
        return learning_rate

    if warmup_iters > 0 and iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters

    if lr_decay_iters <= warmup_iters:
        raise ValueError(
            f"lr_decay_iters={lr_decay_iters} must be greater than warmup_iters={warmup_iters} "
            f"for scheduler={scheduler!r}."
        )

    if scheduler == "cosine_annealing_with_warmup":
        if iter_num >= lr_decay_iters:
            return min_lr
        decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    if scheduler == "cosine_with_warmup":
        # Keep this schedule bounded instead of letting cosine continue past pi.
        decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    if scheduler == "cosine_annealing_with_warm_restarts":
        cycle_len = int(lr_decay_iters)
        if cycle_len <= 0:
            raise ValueError("lr_decay_iters must be positive for warm restarts.")

        iter_after_warmup = iter_num - warmup_iters
        cycle = 0
        iter_in_cycle = iter_after_warmup
        while iter_in_cycle >= cycle_len:
            iter_in_cycle -= cycle_len
            cycle += 1
            cycle_len = max(1, int(cycle_len * conf.training.cycle_steps_mult))

        curr_base_lr = learning_rate * (conf.training.base_lr_decay_mult ** cycle)
        t = iter_in_cycle / cycle_len
        return min_lr + 0.5 * (curr_base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    raise ValueError(f"Unknown lr_scheduler {scheduler!r}")

def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def atomic_torch_save(obj: Mapping[str, Any], path: Path) -> None:
    """Save checkpoint atomically to avoid corrupt files on interruption."""
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def build_checkpoint(
    train_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_args: Mapping[str, Any],
    iter_num: int,
    best_val_loss: float,
) -> Dict[str, Any]:
    """Create a clean, resumable checkpoint payload."""
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        # iter_num is the next optimizer iteration to run after resume.
        "iter_num": int(iter_num),
        "best_val_loss": float(best_val_loss),
        "model": clean_state_dict_keys(unwrap_model(train_model).state_dict()),
        "optimizer": optimizer.state_dict(),
        "model_args": dict(model_args),
        "config": snapshot_run_config(),
    }

def save_checkpoints_if_needed(
    train_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_args: Mapping[str, Any],
    model_output_dir: Path,
    completed_iter: int,
    epoch_num: int,
    losses: Mapping[str, torch.Tensor],
    best_val_loss: float,
    num_failed_checkpoint_checks: int,
    logger_idx: int,
    eval_due: bool,
    epoch_due: bool,
) -> Tuple[float, int, bool]:
    """
    Save best/running/epoch checkpoints on the master process.

    Returns updated (best_val_loss, num_failed_checkpoint_checks, should_stop).
    """
    should_stop = False
    val_loss = float(losses["val"].item())
    train_loss = float(losses["train"].item())

    log_info(
        logger_idx,
        "Training progress: checking checkpoint conditions",
        {"step": completed_iter, "train_loss": train_loss, "val_loss": val_loss},
    )

    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        num_failed_checkpoint_checks = 0
    elif conf.training.max_num_failed_checkpoint_checks > 0:
        num_failed_checkpoint_checks += 1
        should_stop = num_failed_checkpoint_checks >= conf.training.max_num_failed_checkpoint_checks

    checkpoint = build_checkpoint(train_model, optimizer, model_args, completed_iter, best_val_loss)

    if improved:
        best_path = model_output_dir / "ckpt.pt"
        log_info(logger_idx, f"Training progress: saving best checkpoint @ val_loss {val_loss}")
        atomic_torch_save(checkpoint, best_path)

    if eval_due:
        latest_idx, _ = get_latest_running_ckpt_filepath(model_output_dir)
        running_path = model_output_dir / f"ckpt_running_{latest_idx + 1}.pt"
        log_info(logger_idx, f"Training progress: saving current checkpoint @ val_loss {val_loss}")
        atomic_torch_save(checkpoint, running_path)

    if epoch_due:
        epoch_path = model_output_dir / f"ckpt_epoch_{epoch_num}.pt"
        log_info(logger_idx, f"Training progress: saving epoch {epoch_num} checkpoint @ val_loss {val_loss}")
        atomic_torch_save(checkpoint, epoch_path)

    return best_val_loss, num_failed_checkpoint_checks, should_stop

def log_training_progress(
    logger_idx: int,
    completed_iter: int,
    last_train_loss: torch.Tensor,
    losses: Mapping[str, torch.Tensor],
    dt: float,
    running_mfu: float,
) -> None:
    log_info(
        logger_idx,
        "Training progress",
        {
            "iter": completed_iter,
            "loss": float(last_train_loss.item()),
            "train_loss": float(losses["train"].item()),
            "val_loss": float(losses["val"].item()),
            "time": dt * 1000.0,
            "mfu": running_mfu,
        },
    )

def log_benchmark_progress(
    logger_idx: int,
    completed_iter: int,
    last_train_loss: torch.Tensor,
    losses: Mapping[str, torch.Tensor],
    dt: float,
    running_mfu: float,
    step_times_ms: Iterable[float],
    global_batch_size: int,
    iters_since_last_log: int,
    block_size: int,
    device: str,
) -> None:
    step_times = list(step_times_ms)
    avg_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
    tokens_processed = global_batch_size * block_size * max(1, iters_since_last_log)
    tokens_per_sec = tokens_processed / dt if dt > 0 else 0.0

    mem_allocated = 0.0
    mem_reserved = 0.0
    if "cuda" in device and torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated(device=device) / (1024**3)
        mem_reserved = torch.cuda.max_memory_reserved(device=device) / (1024**3)
        torch.cuda.reset_peak_memory_stats(device=device)

    log_info(
        logger_idx,
        "Benchmarking progress",
        {
            "iter": completed_iter,
            "loss": float(last_train_loss.item()),
            "train_loss": float(losses["train"].item()),
            "val_loss": float(losses["val"].item()),
            "time_ms": dt * 1000.0,
            "step_ms": avg_step_ms,
            "tok_per_sec": tokens_per_sec,
            "mem_alloc_gb": mem_allocated,
            "mem_res_gb": mem_reserved,
            "mfu": running_mfu,
        },
    )

def run_eval_only(
    train_model: torch.nn.Module,
    loaders: Mapping[str, DataLoader],
    precision: PrecisionState,
    ddp_state: DDPState,
    iter_num: int,
    logger_idx: int,
) -> None:
    """Run one loss estimate and exit."""
    losses = estimate_loss(train_model, loaders, precision, ddp_state, iter_num)
    if ddp_state.master_process:
        log_info(
            logger_idx,
            "Eval-only result",
            {
                "iter": iter_num,
                "train_loss": float(losses["train"].item()),
                "val_loss": float(losses["val"].item()),
            },
        )

def train_loop(
    train_model: torch.nn.Module,
    base_model: GPT,
    optimizer: torch.optim.Optimizer,
    loaders: Mapping[str, DataLoader],
    model_output_dir: Path,
    model_args: Mapping[str, Any],
    precision: PrecisionState,
    ddp_state: DDPState,
    logger_idx: int,
    iter_num: int,
    best_val_loss: float,
) -> None:
    """Main training loop."""
    max_micro_batches_per_epoch = validate_epoch_shape(loaders["train_step"])
    log_info(logger_idx, f"Iterations per epoch is {conf.training.iterations_per_epoch}")
    log_info(logger_idx, "iterations_per_epoch", {"iterations_per_epoch": conf.training.iterations_per_epoch})

    if conf.training.eval_only:
        run_eval_only(train_model, {"train": loaders["train_eval"], "val": loaders["val"]}, precision, ddp_state, iter_num, logger_idx)
        return

    train_model.train()
    optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    last_log_iter = iter_num
    local_iter_num = 0
    num_failed_checkpoint_checks = 0
    running_mfu = -1.0

    meta_benchmarking = bool(conf.training.meta_benchmarking)
    global_batch_size = conf.training.batch_size * conf.training.gradient_accumulation_steps * ddp_state.world_size
    step_times_ms = []

    if meta_benchmarking:
        log_info(
            logger_idx,
            "Meta benchmarking is enabled. Micro-step time, throughput, and peak memory will be logged.",
        )
        if precision.device_type == "cuda" and ddp_state.master_process:
            torch.cuda.reset_peak_memory_stats(device=ddp_state.device)

    start_epoch, start_iter_in_epoch = get_epoch_from_iter(iter_num)
    micro_batches_to_skip = start_iter_in_epoch * conf.training.gradient_accumulation_steps

    stop_training = False
    max_epochs = int(conf.training.max_epochs)
    max_iters = int(conf.training.max_iters)

    for epoch_num in range(start_epoch, max_epochs):
        if stop_training or iter_num >= max_iters:
            break

        set_sampler_epoch(loaders["train_step"], epoch_num)
        epoch_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]] = loaders["train_step"]
        if precision.device_type == "cuda":
            epoch_loader = CUDAPrefetcher(loaders["train_step"], ddp_state.device)

        for micro_step_idx, (x, y) in enumerate(epoch_loader):
            if micro_step_idx >= max_micro_batches_per_epoch:
                # Drop incomplete gradient-accumulation leftovers at the end of the epoch.
                break

            if micro_batches_to_skip > 0:
                micro_batches_to_skip -= 1
                continue

            if iter_num >= max_iters:
                stop_training = True
                break

            step_start = time.time() if meta_benchmarking else None

            if precision.device_type != "cuda":
                x = x.to(ddp_state.device)
                y = y.to(ddp_state.device)

            accum_boundary = (micro_step_idx + 1) % conf.training.gradient_accumulation_steps == 0
            if ddp_state.enabled:
                train_model.require_backward_grad_sync = accum_boundary

            set_lr(optimizer, get_lr(iter_num))

            with precision.autocast_context():
                logits, raw_loss, _ = train_model(x, y)
                del logits
                scaled_loss = raw_loss / conf.training.gradient_accumulation_steps

            precision.scaler.scale(scaled_loss).backward()

            if meta_benchmarking and step_start is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_times_ms.append((time.time() - step_start) * 1000.0)

            if not accum_boundary:
                continue

            if conf.training.grad_clip != 0.0:
                precision.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), conf.training.grad_clip)

            precision.scaler.step(optimizer)
            precision.scaler.update()
            optimizer.zero_grad(set_to_none=True)

            completed_iter = iter_num + 1
            local_iter_num += 1

            eval_due = conf.training.eval_interval > 0 and (completed_iter % conf.training.eval_interval == 0)
            epoch_due = bool(conf.training.eval_every_epoch) and (completed_iter % conf.training.iterations_per_epoch == 0)
            log_due = conf.training.log_interval > 0 and (completed_iter % conf.training.log_interval == 0)

            losses: Optional[Dict[str, torch.Tensor]] = None
            if eval_due or epoch_due or log_due:
                losses = estimate_loss(
                    train_model,
                    {"train": loaders["train_eval"], "val": loaders["val"]},
                    precision,
                    ddp_state,
                    completed_iter,
                )

            should_stop = False
            if (eval_due or epoch_due) and ddp_state.master_process:
                assert losses is not None
                best_val_loss, num_failed_checkpoint_checks, should_stop = save_checkpoints_if_needed(
                    train_model=train_model,
                    optimizer=optimizer,
                    model_args=model_args,
                    model_output_dir=model_output_dir,
                    completed_iter=completed_iter,
                    epoch_num=epoch_num,
                    losses=losses,
                    best_val_loss=best_val_loss,
                    num_failed_checkpoint_checks=num_failed_checkpoint_checks,
                    logger_idx=logger_idx,
                    eval_due=eval_due,
                    epoch_due=epoch_due,
                )

            should_stop = broadcast_bool(should_stop, ddp_state)

            if log_due and ddp_state.master_process:
                assert losses is not None
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                iters_since_last_log = completed_iter - last_log_iter
                last_log_iter = completed_iter

                if local_iter_num >= 5:
                    mfu = base_model.estimate_mfu(global_batch_size, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

                if meta_benchmarking:
                    recent_steps = step_times_ms[-conf.training.log_interval :]
                    log_benchmark_progress(
                        logger_idx=logger_idx,
                        completed_iter=completed_iter,
                        last_train_loss=raw_loss.detach().cpu(),
                        losses=losses,
                        dt=dt,
                        running_mfu=running_mfu,
                        step_times_ms=recent_steps,
                        global_batch_size=global_batch_size,
                        iters_since_last_log=iters_since_last_log,
                        block_size=conf.training.block_size,
                        device=ddp_state.device,
                    )

                log_training_progress(
                    logger_idx=logger_idx,
                    completed_iter=completed_iter,
                    last_train_loss=raw_loss.detach().cpu(),
                    losses=losses,
                    dt=dt,
                    running_mfu=running_mfu,
                )

            iter_num = completed_iter

            if should_stop:
                stop_training = True
                break

        micro_batches_to_skip = 0

    barrier(ddp_state)
    if ddp_state.master_process:
        log_info(logger_idx, "Training finished.")

def log_training_configuration(
    logger_idx: int,
    ddp_state: DDPState,
    precision: PrecisionState,
    prep_vocab_size: int,
    prep_max_sequence_length: int,
) -> None:
    """Write the most useful run metadata to the training log."""
    log_info(
        logger_idx,
        "Training configuration",
        {
            "config_file_path": conf.generic.config_file_path,
            "preparation": conf.generic.preparation_name,
            "model_name": conf.generic.model_name,
            "ddp": ddp_state.enabled,
            "ddp_rank": ddp_state.rank,
            "ddp_world_size": ddp_state.world_size,
            "device": ddp_state.device,
            "device_type": precision.device_type,
            "dtype": conf.training.dtype,
            "compile": conf.training.compile,
            "prep_vocab_size": prep_vocab_size,
            "prep_max_sequence_length": prep_max_sequence_length,
            "eval_interval": conf.training.eval_interval,
            "log_interval": conf.training.log_interval,
            "eval_iters": conf.training.eval_iters,
            "eval_only": conf.training.eval_only,
            "init_from": conf.training.init_from,
            "gradient_accumulation_steps_per_rank": conf.training.gradient_accumulation_steps,
            "batch_size_per_rank": conf.training.batch_size,
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
            "max_epochs": conf.training.max_epochs,
            "weight_decay": conf.training.weight_decay,
            "beta1": conf.training.beta1,
            "beta2": conf.training.beta2,
            "grad_clip": conf.training.grad_clip,
            "warmup_iters": conf.training.warmup_iters,
            "lr_decay_iters": conf.training.lr_decay_iters,
            "min_lr": conf.training.min_lr,
            "backend": conf.training.backend,
        },
    )
    
def find_free_port() -> int:
    """Find an available localhost port for single-node internal DDP."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

def should_launch_internal_ddp() -> bool:
    """
    Decide whether plain `python train.py` should internally spawn one process
    per GPU.

    This is intentionally disabled when torchrun already provided RANK. Using
    torchrun manually implies the user wants to control DDP themselves, likely
    to run on multiple nodes or with custom environment variables.
    """
    if int(os.environ.get("RANK", -1)) != -1:
        return False

    if os.environ.get("PARTICLEGPT_INTERNAL_DDP", "0") == "1":
        return False

    if not torch.cuda.is_available():
        return False

    if torch.cuda.device_count() <= 1:
        return False

    return bool(getattr(conf.training, "auto_ddp", False))

def internal_ddp_worker(local_rank: int, world_size: int, master_addr: str, master_port: int) -> None:
    """
    Worker entrypoint used by `python train.py` internal DDP.

    This sets the same environment variables torchrun would normally set,
    then calls the normal training main().
    """
    os.environ["PARTICLEGPT_INTERNAL_DDP"] = "1"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    main()

def launch_internal_ddp() -> None:
    """Launch single-node DDP from plain `python train.py`."""
    world_size = int(getattr(conf.training, "auto_ddp_world_size", torch.cuda.device_count()))
    world_size = min(world_size, torch.cuda.device_count())

    if world_size <= 1:
        main()
        return

    master_addr = str(getattr(conf.training, "auto_ddp_master_addr", "127.0.0.1"))
    master_port = int(getattr(conf.training, "auto_ddp_master_port", 0)) or find_free_port()

    mp.spawn(
        internal_ddp_worker,
        args=(world_size, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )

def main() -> None:
    ddp_state = init_distributed()
    logger_idx = -1

    try:
        model_output_dir = SCRIPT_DIR / "trained_models" / conf.generic.model_name
        if ddp_state.master_process:
            model_output_dir.mkdir(parents=True, exist_ok=True)
            logger_idx = pLogging.create_training_logger(conf.generic.model_name, 1)

        # Let the model module inherit the training logger. Non-master ranks use
        # -1, which preserves the old convention of silent worker ranks.
        model_module.set_logger(logger_idx)
        log_info(logger_idx, "Training started.")

        setup_reproducibility(ddp_state.seed_offset)
        precision = configure_precision(ddp_state.device)

        model_output_dir, prep_dir, prep_info_filepath = resolve_paths(logger_idx)
        prep_info = load_json(prep_info_filepath)
        prep_vocab_size, prep_max_sequence_length = apply_derived_config(prep_info)

        train_dataset = TokenBlockDataset(prep_dir / "train.bin", conf.training.block_size)
        val_dataset = TokenBlockDataset(prep_dir / "val.bin", conf.training.block_size)

        num_workers = int(getattr(conf.training, "num_workers", 4))
        train_step_loader = build_dataloader(train_dataset, ddp_state, shuffle=True, num_workers=num_workers)
        train_eval_loader = build_dataloader(train_dataset, ddp_state, shuffle=True, num_workers=num_workers)
        val_loader = build_dataloader(val_dataset, ddp_state, shuffle=False, num_workers=num_workers)

        # Set this before logging/model init so the config snapshot is useful.
        validate_epoch_shape(train_step_loader)
        log_training_configuration(logger_idx, ddp_state, precision, prep_vocab_size, prep_max_sequence_length)

        model_args = build_model_args(prep_vocab_size)
        base_model, model_args, checkpoint, iter_num, best_val_loss, init_source = initialize_or_resume_model(
            model_output_dir=model_output_dir,
            model_args=model_args,
            logger_idx=logger_idx,
            device=ddp_state.device,
        )

        maybe_crop_block_size(base_model, model_args)
        base_model.to(ddp_state.device)

        optimizer = configure_optimizer(base_model, precision)
        if init_source == "resume" and checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state_to_device(optimizer, ddp_state.device)
        checkpoint = None

        train_model = maybe_compile_and_wrap_ddp(base_model, ddp_state, logger_idx)

        train_loop(
            train_model=train_model,
            base_model=base_model,
            optimizer=optimizer,
            loaders={"train_step": train_step_loader, "train_eval": train_eval_loader, "val": val_loader},
            model_output_dir=model_output_dir,
            model_args=model_args,
            precision=precision,
            ddp_state=ddp_state,
            logger_idx=logger_idx,
            iter_num=iter_num,
            best_val_loss=best_val_loss,
        )

    finally:
        destroy_distributed(ddp_state)

if __name__ == "__main__":
    if should_launch_internal_ddp():
        launch_internal_ddp()
    else:
        main()