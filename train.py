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

import json
import math
import os
import re
import time
import socket
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
from enum import Enum

import paths
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

import particleGPT.configurator as conf
import particleGPT.model as model_module
import pLogging
from particleGPT.model import GPT, GPTConfig
from particleGPT.preparation import ESplitTypes, DataloaderSplitConfig, TokenizedMetadataConfig

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

@dataclass
class ModelInitResult:
    """
    base_model: unwrapped GPT instance
    model_args: final args used to build the model
    checkpoint: loaded checkpoint if resuming, else None
    iter_num: next optimizer iteration to run
    best_val_loss: best validation loss carried from checkpoint or inf
    init_source: "scratch" or "resume"
    num_failed_checkpoint_checks: number of consecutive failed checkpoint checks, from checkpoint or 0
    """
    base_model: GPT
    model_args: Dict[str, Any]
    checkpoint: Optional[Dict[str, Any]]
    iter_num: int
    best_val_loss: float
    init_source: str
    num_failed_checkpoint_checks: int

# =====================
# Datasets, Prefetcher, Custom sampler, Overlap detection
# =====================

class TokenBlockDataset(Dataset):
    """
    Memmap-backed dataset of fixed-length token blocks.
    
    Provided a split name, this class will derive the rest from the `dataloader` section
    in the model config file.

    Each item returns:
        x: input tokens, shape (block_size,)
        y: next-token targets, shape (block_size,)

    If use_extra_target_token=True:
        The dataset reads block_size + 1 raw tokens and returns:
            x = chunk[:-1]
            y = chunk[1:]

        - In this mode, y[-1] is the real next token after the block.
        - This mode is ideal for natural language training.

    If use_extra_target_token=False:
        The dataset reads exactly block_size raw tokens and returns:
            x = chunk
            y[:-1] = x[1:]
            y[-1] = padding_token

        - In this mode, the final target token is padding because there is no next
        token inside the current fixed block. The model loss should ignore this
        padding token.
        - This mode is ideal for training on collision events because it ensures
        each block is self-contained.
    """
    
    padding_token = PADDING_TOKEN
    use_extra_target_token = True
    
    def __init__(self, split_type: ESplitTypes):
        if split_type != ESplitTypes.TRAIN and split_type != ESplitTypes.VALIDATION:
            raise ValueError("Split type has to be train or validation!")
        if conf.generic.preparation_config_file is None:
            raise ValueError("preparation_config_file in configuration cannot be None!")
        
        self.block_size = int(conf.training.block_size)
        if self.block_size <= 0:
            raise ValueError(f"block_size must be a positive integer, got {self.block_size}.")
        
        use_self_contained_blocks = getattr(conf.training, "use_self_contained_blocks", None)
        if use_self_contained_blocks is None:
            raise RuntimeError("use_self_contained_blocks config cannot be None!")
        if not isinstance(use_self_contained_blocks, bool):
            raise TypeError("use_self_contained_blocks must be a bool!")
        self.use_extra_target_token = not use_self_contained_blocks
        
        self.split_type = split_type
        
        # Load preparation config file and extract relevant data from it
        preparation_config_filepath = Path(conf.generic.preparation_config_file)
        self.dls_conf = DataloaderSplitConfig(self.split_type, preparation_config_filepath)
        if not self.dls_conf.verify():
            raise RuntimeError("Failure when verifying dataloader split config. Ensure all required arguments exist.")
        
        # Load tokenized metadata and extract relevant data from it
        tokenized_metadata_filepath = Path(self.dls_conf.tokenized_metadata_filepath)
        self.tmd_conf = TokenizedMetadataConfig(tokenized_metadata_filepath)
        if not self.tmd_conf.verify():
            raise RuntimeError("Failure when verifying tokenized metadata config. Ensure all required arguments exist.")
        
        if use_self_contained_blocks and self.block_size % self.tmd_conf.sequence_length != 0:
            raise ValueError(
                f"block_size {self.block_size} must be a multiple of sequence_length {self.tmd_conf.sequence_length} "
                "when use_self_contained_blocks is True to ensure blocks do not cross sequence boundaries."
            )
                
        # Fancy way to calculate token count in dataset without using np.memmap and len().
        # Small performance optimization
        file_bytes = self.tmd_conf.tokenized_data_filepath.stat().st_size
        dtype_bytes = np.dtype(self.tmd_conf.dtype).itemsize
        if file_bytes % dtype_bytes != 0:
            raise ValueError(
                f"Size of tokenized data file ({file_bytes} bytes) is not divisible by the size of the dtype {self.tmd_conf.dtype} "
                f"({dtype_bytes} bytes). The file may be corrupted or the metadata may be wrong."
            )
        _data_total_tokens = file_bytes // dtype_bytes
        
        # Verify data is well-formed and our metadata is reliable
        if _data_total_tokens != self.tmd_conf.total_tokens:
            raise ValueError(
                f"Tokenized data contains {_data_total_tokens:,} tokens, but expected {self.tmd_conf.total_tokens:,} "
                "based on the tokenized metadata. Regenerate the prepared data or fix the metadata."
            )
        if _data_total_tokens % self.tmd_conf.sequence_length != 0:
            raise ValueError(
                f"Tokenized data contains {_data_total_tokens:,} tokens, which is not divisible by the sequence length "
                f"{self.tmd_conf.sequence_length}. Regenerate the prepared data or fix the metadata."
            )
        _data_total_sequences = _data_total_tokens // self.tmd_conf.sequence_length
        if _data_total_sequences != self.tmd_conf.num_full_sequences:
            raise ValueError(
                f"Tokenized data contains {_data_total_sequences:,} full sequences, but expected {self.tmd_conf.num_full_sequences:,} "
                "based on the tokenized metadata. Regenerate the prepared data or fix the metadata."
            )
            
        raw_split_tokens = self.dls_conf.num_sequences * self.tmd_conf.sequence_length
        tokens_needed_per_sample = self.block_size + int(self.use_extra_target_token)
        # Ensure this preparation provides this split at least enough tokens for one sample given the configured block size
        if raw_split_tokens < tokens_needed_per_sample:
            raise ValueError(
                f"Split has {raw_split_tokens} tokens, but block_size={self.block_size} "
                f"with use_extra_target_token={self.use_extra_target_token} requires at least "
                f"{tokens_needed_per_sample} tokens."
            )
            
        if self.use_extra_target_token:
            # Need block_size + 1 raw tokens per sample: x = chunk[:-1], y = chunk[1:]
            self.split_num_samples = (raw_split_tokens - 1) // self.block_size
            self.num_split_tokens = self.split_num_samples * self.block_size + 1
        else:
            # Need block_size raw tokens per sample: x = chunk, y[:-1] = x[1:], y[-1] = PADDING
            self.split_num_samples = raw_split_tokens // self.block_size
            self.num_split_tokens = self.split_num_samples * self.block_size
            
        self.split_tokens_dropped = raw_split_tokens - self.num_split_tokens
        if self.split_tokens_dropped != 0:
            warning_txt = (f"The raw split has {raw_split_tokens} tokens, but only {self.num_split_tokens} "
                f"tokens are usable with block_size={self.block_size} and "
                f"use_extra_target_token={self.use_extra_target_token}. "
                f"Dropping {self.split_tokens_dropped} token(s) from the end of the split.")
            warnings.warn(warning_txt, RuntimeWarning)
            log_info(0, warning_txt)
        
        # Calculate raw (before block_size adjustment) start and end token indices and verify the dataset can provide it.
        # This is for verification for the config.
        # Example:
        #     total_tokens = 100
        #     sequence_length = 10
        #     skip_sequences = 8      # start at token 80
        #     num_sequences = 3       # raw request is tokens 80:110, invalid
        #     block_size = 19
        # Raw split asks for 30 tokens, but usable split becomes 20 tokens. This makes the final range
        # 80:100, which is valid, but the config is misleading because it implies 30 tokens when only 20 are usable.
        # The following check catches this style of issue.
        if self.dls_conf.from_end:
            raw_end = (self.tmd_conf.num_full_sequences - self.dls_conf.skip_sequences) * self.tmd_conf.sequence_length
            raw_start = raw_end - raw_split_tokens
        else:
            raw_start = self.dls_conf.skip_sequences * self.tmd_conf.sequence_length
            raw_end = raw_start + raw_split_tokens

        if raw_start < 0 or raw_end > self.tmd_conf.total_tokens or raw_start >= raw_end:
            raise ValueError(
                f"Invalid raw split range: raw_start={raw_start}, raw_end={raw_end}, "
                f"total_tokens={self.tmd_conf.total_tokens}."
            )
            
        self.split_start_token_idx = raw_start
        self.split_end_token_idx = raw_start + self.num_split_tokens
        
        if (self.split_start_token_idx < 0
            or self.split_end_token_idx > self.tmd_conf.total_tokens
            or self.split_start_token_idx >= self.split_end_token_idx
            or self.split_end_token_idx - self.split_start_token_idx != self.num_split_tokens
        ):
            raise ValueError(
                f"Invalid usable token range: split_start_token_idx={self.split_start_token_idx}, "
                f"split_end_token_idx={self.split_end_token_idx}, "
                f"num_split_tokens={self.num_split_tokens}, "
                f"total_tokens={self.tmd_conf.total_tokens}."
            )
        
        # Set to None so _get_data can function properly, i.e. so lazy loading works
        self._data = None

    def _get_data(self) -> np.memmap:
        """Open the memmap lazily so each DataLoader worker owns its handle."""
        if self._data is None:
            data = np.memmap(
                self.tmd_conf.tokenized_data_filepath, dtype=self.tmd_conf.dtype, mode="r"
            )
            self._data = data[self.split_start_token_idx : self.split_end_token_idx]
        return self._data

    def __len__(self) -> int:
        return self.split_num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        index: refers to the n-th sample
        """
        # These if-blocks add support for normal python style indexing
        # i.e. dataset[-1] becomes dataset[num_samples - 1], etc.
        if index < 0:
            index += self.split_num_samples
        if index < 0 or index >= self.split_num_samples:
            raise IndexError(index)
        
        local_start = index * self.block_size
        if self.use_extra_target_token:
            local_end = local_start + self.block_size + 1
            chunk = self._get_data()[local_start:local_end]
            if len(chunk) != self.block_size + 1:
                raise RuntimeError(f"Expected {self.block_size + 1} tokens, got {len(chunk)}.")
            
            # Embedding inputs must be int64/long.
            # copy=True avoids PyTorch warnings about non-writable numpy memmap buffers.
            chunk = chunk.astype(np.int64, copy=True)
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])
        else:
            local_end = local_start + self.block_size
            chunk = self._get_data()[local_start:local_end]
            if len(chunk) != self.block_size:
                raise RuntimeError(f"Expected {self.block_size} tokens, got {len(chunk)}.")

            chunk = chunk.astype(np.int64, copy=True)
            x = torch.from_numpy(chunk)
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

class EpochSeededRandomSampler(Sampler[int]):
    """
    Needed to ensure single-GPU sampling is deterministic.
    For DDP, this is handled by DistributedSampler.
    """
    def __init__(self, dataset: Dataset, seed: int = 1337):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)

def assert_no_overlap(a: TokenBlockDataset, b: TokenBlockDataset) -> None:
    a_path = Path(a.tmd_conf.tokenized_data_filepath).resolve()
    b_path = Path(b.tmd_conf.tokenized_data_filepath).resolve()
    
    # If the datasets are not the same, then probably no overlap. We do not check
    # for overlap across different files because that overlap would not be a sequence
    # index overlap, but a sequence content overlap. That is a different issue and
    # the files are expected to be verified for sequence content overlap before usage.
    if a_path != b_path:
        return

    lo = max(a.split_start_token_idx, b.split_start_token_idx)
    hi = min(a.split_end_token_idx, b.split_end_token_idx)

    if lo < hi:
        raise ValueError(
            "Dataset split overlap detected: "
            f"a=[{a.split_start_token_idx}, {a.split_end_token_idx}), "
            f"b=[{b.split_start_token_idx}, {b.split_end_token_idx}), "
            f"overlap=[{lo}, {hi})"
        )


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
    if requested_grad_accum <= 0:
        raise ValueError(f"gradient_accumulation_steps must be a positive integer, got {requested_grad_accum}.")
    if requested_grad_accum % world_size != 0:
        raise ValueError(
            "gradient_accumulation_steps must be divisible by WORLD_SIZE. "
            f"Got gradient_accumulation_steps={requested_grad_accum}, WORLD_SIZE={world_size}."
        )
    
    per_rank_grad_accum = requested_grad_accum // world_size
    if per_rank_grad_accum <= 0:
        raise ValueError(
            f"Per-rank gradient accumulation became {per_rank_grad_accum}; "
            f"got global gradient_accumulation_steps={requested_grad_accum}, world_size={world_size}."
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

def choose_resume_checkpoint(model_output_dir: Path) -> Tuple[Optional[Path], int]:
    """Pick the newest resumable checkpoint among running and epoch checkpoints."""
    if conf.training.init_from == "scratch":
        return None, 0

    _, running_path = find_latest_indexed_checkpoint(model_output_dir, RUNNING_CKPT_RE)
    _, epoch_path = find_latest_indexed_checkpoint(model_output_dir, EPOCH_CKPT_RE)

    candidates = [path for path in (running_path, epoch_path) if path is not None]
    if not candidates:
        return None, 0

    best_path: Optional[Path] = None
    best_iter = -1
    for path in candidates:
        # Open with CPU to avoid GPU transfers.
        # @TODO: consider using a custom metadata file alongside each checkpoint to avoid loading 
        #   the full checkpoint just to read the iteration number.
        # @TODO: for very large models, perhaps using FakeTensors will be ideal.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        resume_iter = checkpoint_resume_iter(checkpoint)
        del checkpoint

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
        # --
        "data_mode": conf.generic.data_mode,
        "mlp_type": conf.generic.mlp_type,
        "mlp_ratio": conf.generic.mlp_ratio,
        "qk_norm": conf.generic.qk_norm,
        "embedding_norm_type": conf.generic.embd_norm_type,
        "embedding_norm_init_scale": conf.generic.embd_norm_init_scale,
        "use_particle_index_embeddings": conf.generic.use_particle_index_embd,
        "use_bin_value_embeddings": conf.generic.use_bin_value_embd,
        "bin_embedding_init_scale": conf.generic.bin_value_embd_init_scale,
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
    
    # Maybe override model args
    for field in optional_config_fields:
        if not hasattr(conf.training, field):
            continue

        training_value = getattr(conf.training, field)
        # Treat None as "not specified in training config"
        if training_value is None:
            continue
        
        existing_value = model_args.get(field)
        # Do not allow accidental architecture drift.
        # @TODO: maybe add a flag that overlooks this check and allows for architecture to change
        #   across runs.
        if existing_value is not None and training_value != existing_value:
            raise ValueError(
                f"Conflicting model arg {field!r}: generic/config value={existing_value!r}, "
                f"training value={training_value!r}."
            )

        model_args[field] = training_value
                            
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
        "data_mode",
        "embedding_norm_init_scale",
        "bin_embedding_init_scale",
    )

def initialize_or_resume_model(model_output_dir: Path, model_args: Dict[str, Any], logger_idx: int, device: str) -> ModelInitResult:
    """
    Create a new GPT or resume one from the newest running/epoch checkpoint.
    Loads the model on CPU in case of resume to avoid GPU memory spikes.
    """
    resume_path, resume_iter = choose_resume_checkpoint(model_output_dir)
    if resume_path is None:
        log_info(logger_idx, "Initializing a new model from scratch")
        log_info(logger_idx, "Training progress", {"info": "Initializing a new model from scratch"})
        init_result = ModelInitResult(
            base_model=GPT(GPTConfig(**model_args)),
            model_args=model_args,
            checkpoint=None,
            iter_num=0,
            best_val_loss=float("inf"),
            init_source="scratch",
            num_failed_checkpoint_checks=0,
        )
        return init_result

    log_info(logger_idx, f"Resuming training from {resume_path}")
    log_info(logger_idx, "Training progress", {"info": f"Resuming training from {resume_path}"})

    # Load on CPU first to avoid memory spike. This is before model and optimizer are fully constructed.
    checkpoint = torch.load(resume_path, **{"map_location": "cpu", "weights_only": False})
    checkpoint_model_args = dict(checkpoint["model_args"])

    # Keep architecture fixed to the checkpoint. Non-architectural training choices
    # such as dropout may still come from the current config.
    for key in architecture_keys():
        if key not in checkpoint_model_args:
            continue

        current_value = model_args.get(key)
        checkpoint_value = checkpoint_model_args[key]
        if current_value != checkpoint_value:
            log_info(
                logger_idx,
                "Overriding current architecture config with checkpoint value",
                {"key": key, "current": current_value, "checkpoint": checkpoint_value},
            )

        model_args[key] = checkpoint_value

    base_model = GPT(GPTConfig(**model_args))
    base_model.load_state_dict(clean_state_dict_keys(checkpoint["model"]))

    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    num_failed_checkpoint_checks = int(checkpoint.get("num_failed_checkpoint_checks", 0))
    init_result = ModelInitResult(
        base_model=base_model,
        model_args=model_args,
        checkpoint=checkpoint,
        iter_num=resume_iter,
        best_val_loss=best_val_loss,
        init_source="resume",
        num_failed_checkpoint_checks=num_failed_checkpoint_checks,
    )
    return init_result

def build_dataloader(dataset: Dataset, ddp_state: DDPState, shuffle: bool, num_workers: int) -> DataLoader:
    """
    Create a DataLoader with correct sampler/shuffle behavior for DDP or single GPU.
    
    drop_last is added to allow eval datasets to not drop samples. This is useful when validation datasets
        are small and we want to use every sample for evaluation, even if it means the last batch is smaller
        than the training batch size.
    """
    
    sampler = None
    if ddp_state.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_state.world_size,
            rank=ddp_state.rank,
            shuffle=shuffle,
            drop_last=True,
        )
    elif shuffle:
        training_seed = int(getattr(conf.training, "seed", 1337))
        sampler = EpochSeededRandomSampler(dataset, seed=training_seed)

    loader_kwargs: Dict[str, Any] = {
        "batch_size": conf.training.batch_size,
        "drop_last": True,
        "pin_memory": ("cuda" in conf.training.device),
        "num_workers": num_workers,
        "persistent_workers": (num_workers > 0),
        "sampler": sampler,
        "shuffle": False if sampler is not None else shuffle,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(getattr(conf.training, "prefetch_factor", 4))

    return DataLoader(dataset, **loader_kwargs)

def set_sampler_epoch(loader: DataLoader, epoch: int) -> None:
    """Set DistributedSampler epoch when present; no-op otherwise."""
    sampler = getattr(loader, "sampler", None)
    # if isinstance(sampler, DistributedSampler):
    if hasattr(sampler, "set_epoch"):
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

def atomic_torch_save(obj: Mapping[str, Any], path: Path) -> None:
    """Save checkpoint atomically to avoid corrupt files on interruption."""
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

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

    improved = val_loss < (best_val_loss - conf.training.min_val_loss_improvement_criteria)
    if improved:
        best_val_loss = val_loss
        num_failed_checkpoint_checks = 0
    elif conf.training.max_num_failed_checkpoint_checks > 0:
        num_failed_checkpoint_checks += 1
        should_stop = num_failed_checkpoint_checks >= conf.training.max_num_failed_checkpoint_checks
        
    # Unwrap model--return the real GPT module underneath DDP and/or torch.compile wrappers
    module = train_model.module if isinstance(train_model, DDP) else train_model
    unwrapped_model = getattr(module, "_orig_mod", module)

    checkpoint =  {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": clean_state_dict_keys(unwrapped_model.state_dict()),
        "model_args": dict(model_args),
        # iter_num is the next optimizer iteration to run after resume.
        "iter_num": int(completed_iter),
        "best_val_loss": float(best_val_loss),
        "num_failed_checkpoint_checks": num_failed_checkpoint_checks,
        "optimizer": optimizer.state_dict(),
        "config": snapshot_run_config(),
    }

    if improved:
        best_path = model_output_dir / "ckpt.pt"
        log_info(logger_idx, f"Training progress: saving best checkpoint @ val_loss {val_loss}")
        atomic_torch_save(checkpoint, best_path)

    if eval_due:
        latest_idx, _ = find_latest_indexed_checkpoint(model_output_dir, RUNNING_CKPT_RE)
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
    num_failed_checkpoint_checks: int
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
                
            # Set learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = get_lr(iter_num)

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
                
            # Broadcast the boolean decision from rank 0 to every rank
            if ddp_state.enabled:
                flag = torch.tensor([1 if should_stop else 0], device=ddp_state.device, dtype=torch.int32)
                dist.broadcast(flag, src=0)
                should_stop = bool(flag.item())

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

    if ddp_state.enabled:
        dist.barrier()
        
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

# =====================
# Auto DDP
# =====================

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

    mp.spawn(internal_ddp_worker, args=(world_size, master_addr, master_port), nprocs=world_size, join=True)


def main() -> None:
    ddp_state = init_distributed()
    logger_idx = -1

    try:
        model_output_dir = paths.PROJECT_DIR / "trained_models" / conf.generic.model_name
        if ddp_state.master_process:
            model_output_dir.mkdir(parents=True, exist_ok=True)
            logger_idx = pLogging.create_training_logger(conf.generic.model_name, 1)
        
        # Wait while the directory is created
        if ddp_state.enabled:
            dist.barrier()

        # Let the model module inherit the training logger. Non-master ranks use
        # -1, which preserves the old convention of silent worker ranks.
        model_module.set_logger(logger_idx)
        log_info(logger_idx, "Training started.")
        
        # Setup reproducibility--set deterministic-ish seeds and allow TF32 for speed on Ampere+ GPUs."""
        base_seed = int(getattr(conf.training, "seed", 1337))
        torch.manual_seed(base_seed + ddp_state.seed_offset)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed + ddp_state.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        precision = configure_precision(ddp_state.device)

        if conf.generic.preparation_config_file is None:
            raise ValueError("preparation_config_file in configuration cannot be None!")
        
        # Load preparation config file and extract relevant data from it
        preparation_config_filepath = Path(conf.generic.preparation_config_file)
        dls_conf = DataloaderSplitConfig(ESplitTypes.TRAIN, preparation_config_filepath)
        if not dls_conf.verify():
            raise RuntimeError("Failure when verifying dataloader split config. Ensure all required arguments exist.")
        
        # Load tokenized metadata and extract relevant data from it
        tokenized_metadata_filepath = Path(dls_conf.tokenized_metadata_filepath)
        tmd_conf = TokenizedMetadataConfig(tokenized_metadata_filepath)
        if not tmd_conf.verify():
            raise RuntimeError("Failure when verifying tokenized metadata config. Ensure all required arguments exist.")
        
        if conf.training.block_size == -1 and conf.training.context_sequences != -1:
            conf.training.block_size = conf.training.context_sequences * tmd_conf.sequence_length
        if conf.training.block_size <= 0:
            raise ValueError(
                f"Invalid block_size {conf.training.block_size}. Must be positive or -1 to infer from context_sequences."
            )
        
        train_dataset = TokenBlockDataset(ESplitTypes.TRAIN)
        val_dataset = TokenBlockDataset(ESplitTypes.VALIDATION)
        assert_no_overlap(train_dataset, val_dataset)
        
        num_workers = int(getattr(conf.training, "num_workers", 4))
        train_step_loader = build_dataloader(train_dataset, ddp_state, shuffle=True, num_workers=num_workers)
        train_eval_loader = build_dataloader(train_dataset, ddp_state, shuffle=True, num_workers=num_workers)
        val_loader = build_dataloader(val_dataset, ddp_state, shuffle=False, num_workers=num_workers)

        # Set this before logging/model init so the config snapshot is useful.
        validate_epoch_shape(train_step_loader)
        log_training_configuration(
            logger_idx, ddp_state, precision, 
            train_dataset.tmd_conf.vocab_size,
            train_dataset.tmd_conf.sequence_length
        )

        model_args = build_model_args(train_dataset.tmd_conf.vocab_size)
        init_result = initialize_or_resume_model(model_output_dir=model_output_dir, model_args=model_args, logger_idx=logger_idx, device=ddp_state.device)
        
        # maybe crop block size--Crop RoPE/cache buffers if the current config asks for a shorter context.
        if conf.training.block_size < init_result.base_model.config.block_size:
            init_result.base_model.crop_block_size(conf.training.block_size)
            init_result.model_args["block_size"] = conf.training.block_size
            
        init_result.base_model.to(ddp_state.device)
        optimizer = init_result.base_model.configure_optimizers(
            conf.training.weight_decay,
            conf.training.learning_rate,
            (conf.training.beta1, conf.training.beta2),
            precision.device_type,
        )
            
        if init_result.init_source == "resume" and init_result.checkpoint is not None:
            optimizer.load_state_dict(init_result.checkpoint["optimizer"])
            move_optimizer_state_to_device(optimizer, ddp_state.device)
        init_result.checkpoint = None

        train_model = maybe_compile_and_wrap_ddp(init_result.base_model, ddp_state, logger_idx)

        train_loop(
            train_model=train_model,
            base_model=init_result.base_model,
            optimizer=optimizer,
            loaders={"train_step": train_step_loader, "train_eval": train_eval_loader, "val": val_loader},
            model_output_dir=model_output_dir,
            model_args=init_result.model_args,
            precision=precision,
            ddp_state=ddp_state,
            logger_idx=logger_idx,
            iter_num=init_result.iter_num,
            best_val_loss=init_result.best_val_loss,
            num_failed_checkpoint_checks=init_result.num_failed_checkpoint_checks
        )

    finally:
        # Cleanly tear down the process group
        if ddp_state.enabled and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    if should_launch_internal_ddp():
        launch_internal_ddp()
    else:
        main()