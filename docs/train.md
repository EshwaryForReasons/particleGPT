# `train.py`

## Overview

`train.py` is the main training entrypoint for particleGPT. It handles:

- **Single-GPU** training (debug / small-scale)
- **DistributedDataParallel (DDP)** training launched with `torchrun` (single- or multi-node)
- **Resumable checkpoints** — picks up the newest running or epoch checkpoint automatically
- **Optional `torch.compile`** for throughput
- **CUDA prefetching** for the training DataLoader via a side stream

The file is self-contained: it reads configuration via `configurator.py`, builds `TokenBlockDataset` from tokenized binary files, wraps the GPT model in DDP, runs the training loop, and writes checkpoints. It does **not** import or depend on the `analysis/` pipeline.

---

## Quick Start

```bash
# 1. Prepare a model config JSON (see configurator.py docs)
# 2. Single-GPU
python train.py config/models/my_model.json

# 3. Single-node 4-GPU DDP
torchrun --standalone --nproc_per_node=4 train.py config/models/my_model.json
```

To resume, just re-run the same command. The script will find the latest `ckpt_running_N.pt` or `ckpt_epoch_N.pt` and continue.

---

## Key Classes

### `TokenBlockDataset`

```python
class TokenBlockDataset(Dataset):
    def __init__(self, split_type: ESplitTypes)
```

A `torch.utils.data.Dataset` backed by a memory-mapped binary (`.bin`) file.

**Two modes** controlled by `use_self_contained_blocks` in the config:

| Mode | `use_extra_target_token` | How X/Y are built |
|---|---|---|
| `use_self_contained_blocks=False` (NLP default) | `True` | Reads `block_size + 1` tokens; `x = chunk[:-1]`, `y = chunk[1:]`. Blocks can span event boundaries. |
| `use_self_contained_blocks=True` (physics mode) | `False` | Reads exactly `block_size` tokens; `x = chunk`, `y[:-1] = x[1:]`, `y[-1] = PADDING`. Guarantees no block crosses an event boundary (requires `block_size` to be a multiple of `sequence_length`). |

**Constructor validation** is extensive: file size vs. metadata consistency, sequence count, token count, and split range sanity are all checked at init time. Errors surface early rather than mid-training.

**Lazy memmap**: the underlying `np.memmap` is opened in `_get_data()` the first time `__getitem__` is called. This ensures each DataLoader worker gets its own file handle (safe for `num_workers > 0`).

**Parameters**

| Param | Type | Description |
|---|---|---|
| `split_type` | `ESplitTypes` | Must be `TRAIN` or `VALIDATION` |

**Key attributes set after `__init__`**

| Attribute | Description |
|---|---|
| `split_num_samples` | Number of `(x, y)` pairs available |
| `split_start_token_idx` | Absolute index into the full tokenized array where this split begins |
| `split_end_token_idx` | Absolute index where this split ends |
| `num_split_tokens` | Usable token count (may be less than raw split tokens due to block alignment) |
| `split_tokens_dropped` | Tokens lost due to block alignment (warns if > 0) |

---

### `CUDAPrefetcher`

```python
class CUDAPrefetcher:
    def __init__(self, loader: DataLoader, device: str)
```

Wraps a `DataLoader` to overlap data transfer with computation. While the current batch is being processed on the GPU, the next batch is already being moved from CPU to GPU on a side CUDA stream.

Only used for the **training** DataLoader. Evaluation uses a plain DataLoader loop to keep loss estimation simple.

---

### `EpochSeededRandomSampler`

```python
class EpochSeededRandomSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, seed: int = 1337)
    def set_epoch(self, epoch: int) -> None
```

A deterministic random sampler for single-GPU training. Shuffles indices differently each epoch (seed + epoch), mirroring the behaviour of `DistributedSampler.set_epoch()` in DDP mode.

---

### `DDPState`

```python
@dataclass(frozen=True)
class DDPState:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    master_process: bool
    seed_offset: int
    device: str
```

Immutable container for distributed-training metadata. Centralises DDP bookkeeping so other functions only need to accept one argument.

---

### `PrecisionState`

```python
@dataclass(frozen=True)
class PrecisionState:
    device_type: str
    dtype: torch.dtype
    scaler: torch.amp.GradScaler

    def autocast_context(self): ...
```

Encapsulates autocast and GradScaler settings derived from `conf.training.dtype`. Call `autocast_context()` to get a context manager for the current precision mode.

---

### `ModelInitResult`

```python
@dataclass
class ModelInitResult:
    base_model: GPT
    model_args: Dict[str, Any]
    checkpoint: Optional[Dict[str, Any]]
    iter_num: int
    best_val_loss: float
    init_source: str                      # "scratch" or "resume"
    num_failed_checkpoint_checks: int
```

Returned by `initialize_or_resume_model()`. Bundles together everything needed to pick up training.

---

## Key Functions

### `init_distributed() → DDPState`

Detects whether `torchrun` has set `RANK`/`LOCAL_RANK`/`WORLD_SIZE` env vars and initialises `dist.init_process_group`. Also divides `gradient_accumulation_steps` by `world_size` so the effective batch size is constant regardless of the number of GPUs.

---

### `configure_precision(device: str) → PrecisionState`

Reads `conf.training.dtype` (`"float32"`, `"bfloat16"`, or `"float16"`) and creates the appropriate `GradScaler`. Returns a `PrecisionState`.

---

### `initialize_or_resume_model(model_output_dir, model_args, logger_idx, device) → ModelInitResult`

Finds the newest resumable checkpoint (running or epoch) using `choose_resume_checkpoint()`. If found, loads the state dict and validates architecture keys against the current config. If not found (or `init_from = "scratch"`), creates a fresh GPT.

Architecture keys (layer count, head count, vocab size, etc.) are **locked to the checkpoint** on resume; training-only keys like `dropout` may differ.

---

### `build_dataloader(dataset, ddp_state, shuffle, num_workers) → DataLoader`

Creates a DataLoader with the correct sampler:
- DDP: `DistributedSampler` (each rank sees a disjoint shard)
- Single-GPU + shuffle: `EpochSeededRandomSampler`
- Single-GPU + no shuffle: no sampler (sequential)

`drop_last=True` for training (ensures full gradient-accumulation groups); evaluation loaders use `drop_last=False` so no samples are wasted.

---

### `assert_no_overlap(a: TokenBlockDataset, b: TokenBlockDataset) → None`

Raises `ValueError` if the train and validation datasets read from overlapping token ranges of the same file. This is a safety check — overlapping splits would cause validation loss to be unreliable.

---

### `estimate_loss(train_model, loaders, precision, ddp_state, iter_num) → Dict[str, Tensor]`

Evaluates train and validation loss over `conf.training.eval_iters` batches. In DDP mode, sums loss across all ranks with `dist.all_reduce` before averaging — giving a true global mean rather than a fragile average of per-rank means.

---

### `get_lr(iter_num: int) → float`

Returns the learning rate for a given iteration. Supports four schedulers:

| Scheduler string | Behaviour |
|---|---|
| `"constant"` | Always `learning_rate` |
| `"cosine_annealing_with_warmup"` | Linear warmup → cosine decay → `min_lr` |
| `"cosine_with_warmup"` | Same, but without the floor at `min_lr` past `lr_decay_iters` |
| `"cosine_annealing_with_warm_restarts"` | Cosine cycles with optional cycle length scaling and base-LR decay |

---

### `save_checkpoints_if_needed(...) → Tuple[float, int, bool]`

Called after each evaluation. Saves:
- **Best checkpoint** (`ckpt.pt`): updated whenever `val_loss < best_val_loss`
- **Running checkpoint** (`ckpt_running_N.pt`): saved every `eval_interval` iterations
- **Epoch checkpoint** (`ckpt_epoch_N.pt`): saved at the end of each epoch

Also handles early stopping via `max_num_failed_checkpoint_checks`.

Returns `(new_best_val_loss, new_num_failed_checks, should_stop)`.

---

### `clean_state_dict_keys(state_dict) → Dict[str, Tensor]`

Strips `"module."` and `"_orig_mod."` prefixes introduced by DDP and `torch.compile` respectively. Call this before saving or loading any state dict.

---

### `atomic_torch_save(obj, path) → None`

Writes a checkpoint to `path.tmp` then atomically renames it to `path`. Prevents corrupt checkpoint files on interruption.

---

## Gotchas

- **`gradient_accumulation_steps` must be divisible by `world_size`**. The script enforces this and raises `ValueError` if not. Plan your batch size accordingly.

- **`block_size` must be a multiple of `sequence_length`** when `use_self_contained_blocks=True`. This ensures blocks don't cut through event boundaries. Failing this check raises `ValueError` at dataset init, not at training time.

- **Architecture drift on resume is blocked**. If you change an architectural hyperparameter (e.g. `n_layer`) in your config and try to resume from an old checkpoint, the script will override your config value with the checkpoint's. This is intentional — loading a mismatched architecture silently corrupts weights.

- **`split_tokens_dropped` warning** appears when `raw_split_tokens % block_size != 0`. A small number of tokens at the end of the split are discarded. This is normal and usually a handful of tokens; it is only a problem if your split is very small relative to `block_size`.

- **Memmap is opened lazily**. If you call `dataset[0]` before any DataLoader worker is spawned (e.g. in a test), the main process opens the memmap. In a multi-worker DataLoader, each worker opens its own. This is intentional and safe.

- **Evaluation uses epoch 0 for the val sampler**. This makes val loss deterministic across evaluations regardless of training epoch. Train eval uses the current epoch, giving a representative shuffled sample.

- **`torch.compile` with `fullgraph=False`**. Graph breaks are allowed. If you see significantly slower throughput than expected, profile with `cProfile` + `snakeviz` (see README Notes).

---

## Related

| Module | How it connects |
|---|---|
| `configurator.py` | All config loaded from `conf.generic`, `conf.training` |
| `particleGPT/model.py` | `GPT` and `GPTConfig` instantiated here |
| `preparation.py` | `DataloaderSplitConfig`, `TokenizedMetadataConfig` used by `TokenBlockDataset` |
| `dictionary.py` | Not imported directly; vocab size comes through `TokenizedMetadataConfig` |
| `pLogging.py` | All `log_info()` calls route through here |
| `sample.py` | Uses `clean_state_dict_keys()` from this module |
