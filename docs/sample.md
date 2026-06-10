# `sample.py`

## Overview

`sample.py` is the generation entrypoint. It loads a trained checkpoint and generates new particle events by:

1. Reading test-split sequences from the tokenized binary
2. Taking the first `starter_tokens` (default: 5) of each test event as a prompt
3. Running batched autoregressive generation on one or more GPUs
4. Writing one CSV per GPU worker, then concatenating shards into a final `samples.csv`

The output file lives at:
```
generated_samples/<model_name>/sampling_<N>/samples.csv
```

Each row is a **space-separated sequence of integer token IDs**, including the starter tokens at the front.

---

## Quick Start

```bash
# 1. Make sure a trained checkpoint exists at trained_models/<model_name>/ckpt.pt

# 2. Sample using your model config
python sample.py config/models/my_model.json

# 3. Inspect the output
head -1 generated_samples/my_model/sampling_0/samples.csv
# → 3 12 87 245 301 88 ... (space-separated token IDs)
```

---

## Key Functions

### `main() → None`

The top-level orchestrator. Reads preparation/tokenized-metadata configs, validates the checkpoint vocab size against the tokenized data, resolves devices, builds the job dict, spawns workers, and concatenates shards.

Key decisions made in `main`:
- How many test sequences to use (`conf.sampling.max_test_sequences` caps this)
- Which devices to use (`resolve_device_names`)
- Which `sampling_<N>` directory index to write to (`resolve_sample_idx`)
- Whether to keep per-worker shard files after concatenation (`conf.sampling.keep_shards`)

A `sampling_metadata.json` is also written alongside `samples.csv`, capturing all sampling hyperparameters for reproducibility.

---

### `sampling_worker(rank: int, job: dict) → None`

The function that runs on each GPU worker (spawned via `torch.multiprocessing.spawn`). Must be a top-level function (not a lambda or nested function) for pickling to work.

Each worker:
1. Sets device and seeds for reproducibility
2. Memory-maps the tokenized binary
3. Loads the model onto its device
4. Iterates over its assigned slice of test sequences in batches
5. Calls `generate_batch()` for each batch
6. Writes results line-by-line to its shard CSV

Workers divide the total test sequences as evenly as possible using integer division: worker `r` handles sequences `[total * r // world_size, total * (r+1) // world_size)`.

**Parameters**

| Param | Type | Description |
|---|---|---|
| `rank` | `int` | Worker index (0-based) |
| `job` | `dict` | Serialised job configuration (see below) |

---

### `generate_batch(model, idx, max_new_tokens, temperature, top_k, require_batch_generate) → Tensor`

Dispatches batched generation to the model.

Checks for `generate_batch`, `batch_generate`, or `generate_batched` methods on the model in that order. If none exist, falls back to `model.generate(idx, ...)` which works for any nanoGPT-style model when `idx` is 2-D (batch × tokens).

If `require_batch_generate=True` and no explicit batch method is found, raises `AttributeError`.

**Parameters**

| Param | Type | Description |
|---|---|---|
| `model` | `nn.Module` | Loaded model in eval mode |
| `idx` | `Tensor` | Shape `(B, starter_tokens)` — the starter tokens for each event in the batch |
| `max_new_tokens` | `int` | Number of new tokens to generate per event |
| `temperature` | `float` | Softmax temperature; < 1 sharpens, > 1 flattens the distribution |
| `top_k` | `Optional[int]` | If set, restricts sampling to the top-k logits |
| `require_batch_generate` | `bool` | If `True`, raises if no explicit batch method exists |

**Returns**: `Tensor` of shape `(B, starter_tokens + max_new_tokens)`

---

### `load_model_for_sampling(checkpoint_path, device, compile_model) → nn.Module`

Loads a checkpoint from disk (always to CPU first, then moves to device). Cleans state dict keys, instantiates `GPT`, loads state, and optionally compiles with `torch.compile`. Sets the model to eval mode.

---

### `resolve_device_names(total_starters: int) → tuple[str, ...]`

Determines which devices to use. Priority order:

1. If `conf.sampling.device` is `"cpu"` → `("cpu",)`
2. If `conf.sampling.device` is `"cuda:N"` → `("cuda:N",)`
3. Check `PARTICLEGPT_SAMPLE_NUM_GPUS` env var
4. Check `conf.sampling.force_single_gpu`
5. Default: use all available CUDA GPUs

The number of GPUs is also clamped to `total_starters` — no point spawning more workers than events to generate.

---

### `resolve_sample_idx(generated_samples_dir: Path) → int`

If `conf.sampling.sampling_idx_override` is set, returns that value (useful for overwriting a specific run). Otherwise, scans for existing `sampling_N` directories and returns the next integer index. This means re-running `sample.py` never overwrites previous results by default.

---

### `dtype_from_config(dtype_name: str) → torch.dtype`

Converts a config string (`"float32"`, `"bfloat16"`, `"float16"`) to `torch.dtype`. Raises `ValueError` for unknown strings.

---

### `autocast_context(device_type, dtype)`

Returns the appropriate `torch.autocast` context manager, or `nullcontext()` for float32 / CPU.

---

### `shard_path(output_dir: Path, rank: int) → Path`

Returns the temporary shard path for worker `rank`: `output_dir/worker_RRR.csv.part`.

---

## The Job Dict

`main()` builds a plain `dict` that is passed to every worker via `mp.spawn`. It must be fully picklable. Key fields:

| Key | Description |
|---|---|
| `checkpoint_path` | Path to `ckpt.pt` |
| `output_dir` | Where to write shard files |
| `device_names` | Tuple of device strings |
| `batch_size` | Per-worker batch size |
| `starter_tokens` | How many tokens to feed as prompt (currently hard-coded to 5) |
| `max_new_tokens` | How many new tokens to generate per event |
| `temperature` | Sampling temperature |
| `top_k` | Top-k threshold (or `None`) |
| `dtype_name` | Autocast dtype string |
| `seed` | Base seed (each worker offsets by `rank`) |
| `compile` | Whether to `torch.compile` on each worker |
| `total_starters` | Total number of test sequences to sample from |
| `tokenized_data_filepath` | Path to the `.bin` file |
| `tokenized_dtype` | NumPy dtype name of the binary |
| `sequence_length` | Tokens per event |
| `split_start_token_idx` | Byte offset for the test split |
| `max_test_sequences` | Recorded in metadata (may be `None`) |

---

## Output Format

`samples.csv` is a plain text file (despite the `.csv` extension, it uses space as delimiter, not commas):

```
3 12 87 245 301 88 176 4 ...
3 12 45 190 312 91 202 4 ...
```

- One row = one generated event
- Each value = an integer token ID
- Rows include the starter tokens at the beginning
- The file is produced by rank-order concatenation of worker shards (worker 0 first, then 1, …)

---

## Gotchas

- **`starter_tokens` is currently hard-coded to 5** inside `main()`. There is a `# @TODO` comment noting this should be configurable. If your events require a different number of starter tokens, you need to edit the script directly for now.

- **`max_new_tokens` defaults to `sequence_length - starter_tokens`**. If `conf.sampling.max_new_tokens` is not set, the model is asked to complete the rest of a full event. If you want shorter/longer generations, set it explicitly in your config.

- **The model is loaded independently on each worker**. For a large model on many GPUs, startup time can be significant. This is by design — shared memory model loading across processes is fragile.

- **`torch.compile` on each worker** is done after moving the model to device. On first run this triggers JIT compilation, which can take 30-60s per worker. Subsequent runs reuse the compiled cache.

- **Shard files are deleted after concatenation by default** (`keep_shards=False`). Set `conf.sampling.keep_shards=True` if you need to inspect per-GPU output (e.g. to diagnose worker-specific issues).

- **Multi-GPU sampling uses `mp.spawn`, not `torchrun`**. You do not need `torchrun` for sampling — just `python sample.py config.json`.

- **Checkpoint vocab size is validated against tokenized metadata**. If they differ, sampling aborts immediately. This can happen if you re-tokenize data with a different dictionary without re-training.

---

## Related

| Module | How it connects |
|---|---|
| `configurator.py` | Loads `conf.generic`, `conf.sampling` |
| `train.py` | Uses `clean_state_dict_keys()` from `train.py` to load state dict |
| `particleGPT/model.py` | `GPT` and `GPTConfig` used to reconstruct the model |
| `preparation.py` | `DataloaderSplitConfig`, `TokenizedMetadataConfig` to locate the test binary |
| `analysis/analyzer.py` | Reads the `samples.csv` written here |
