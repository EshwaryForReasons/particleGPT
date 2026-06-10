# `configurator.py`

## Overview

`configurator.py` is the configuration layer for all particleGPT scripts. It defines three `dataclass` containers and populates them from a JSON file passed as the first CLI argument.

Every script that needs config simply does `import configurator as conf` and then reads from `conf.generic`, `conf.training`, or `conf.sampling`. The module-level code runs automatically at import time, parsing `sys.argv[1]` if present.

Config is flat JSON — there is no YAML, no environment variable interpolation, and no inheritance. Simple to read, simple to debug.

---

## Quick Start

```bash
# 1. Create a JSON config:
cat > config/models/my_model.json << 'EOF'
{
  "model_name": "my_model",
  "dataset": "dataset.csv",
  "preparation_config_file": "config/preparations/my_prep.json",
  "training_config": {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "block_size": 512,
    "batch_size": 32,
    "learning_rate": 6e-4,
    "max_iters": 100000
  },
  "sampling_config": {
    "temperature": 0.8,
    "top_k": 200
  }
}
EOF

# 2. Run any script; config is loaded automatically:
python train.py config/models/my_model.json
```

---

## API Reference

### `perform_configuration(config_file_path: str) → Tuple[GenericConfiguration, TrainingConfiguration, SamplingConfiguration]`

Reads the JSON at `config_file_path`, creates the three config objects, and populates them. Keys under the root object go into `GenericConfiguration`. Keys under `"training_config"` go into `TrainingConfiguration`. Keys under `"sampling_config"` go into `SamplingConfiguration`.

Unknown keys trigger a `Warning` but do not raise an error — this is intentional to allow forward-compatible configs.

If `model_name` is not specified, it defaults to the config file's stem (e.g. `"config/models/my_exp.json"` → `model_name="my_exp"`).

**Parameters**

| Param | Type | Description |
|---|---|---|
| `config_file_path` | `str` | Path to the JSON config file |

**Returns**: `(generic, training, sampling)` tuple

---

### `class GenericConfiguration`

Top-level settings shared across training and sampling.

| Field | Type | Default | Description |
|---|---|---|---|
| `config_file_path` | `str` | `''` | Set automatically from CLI arg |
| `preparation_name` | `str` | `''` | Legacy; prefer `preparation_config_file` |
| `model_name` | `str` | `''` | Used for checkpoint and output dirs |
| `dataset` | `str` | `''` | Raw dataset filename (relative to `data/raw/`) |
| `data_mode` | `str` | `'particle'` | `'particle'` or `'generic'` |
| `mlp_type` | `str` | `'swiglu'` | MLP variant: `'swiglu'`, `'relu2'`, `'gelu'` |
| `mlp_ratio` | `float` | `4.0` | Hidden-to-embedding size ratio in MLP |
| `qk_norm` | `bool` | `True` | Apply QK normalisation in attention |
| `embd_norm_type` | `str` | `'none'` | Embedding norm: `'none'` or `'rmsnorm'` |
| `embd_norm_init_scale` | `float` | `0.02` | Init scale for embedding norm |
| `use_particle_index_embd` | `bool` | `False` | Add per-particle-position embeddings |
| `use_bin_value_embd` | `bool` | `False` | Add continuous bin-value auxiliary embeddings |
| `bin_value_embd_init_scale` | `float` | `0.0` | Init scale for bin value embeddings |
| `preparation_config_file` | `str \| None` | `None` | **Required.** Path to the preparation JSON |

---

### `class TrainingConfiguration`

All training hyperparameters.

**I/O**

| Field | Type | Default | Description |
|---|---|---|---|
| `eval_interval` | `int` | `2000` | Evaluate every N iterations; set negative to disable |
| `log_interval` | `int` | `10` | Log every N iterations |
| `eval_iters` | `int` | `200` | Number of batches per loss estimate |
| `eval_only` | `bool` | `False` | Exit after first evaluation |
| `init_from` | `str` | `''` | `'scratch'` or `'resume'` |
| `eval_every_epoch` | `bool` | `True` | Force eval at epoch boundaries |

**Data**

| Field | Type | Default | Description |
|---|---|---|---|
| `gradient_accumulation_steps` | `int` | `40` | Micro-steps before an optimizer step |
| `batch_size` | `int` | `12` | Micro-batch size per GPU |
| `block_size` | `int` | `-1` | Context length in tokens. Set `-1` to compute dynamically |
| `context_sequences` | `int` | `-1` | Sequences per block (used if `block_size=-1`) |
| `use_self_contained_blocks` | `bool` | `False` | Force blocks to not cross event boundaries |

**Model architecture**

| Field | Type | Default | Description |
|---|---|---|---|
| `n_layer` | `int` | `12` | Number of Transformer blocks |
| `n_head` | `int` | `12` | Number of attention heads |
| `n_kv_heads` | `int` | `0` | Key/value heads (0 = standard MHA; < n_head = GQA) |
| `n_embd` | `int` | `768` | Embedding dimension |
| `dropout` | `float` | `0.0` | Dropout probability |
| `bias` | `bool` | `False` | Include bias in linear layers |

**Optimizer**

| Field | Type | Default | Description |
|---|---|---|---|
| `learning_rate` | `float` | `6e-4` | Peak learning rate |
| `max_iters` | `int` | `600000` | Maximum training iterations |
| `max_epochs` | `int` | `10e9` | Maximum training epochs |
| `max_num_failed_checkpoint_checks` | `int` | `4` | Early stop after N consecutive non-improvements |
| `weight_decay` | `float` | `0.1` | AdamW weight decay |
| `beta1` | `float` | `0.9` | AdamW β₁ |
| `beta2` | `float` | `0.95` | AdamW β₂ |
| `grad_clip` | `float` | `1.0` | Gradient norm clip |

**LR schedule**

| Field | Type | Default | Description |
|---|---|---|---|
| `lr_scheduler` | `str` | `"cosine_annealing_with_warmup"` | Scheduler name |
| `warmup_iters` | `int` | `2000` | Linear warmup duration |
| `lr_decay_iters` | `int` | `600000` | Point at which LR reaches `min_lr` |
| `min_lr` | `float` | `6e-5` | Minimum learning rate |
| `cycle_steps_mult` | `float` | `1.0` | Cycle length multiplier (warm restarts only) |
| `base_lr_decay_mult` | `float` | `1.0` | Per-cycle LR decay (warm restarts only) |

**System**

| Field | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `'cuda'` | Target device |
| `dtype` | `str` | auto | `'bfloat16'` if supported, else `'float16'` |
| `compile` | `bool` | `True` | Use `torch.compile` |
| `seed` | `int` | `1337` | Global random seed |
| `backend` | `str` | `'nccl'` | DDP backend |

---

### `class SamplingConfiguration`

Sampling-specific settings.

| Field | Type | Default | Description |
|---|---|---|---|
| `batch_size` | `int` | `128` | Per-worker batch size for generation |
| `max_new_tokens` | `int \| None` | `500` | New tokens per event (`None` = auto: `seq_len - starters`) |
| `temperature` | `float` | `0.8` | Softmax temperature |
| `top_k` | `int` | `200` | Top-k filtering (`None` or `≤ 0` = disabled) |
| `seed` | `int` | `1337` | Base seed (workers offset by rank) |
| `device` | `str` | `'cuda'` | Device string |
| `dtype` | `str` | auto | Autocast dtype |
| `compile` | `bool` | `True` | `torch.compile` on sampling workers |
| `max_test_sequences` | `int \| None` | `None` | Cap on test sequences to sample |
| `keep_shards` | `bool` | `False` | Keep per-worker CSV shards |
| `sampling_idx_override` | `int \| None` | `None` | Force a specific `sampling_N` index |
| `force_single_gpu` | `bool` | `False` | Use only one GPU even if more are available |
| `stop_at_event_end` | `bool` | `True` | Stop generation at `event_end` token |
| `stop_at_padding` | `bool` | `True` | Stop generation at `padding` token |
| `float_precision` | `float` | `5` | Decimal places for untokenized output |

---

## Gotchas

- **Config is loaded at import time**. As soon as any script does `import configurator as conf`, `sys.argv[1]` is parsed. If you're writing a test or utility that imports from a script that uses configurator, make sure the config file path is in argv or the import will print a warning.

- **Unknown keys silently warn, not raise**. If you typo a key (e.g. `"learing_rate"` instead of `"learning_rate"`), you'll see a warning but the default value will be used silently. Always check the warning output when first running with a new config.

- **`dtype` is auto-detected, not settable in `TrainingConfiguration`**. The field is set in `__post_init__` based on hardware. You can override it in `SamplingConfiguration` because sampling dtype is often a user choice (e.g. forcing float32 for reproducibility).

- **`block_size=-1` is a special sentinel**. The training script computes it from `context_sequences * sequence_length` when it sees `-1`. Setting both `block_size` and `context_sequences` is redundant; `block_size` takes precedence if both are positive.

- **`init_from=''`** (empty string, the default) causes the script to print a warning and may behave unexpectedly. Always set it to `"scratch"` or `"resume"` explicitly.

- **Module-level globals `generic`, `training`, `sampling`** are `None` unless `perform_configuration` has been called (i.e. unless a config file was passed). Code that imports configurator without a config file (e.g. unit tests) must handle `None` globals.

---

## Related

| Module | How it connects |
|---|---|
| `train.py` | Reads `conf.generic`, `conf.training` |
| `sample.py` | Reads `conf.generic`, `conf.sampling` |
| `analysis/analyzer.py` | Reads `conf.generic` |
| `dictionary.py` | Uses `conf.generic.dataset` and related fields |
