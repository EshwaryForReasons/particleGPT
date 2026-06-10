# `particleGPT/model.py`

## Overview

`model.py` implements the GPT model used by particleGPT. It is a decoder-only Transformer derived from [nanoGPT](https://github.com/karpathy/nanoGPT) with several particle-physics-specific extensions:

- **Multiple MLP variants**: SwiGLU, ReLU², GELU (controlled by `mlp_type`)
- **QK normalisation** for improved training stability at scale
- **Grouped-query attention (GQA)** for memory-efficient inference
- **Particle index embeddings**: a separate positional embedding for which particle-slot within an event the current token belongs to
- **Bin-value embeddings**: a continuous auxiliary embedding that injects the "fractional position within a bin" as a learned correction signal
- **Flexible embedding normalisation** (none or RMSNorm)

The model takes a sequence of integer token IDs as input and outputs logits over the vocabulary. It is a standard autoregressive language model; the physics specialisation lives entirely in the embedding and MLP layers.

---

## Quick Start

```python
from particleGPT.model import GPT, GPTConfig

# Build a small model from scratch
config = GPTConfig(
    vocab_size=512,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    mlp_type="swiglu",
    qk_norm=True,
)
model = GPT(config)

# Forward pass
import torch
x = torch.randint(0, 512, (4, 256))   # (batch, seq_len)
logits, loss, _ = model(x, targets=x)  # loss is cross-entropy

# Generate
starters = torch.randint(0, 512, (4, 5))  # (batch, 5 starter tokens)
generated = model.generate(starters, max_new_tokens=50, temperature=0.8, top_k=200)
# generated.shape == (4, 55)
```

---

## `GPTConfig`

```python
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_heads: int = 0          # 0 = standard MHA; < n_head = GQA
    dropout: float = 0.0
    bias: bool = False
    data_mode: str = "particle"
    mlp_type: str = "swiglu"
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    embedding_norm_type: str = "none"
    embedding_norm_init_scale: float = 0.02
    use_particle_index_embeddings: bool = False
    num_features_per_particle: int = 6
    max_particles_per_event: int = 50
    use_bin_value_embeddings: bool = False
    bin_embedding_init_scale: float = 0.0
```

All fields correspond directly to config JSON keys (see `configurator.py`). `vocab_size` and `block_size` are always required.

---

## `GPT`

### `__init__(self, config: GPTConfig)`

Builds the full model. Layers created:

- `transformer.wte`: token embedding `(vocab_size, n_embd)`
- `transformer.wpe`: positional embedding `(block_size, n_embd)`
- `transformer.drop`: input dropout
- `transformer.h`: a `ModuleList` of `Block` instances (length = `n_layer`)
- `transformer.ln_f`: final layer norm
- `lm_head`: output projection `(n_embd, vocab_size)`, **weight-tied to `wte`**

Optional extensions (when enabled in config):
- `transformer.particle_idx_embd`: particle-position embedding `(max_particles_per_event, n_embd)`
- `transformer.bin_value_embd`: bin-value auxiliary embedding `(vocab_size, n_embd)`
- `transformer.embd_norm`: RMSNorm applied after the sum of all embedding contributions

---

### `forward(idx, targets=None) → Tuple[Tensor, Optional[Tensor], Optional[Tensor]]`

**Parameters**

| Param | Type | Description |
|---|---|---|
| `idx` | `Tensor` | Shape `(B, T)` — integer token IDs |
| `targets` | `Tensor \| None` | Shape `(B, T)` — target token IDs for loss computation |

**Returns**: `(logits, loss, aux_info)`

- `logits`: shape `(B, T, vocab_size)` — raw unnormalised scores
- `loss`: scalar cross-entropy loss (only if `targets` is provided, else `None`)
- `aux_info`: optional dict with auxiliary information (e.g. attention weights)

The forward pass:
1. Adds token + positional embeddings
2. Optionally adds particle-index and bin-value embeddings
3. Optionally applies embedding norm
4. Runs through all Transformer blocks
5. Applies final layer norm
6. Projects to logits
7. If `targets`: computes cross-entropy, ignoring positions where `targets == 0` (padding)

---

### `generate(idx, max_new_tokens, temperature=1.0, top_k=None) → Tensor`

Standard autoregressive generation. Works on a 2-D input tensor `(B, T)`.

At each step:
1. Crops `idx` to `block_size` if longer (leftmost tokens are discarded)
2. Forward pass → logits at the last position
3. Scale logits by `1/temperature`
4. Optionally apply top-k masking
5. Sample from the resulting distribution
6. Append the new token

**Parameters**

| Param | Type | Description |
|---|---|---|
| `idx` | `Tensor` | `(B, T)` starter tokens |
| `max_new_tokens` | `int` | How many tokens to generate |
| `temperature` | `float` | Sampling temperature |
| `top_k` | `int \| None` | Restrict to top-k logits before sampling |

**Returns**: `Tensor` of shape `(B, T + max_new_tokens)`

---

### `configure_optimizers(weight_decay, learning_rate, betas, device_type) → torch.optim.Optimizer`

Creates an AdamW optimiser with **selective weight decay** — weight decay is applied to non-bias, non-LayerNorm parameters only (standard practice for Transformers).

**Parameters**

| Param | Type | Description |
|---|---|---|
| `weight_decay` | `float` | L2 penalty for eligible parameters |
| `learning_rate` | `float` | Initial learning rate |
| `betas` | `Tuple[float, float]` | AdamW β₁, β₂ |
| `device_type` | `str` | `"cuda"` or `"cpu"` (enables fused AdamW on CUDA) |

---

### `get_num_params(non_embedding=True) → int`

Returns the number of trainable parameters, optionally excluding embedding parameters.

---

### `estimate_mfu(fwdbwd_per_iter, dt) → float`

Estimates Model FLOPs Utilisation — a useful throughput metric. Returns a fraction (0–1) of the theoretical peak FLOPS of the hardware being used.

---

## Sub-modules

### `Block`

A single Transformer layer:
1. Pre-norm → `CausalSelfAttention`
2. Residual add
3. Pre-norm → MLP (variant selected by `mlp_type`)
4. Residual add

### `CausalSelfAttention`

Standard multi-head causal self-attention with:
- Optional GQA (`n_kv_heads < n_head`)
- Optional QK normalisation (`qk_norm=True`)
- Flash attention via `F.scaled_dot_product_attention`
- Causal mask is handled by `is_causal=True` in `sdpa`

### MLP variants

| `mlp_type` | Class | Notes |
|---|---|---|
| `"swiglu"` | `SwiGLUMLP` | Gate × activation pattern; `gate_proj * silu(up_proj)` |
| `"relu2"` | `ReLU2MLP` | Squared ReLU activation |
| `"gelu"` | `GELUMLP` | Standard GELU; nanoGPT default |

All variants follow the pre-norm + two-linear-layer pattern but differ in activation and projection structure.

### `LayerNorm` / `RMSNorm`

Implementations of LayerNorm (with optional bias) and RMSNorm. Used throughout the model. RMSNorm is used for `qk_norm` and optionally for `embedding_norm_type`.

---

## Gotchas

- **Weight tying**: `lm_head.weight` shares memory with `wte.weight`. Do not replace or reinitialise `lm_head` without also updating `wte`, or training will silently diverge.

- **Padding targets**: The forward pass ignores positions where `targets == 0` when computing loss. Token `0` is therefore reserved as the padding token and **must** not be used as a real vocabulary token.

- **GQA requires `n_kv_heads` to divide `n_head`**. Setting `n_kv_heads=4` with `n_head=12` means 3 query heads per key/value head. Setting `n_kv_heads=0` reverts to standard MHA (`n_kv_heads = n_head`).

- **`block_size` is hard-coded at init**: the positional embedding table has shape `(block_size, n_embd)`. You cannot process sequences longer than `block_size` without cropping. `generate()` handles this by left-cropping automatically.

- **Particle index embeddings require correct `num_features_per_particle` and `max_particles_per_event`**. These define how token position maps to particle slot. Misconfiguring these produces wrong embeddings without any error.

- **`torch.compile` compatibility**: the model is written to be `torch.compile`-friendly (`fullgraph=False` in `train.py`). Some attention patterns or dynamic shapes may cause graph breaks; profile with `torch._dynamo.explain()` if you see unexpectedly slow compilation.

---

## Related

| Module | How it connects |
|---|---|
| `train.py` | Instantiates `GPT(GPTConfig(**model_args))` |
| `sample.py` | Loads model from checkpoint and calls `model.generate()` |
| `configurator.py` | Provides all architecture hyperparameters via config JSON |
