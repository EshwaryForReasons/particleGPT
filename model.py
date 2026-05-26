"""
GPT-style language model for particle event token sequences.

This module contains the model architecture, loss functions, and generation helpers used
by the training and sampling scripts. The model is intentionally domain-aware: token
embeddings are augmented with token-type embeddings, optional ordinal/bin embeddings,
and optional particle-index embeddings.

Main design choices:
    * Pre-norm Transformer blocks with RMSNorm.
    * RoPE positional information instead of learned absolute position embeddings.
    * Optional grouped-query attention through torch SDPA's enable_gqa path.
    * Optional Q/K RMSNorm over each attention head's head_dim.
    * Optional ReLU^2, GELU, or SwiGLU MLPs.
    * Cached generation with absolute RoPE positions.
"""

from __future__ import annotations

import inspect
import json
import math
import multiprocessing as mp
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

import configurator as conf
import pLogging

# Won't always have a dictionary if training on generic data
try:
    from dictionary import Dictionary, ETokenTypes
except Exception as exc:  # Allows generic/non-particle training without dictionary.py.
    Dictionary = None
    ETokenTypes = None
    _dictionary_import_error = exc
else:
    _dictionary_import_error = None
    
    
def _conf_data_mode() -> str:
    """Import-time default: particle keeps old behavior; generic skips dictionary loading."""
    return str(getattr(conf.generic, "data_mode", "particle")).lower()

def _default_num_token_types() -> int:
    """Default token-type table size when particle token types exist."""
    return len(ETokenTypes) if ETokenTypes is not None else 1

def _require_particle_dictionary(feature_name: str):
    """Return the particle dictionary or explain which feature requested it."""
    if dictionary is None or ETokenTypes is None:
        raise RuntimeError(
            f"{feature_name} requires the particle Dictionary/ETokenTypes. "
            "For natural-language/generic training, set data_mode='generic' and leave "
            "use_token_type_embeddings=False, use_bin_value_embeddings=False, "
            "use_particle_index_embeddings=False, and use_event_grammar=False."
        )
    return dictionary
    
    
script_dir = Path(__file__).resolve().parent

# The model is normally constructed from a training or generation script. Logging is
# therefore delegated to the caller by setting this process-local logger index.
logger_idx = -1

def set_logger(in_logger_idx: int) -> None:
    """Set the pLogging logger index used by this module."""
    global logger_idx
    logger_idx = in_logger_idx

# =====================
# Dictionary loading
# =====================

dictionary = None
if _conf_data_mode() == "particle":
    if Dictionary is None:
        raise RuntimeError("Could not import particle dictionary module.") from _dictionary_import_error

    try:
        prep_filepath = script_dir / "preparations" / conf.generic.preparation_name / "preparation.json"
        with open(prep_filepath, "r") as f:
            prep_data = json.load(f)

        tokenized_dataset_name = prep_data["train_bin"]["tokenized_dataset"]
        dictionary_filepath = script_dir / "data" / "tokenized" / tokenized_dataset_name / "dictionary.json"
        dictionary = Dictionary(dictionary_filepath)
    except Exception as exc:
        pLogging.info(logger_idx, f"Error occurred while trying to load dictionary: {exc}")
        raise RuntimeError("Error occurred while trying to load dictionary.") from exc

# =====================
# Normalization layers
# =====================

class RMSNorm(nn.Module):
    """
    Root-mean-square normalization over the final tensor dimension.

    For an input `x` this applies
        `x * rsqrt(mean(x**2) + eps) * weight`

    The RMS statistic is computed in fp32 for bf16/fp16 stability, then cast back to the
    original input dtype. This is the block-level normalization used in the Transformer.
    """

    def __init__(self, ndim: int, eps: float = 1e-5, init_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((ndim,), float(init_scale)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        x_normed = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight.float() * x_normed).to(input_dtype)

# Maybe rename to QKRMSNorm; this is a `head_dim` RMSNorm shared across heads.
class HeadRMSNorm(nn.Module):
    """
    RMSNorm over the per-head dimension used for Q/K normalization.

    Expected input shape:
        `(batch, n_heads, seq_len, head_dim)`

    The learned scale is shared across batch, heads, and time, but not across the
    `head_dim` coordinates.
    """

    def __init__(self, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        x_normed = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight.float() * x_normed).to(input_dtype)

# =====================
# Attention
# =====================

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with optional grouped-query attention, QK RMSNorm, RoPE, and
    KV-cache support.

    Tensor layout inside attention is always:
        Q: `(B, n_head,    T, head_dim)`
        K: `(B, n_kv_head, T, head_dim)`
        V: `(B, n_kv_head, T, head_dim)`

    When `n_kv_heads < n_head`, torch SDPA handles the GQA expansion internally through
    `enable_gqa=True`.
    """

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.n_kv_heads = int(config.n_kv_heads or config.n_head)
        assert self.n_head % self.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"

        self.n_head_groups = self.n_head // self.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "RoPE requires an even head_dim"

        self.n_embd = config.n_embd
        self.dropout = config.dropout

        qkv_dim = (self.n_head + 2 * self.n_kv_heads) * self.head_dim
        self.c_attn = nn.Linear(config.n_embd, qkv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        if getattr(config, "qk_norm", True):
            self.q_norm = HeadRMSNorm(self.head_dim)
            self.k_norm = HeadRMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # RoPE cache. This uses the GPT-NeoX style rotate-half convention: frequencies
        # are duplicated across both halves of head_dim, and rotate_half swaps/sign-flips
        # those two halves.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        positions = torch.arange(config.block_size, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)  # (block_size, head_dim // 2)
        rope = torch.cat((freqs, freqs), dim=-1)  # (block_size, head_dim)
        self.register_buffer("cos_cached", rope.cos().unsqueeze(0).unsqueeze(0), persistent=True)
        self.register_buffer("sin_cached", rope.sin().unsqueeze(0).unsqueeze(0), persistent=True)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the final dimension by splitting into two halves: [x1, x2] -> [-x2, x1]."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Should be a faster RoPE since this avoids .item() calls and uses the cached cos/sin values directly.
    # Unlike the previous one, this one should faster since it allows for the fast path for normal
    # training/full-sequence eval.
    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE. Two paths, fast for training, general for inference.

        Fast path:
            position_ids is None
            -> normal full-sequence training/eval/prefill with positions 0..T-1.

        General path:
            position_ids is not None
            -> cached decode or unusual absolute positions.
        """
        B, _, T, D = q.shape
        assert D == self.head_dim

        # ===== FAST PATH: normal training / full-sequence eval / prefill =====
        if position_ids is None:
            cos = self.cos_cached[:, :, :T, :]  # (1, 1, T, head_dim)
            sin = self.sin_cached[:, :, :T, :]

            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)
            return q, k

        # ===== GENERAL PATH: cached decode / explicit absolute positions =====
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        position_ids = position_ids.to(device=q.device, dtype=torch.long)

        if position_ids.size(0) == 1 and B > 1:
            position_ids = position_ids.expand(B, -1)

        if position_ids.shape != (B, T):
            raise ValueError(
                f"position_ids must have shape {(B, T)}, (1, {T}), or ({T},); "
                f"got {tuple(position_ids.shape)}"
            )

        # Optional safety check, but avoid .item() in the hot path because it causes a CPU-GPU sync.
        # Keeping this here because it is nice to enable temporarily while debugging.
        # if torch.any(position_ids < 0) or torch.any(position_ids >= self.cos_cached.size(2)):
        #     raise ValueError("RoPE position_ids out of range")

        flat_pos = position_ids.reshape(-1)

        cos_base = self.cos_cached[0, 0]  # (block_size, head_dim)
        sin_base = self.sin_cached[0, 0]

        cos = cos_base.index_select(0, flat_pos).view(B, T, self.head_dim).unsqueeze(1)
        sin = sin_base.index_select(0, flat_pos).view(B, T, self.head_dim).unsqueeze(1)

        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Run attention for either a full sequence/prefill or a cached decode step.

        `attn_mask` is passed directly to `torch.nn.functional.scaled_dot_product_attention`.
        If an explicit mask is provided for full-sequence training it must already include
        any causal structure we need. With `attn_mask=None`, this method automatically
        uses causal masking for full-sequence/prefill calls and non-causal attention for
        cached decode calls, because the cache contains only past tokens.
        """
        B, T, C = x.shape
        if C != self.n_embd:
            raise ValueError(f"Expected embedding dim {self.n_embd}, got {C}")

        qkv = self.c_attn(x)
        q_size = self.n_head * self.head_dim
        k_size = self.n_kv_heads * self.head_dim
        v_size = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, k_size, v_size], dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self._apply_rope(q, k, position_ids)

        had_kv_cache = kv_cache is not None
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            expected = (B, self.n_kv_heads, self.head_dim)
            got_k = (prev_k.size(0), prev_k.size(1), prev_k.size(3))
            got_v = (prev_v.size(0), prev_v.size(1), prev_v.size(3))
            if got_k != expected or got_v != expected:
                raise ValueError(
                    "KV cache shape mismatch. Expected batch/n_kv_heads/head_dim "
                    f"{expected}, got K {got_k} and V {got_v}."
                )
            k = torch.cat((prev_k, k), dim=2)
            v = torch.cat((prev_v, v), dim=2)

        new_kv_cache = (k, v) if use_cache else None
        is_causal = (attn_mask is None) and (not had_kv_cache)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=(self.n_head_groups > 1),
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv_cache

# =====================
# MLPs
# =====================

class GELUMLP(nn.Module):
    """GPT-2 style MLP using GELU activation and a configurable expansion ratio."""

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        hidden_dim = int(config.mlp_ratio * config.n_embd)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)

class RELU2MLP(nn.Module):
    """
    MLP using the ReLU^2 activation.

    Structure:
        `x -> Linear(n_embd, mlp_ratio*n_embd) -> relu(x)^2 -> Linear(...) -> dropout`
    """

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        hidden_dim = int(config.mlp_ratio * config.n_embd)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return self.dropout(x)

class SwiGLUMLP(nn.Module):
    """
    SwiGLU MLP with a tensor-core-friendly hidden size.

    The `8/3` factor keeps parameter count close to the standard `4x` MLP after
    accounting for the two input projections used by SwiGLU.
    """

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        return self.dropout(x)

def choose_embedding_norm(config: "GPTConfig") -> nn.Module:
    """Return the optional normalization applied after token/type/bin embeddings."""
    norm_type = str(getattr(config, "embedding_norm_type", "none")).lower()

    if norm_type == "none":
        return nn.Identity()
    if norm_type == "rmsnorm":
        return RMSNorm(config.n_embd, init_scale=getattr(config, "embedding_norm_init_scale", 1.0))
    raise ValueError(f"Unknown embedding_norm_type: {norm_type}")

def choose_mlp(config: "GPTConfig") -> nn.Module:
    """Construct the configured feed-forward sublayer used inside each Transformer block."""
    mlp_type = str(getattr(config, "mlp_type", "relu2")).lower()

    if mlp_type == "gelu":
        return GELUMLP(config)
    if mlp_type == "relu2":
        return RELU2MLP(config)
    if mlp_type == "swiglu":
        return SwiGLUMLP(config)
    raise ValueError(f"Unknown mlp_type: {mlp_type}")

class Block(nn.Module):
    """Pre-norm Transformer block: RMSNorm -> attention -> residual -> RMSNorm -> MLP -> residual."""

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = choose_mlp(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv_cache = self.attn(
            self.ln_1(x),
            kv_cache=kv_cache,
            attn_mask=attn_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache

# =====================
# Configuration and multiprocessing worker
# =====================

@dataclass
class GPTConfig:
    """Configuration for the particleGPT model."""

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_heads: int = 0  # 0/None means standard MHA with n_kv_heads == n_head.
    dropout: float = 0.0
    bias: bool = True
    
    # Domain controls. "particle" keeps the original particleGPT behavior.
    # "generic" disables all particle/event grammar features for ordinary LM data.
    data_mode: str = "particle"  # "particle" or "generic".
    use_token_type_embeddings: bool = True
    num_token_types: int = field(default_factory=_default_num_token_types)
    use_event_grammar: bool = True

    # Generic generation/loss controls. For particle mode, None means use dictionary values.
    padding_token_id: int = 0
    eos_token_id: Optional[int] = None
    loss_ignore_index: Optional[int] = None

    mlp_type: str = "swiglu"  # "gelu", "relu2", or "swiglu".
    mlp_ratio: float = 4.0

    qk_norm: bool = True

    embedding_norm_type: str = "none"  # "none" or "rmsnorm".
    embedding_norm_init_scale: float = 0.02

    use_particle_index_embeddings: bool = False
    # @TODO: if we plant to keep these, make sure to implement auto-detection of these values.
    # @IMPORTANT: see above todo
    num_features_per_particle: int = 4
    max_particles_per_event: int = 42

    use_bin_value_embeddings: bool = False
    bin_embedding_init_scale: float = 0.0

    def __post_init__(self) -> None:
        self.n_kv_heads = int(self.n_kv_heads or self.n_head)
        self.data_mode = self.data_mode.lower()
        self.mlp_type = self.mlp_type.lower()
        self.embedding_norm_type = self.embedding_norm_type.lower()
        
        if self.data_mode not in {"particle", "generic"}:
            raise ValueError(f"Unknown data_mode: {self.data_mode}")

        # Natural-language/generic training should not accidentally inherit particle grammar.
        if self.data_mode == "generic":
            self.use_token_type_embeddings = False
            self.use_particle_index_embeddings = False
            self.use_bin_value_embeddings = False
            self.use_event_grammar = False

        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}")
        if self.n_head % self.n_kv_heads != 0:
            raise ValueError(f"n_head={self.n_head} must be divisible by n_kv_heads={self.n_kv_heads}")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("RoPE requires an even head_dim = n_embd // n_head")
        if self.mlp_type not in {"gelu", "relu2", "swiglu"}:
            raise ValueError(f"Unknown mlp_type: {self.mlp_type}")
        if self.embedding_norm_type not in {"none", "rmsnorm"}:
            raise ValueError(f"Unknown embedding_norm_type: {self.embedding_norm_type}")
        if self.use_particle_index_embeddings and self.num_features_per_particle <= 0:
            raise ValueError("num_features_per_particle must be positive")
        if self.use_particle_index_embeddings and self.max_particles_per_event <= 0:
            raise ValueError("max_particles_per_event must be positive")
        if self.use_token_type_embeddings and self.num_token_types <= 0:
            raise ValueError("num_token_types must be positive when token type embeddings are enabled")

def batched_multiGPU_worker(
    split_index: int,
    device_id: int,
    model_config: GPTConfig,
    model_state_dict: dict[str, torch.Tensor],
    starters_chunk: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    batch_size: int,
    grammar_mask: bool,
    return_queue,
) -> None:
    """Worker process used by `GPT.generate_batched_multiGPU`."""
    try:
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")

        model = GPT(model_config)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()

        starters_chunk = starters_chunk.to(device, non_blocking=True)
        generated = model.generate_batched_singleGPU(
            starters_chunk,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            batch_size=batch_size,
            grammar_mask=grammar_mask,
        )
        return_queue.put((split_index, generated.cpu(), None))
    except Exception as exc:  # propagate enough information to avoid parent deadlocks/hangs
        return_queue.put((split_index, None, repr(exc)))
        raise

# =====================
# GPT model
# =====================

class GPT(nn.Module):
    """Particle-aware GPT language model."""
    
    ORDINAL_TOKEN_TYPES = (
        set()
        if ETokenTypes is None
        else {
            ETokenTypes.ENERGY.value,
            ETokenTypes.ETA.value,
            ETokenTypes.THETA.value,
            ETokenTypes.PHI.value,
            ETokenTypes.PT.value,
            ETokenTypes.PX.value,
            ETokenTypes.PY.value,
            ETokenTypes.PZ.value,
        }
    )

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.vocab_size is None or config.block_size is None:
            raise ValueError("config.vocab_size and config.block_size must be set")

        self.config = config
        
        if config.data_mode == "particle":
            _require_particle_dictionary("particle data_mode")

        modules: dict[str, nn.Module] = {
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            # "type_emb": nn.Embedding(config.num_token_types, config.n_embd),
            "emb_norm": choose_embedding_norm(config),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": RMSNorm(config.n_embd),
        }
        
        if config.use_token_type_embeddings:
            modules["type_emb"] = nn.Embedding(config.num_token_types, config.n_embd)
        
        if config.use_token_type_embeddings:
            modules["type_emb"] = nn.Embedding(config.num_token_types, config.n_embd)

        if config.use_bin_value_embeddings:
            modules["bin_value_mlp"] = nn.Sequential(
                nn.Linear(3, config.n_embd, bias=config.bias),
                nn.SiLU(),
                nn.Linear(config.n_embd, config.n_embd, bias=config.bias),
            )

        if config.use_particle_index_embeddings:
            modules["particle_index_emb"] = nn.Embedding(
                config.max_particles_per_event + 1,
                config.n_embd,
            )

        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.use_bin_value_embeddings:
            bin_feature_lut, bin_feature_mask = self._build_bin_feature_lut(config.vocab_size)
            self.register_buffer("bin_feature_lut", bin_feature_lut, persistent=False)
            self.register_buffer("bin_feature_mask", bin_feature_mask, persistent=False)
            self.bin_emb_scale = nn.Parameter(torch.tensor(float(config.bin_embedding_init_scale)))

        self.apply(self._init_weights)
        self.transformer.wte.weight = self.lm_head.weight  # tied token/input-output embedding

        for param_name, param in self.named_parameters():
            if param_name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        if config.data_mode == "particle":
            self._precompute_token_type_ranges()
        else:
            self.token_type_ranges = {}

        pLogging.info(logger_idx, "Model info", {"num_params": f"{self.get_num_params() / 1e6:.3f}M"})
        pLogging.info(
            logger_idx,
            "Model info",
            {"num_params_embedding_incl": f"{self.get_num_params(non_embedding=False) / 1e6:.3f}M"},
        )

    # ----------------------------------------------------------------------------------
    # Token type helpers
    # ----------------------------------------------------------------------------------

    @staticmethod
    def _token_type_specs() -> tuple[tuple[ETokenTypes, int, int], ...]:
        """
        Return contiguous token ranges as `(type, start, length)`.

        Padding is intentionally excluded. Special tokens start at
        `SPECIAL_TOKENS_OFFSET + 1` so that token 0/padding is not accidentally treated
        as a normal special token.
        """
        d = _require_particle_dictionary("token type ranges")
        return (
            (
                ETokenTypes.SPECIAL,
                d.SPECIAL_TOKENS_OFFSET + 1,
                max(d.num_special_tokens - 1, 0),
            ),
            (ETokenTypes.PDGID, d.PDGID_OFFSET, d.num_particles),
            (ETokenTypes.ENERGY, d.ENERGY_OFFSET, len(d.e_bins)),
            (ETokenTypes.ETA, d.ETA_OFFSET, len(d.eta_bins)),
            (ETokenTypes.THETA, d.THETA_OFFSET, len(d.theta_bins)),
            (ETokenTypes.PHI, d.PHI_OFFSET, len(d.phi_bins)),
            (ETokenTypes.PT, d.PT_OFFSET, len(d.pt_bins)),
            (ETokenTypes.PX, d.PX_OFFSET, len(d.px_bins)),
            (ETokenTypes.PY, d.PY_OFFSET, len(d.py_bins)),
            (ETokenTypes.PZ, d.PZ_OFFSET, len(d.pz_bins)),
        )

    def _precompute_token_type_ranges(self) -> None:
        """Build `token_type_value -> (start, end_inclusive)` for non-empty token ranges."""
        self.token_type_ranges: dict[int, tuple[int, int]] = {}
        for token_type, start, length in self._token_type_specs():
            if length > 0:
                self.token_type_ranges[token_type.value] = (start, start + length - 1)

    def get_token_type_ids(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Map token ids to `ETokenTypes` integer ids.

        Unknown/out-of-range ids remain marked as padding. This keeps masking safe in
        generation, but this issue really should never happen.
        """
        if self.config.data_mode != "particle":
            return torch.zeros_like(idx)

        d = _require_particle_dictionary("token type ids")
        type_ids = torch.full_like(idx, fill_value=ETokenTypes.PADDING.value)
        type_ids[idx == d.padding_token] = ETokenTypes.PADDING.value

        for token_type, start, length in self._token_type_specs():
            if length <= 0:
                continue
            type_ids[(idx >= start) & (idx < start + length)] = token_type.value

        return type_ids

    # ----------------------------------------------------------------------------------
    # Optional ordinal/bin embeddings
    # ----------------------------------------------------------------------------------

    @staticmethod
    def _centers_from_bins(bins, expected_len: int) -> torch.Tensor:
        """
        Convert either bin centers or bin edges into centers.

        Args:
            bins: Iterable of bin centers with length `expected_len` or bin edges with
                length `expected_len + 1`.
            expected_len: Number of token ids in the corresponding vocabulary range.
        """
        values = torch.as_tensor(bins, dtype=torch.float32).flatten()

        if expected_len == 0 or values.numel() == 0:
            return torch.empty(0, dtype=torch.float32)
        if values.numel() == expected_len:
            return values
        if values.numel() == expected_len + 1:
            return 0.5 * (values[:-1] + values[1:])

        raise ValueError(
            f"Cannot infer bin centers: got {values.numel()} values, expected "
            f"{expected_len} centers or {expected_len + 1} edges."
        )

    @staticmethod
    def _normalize_to_minus1_plus1(values: torch.Tensor) -> torch.Tensor:
        """Normalize scalar bin centers to [-1, 1], safely handling empty/constant inputs."""
        if values.numel() == 0:
            return values

        vmin = values.min()
        vmax = values.max()
        if torch.isclose(vmin, vmax):
            return torch.zeros_like(values)
        return 2.0 * (values - vmin) / (vmax - vmin) - 1.0

    def _build_bin_feature_lut(self, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build lookup tables used by optional ordinal/bin embeddings.

        `features[token_id] = [normalized_scalar, sin(phi), cos(phi)]`.
        For non-continuous tokens the feature row and mask are zero.
        """
        d = _require_particle_dictionary("bin value embeddings")
        features = torch.zeros(vocab_size, 3, dtype=torch.float32)
        mask = torch.zeros(vocab_size, 1, dtype=torch.float32)

        def valid_range(offset: int, length: int) -> bool:
            if length is None or length <= 0 or offset is None or offset < 0:
                return False
            if offset >= vocab_size:
                return False
            if offset + length > vocab_size:
                raise ValueError(f"Token range [{offset}, {offset + length}) exceeds vocab_size={vocab_size}")
            return True

        def fill_scalar(offset: int, bins, length: int) -> None:
            if not valid_range(offset, length):
                return
            centers = self._centers_from_bins(bins, length)
            if centers.numel() == 0:
                return
            features[offset : offset + length, 0] = self._normalize_to_minus1_plus1(centers)
            mask[offset : offset + length, 0] = 1.0

        def fill_phi(offset: int, bins, length: int) -> None:
            if not valid_range(offset, length):
                return
            centers = self._centers_from_bins(bins, length)
            if centers.numel() == 0:
                return
            features[offset : offset + length, 1] = torch.sin(centers)
            features[offset : offset + length, 2] = torch.cos(centers)
            mask[offset : offset + length, 0] = 1.0

        fill_scalar(d.ENERGY_OFFSET, d.e_bins, len(d.e_bins))
        fill_scalar(d.ETA_OFFSET, d.eta_bins, len(d.eta_bins))
        fill_scalar(d.THETA_OFFSET, d.theta_bins, len(d.theta_bins))
        fill_scalar(d.PT_OFFSET, d.pt_bins, len(d.pt_bins))
        fill_scalar(d.PX_OFFSET, d.px_bins, len(d.px_bins))
        fill_scalar(d.PY_OFFSET, d.py_bins, len(d.py_bins))
        fill_scalar(d.PZ_OFFSET, d.pz_bins, len(d.pz_bins))
        fill_phi(d.PHI_OFFSET, d.phi_bins, len(d.phi_bins))

        return features, mask

    # ----------------------------------------------------------------------------------
    # Parameter counting/init
    # ----------------------------------------------------------------------------------

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return parameter count.

        Args:
            non_embedding: If True, subtract learned embedding lookup tables from the
                total. The tied token embedding/output head is counted once by
                `named_parameters` and is subtracted once here. RoPE has no learned
                positional embedding to subtract.
        """
        n_params = sum(param.numel() for param in self.parameters())
        if not non_embedding:
            return n_params

        n_embed = self.transformer.wte.weight.numel()
        if hasattr(self.transformer, "type_emb"):
            n_embed += self.transformer.type_emb.weight.numel()
        if hasattr(self.transformer, "particle_index_emb"):
            n_embed += self.transformer.particle_index_emb.weight.numel()
        return n_params - n_embed

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize linear and embedding layers with GPT-2 style normal weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ----------------------------------------------------------------------------------
    # Losses
    # ----------------------------------------------------------------------------------

    def distance_sensitive_loss(
        self,
        logits: torch.Tensor,
        target_idxs: torch.Tensor,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Distance-sensitive cross entropy for continuous binned tokens.

        Continuous token types receive a Gaussian soft target over their own contiguous
        vocabulary block. Non-ordinal token types, such as PDGID and special tokens, fall
        back to ordinary hard-label cross entropy because neighboring token ids have no
        physical distance meaning there.

        Args:
            logits: Flattened logits with shape `(N, vocab_size)`.
            target_idxs: Flattened targets with shape `(N,)`.
            sigma: Width of the Gaussian target in token-bin units. Must be positive.
        """
        if self.config.data_mode != "particle":
            raise RuntimeError("distance_sensitive_loss is only valid for particle data_mode")
        d = _require_particle_dictionary("distance_sensitive_loss")
        
        if sigma <= 0:
            raise ValueError(f"sigma must be positive for distance_sensitive_loss, got {sigma}")

        valid_mask = target_idxs != d.padding_token
        if not valid_mask.any():
            return logits.sum() * 0.0

        valid_targets = target_idxs[valid_mask]
        valid_logits = logits[valid_mask]
        target_types = self.get_token_type_ids(valid_targets)

        total_loss = valid_logits.new_zeros(())
        count = 0
        handled = torch.zeros_like(valid_targets, dtype=torch.bool)

        for token_type_value in self.ORDINAL_TOKEN_TYPES:
            if token_type_value not in self.token_type_ranges:
                continue

            start, end = self.token_type_ranges[token_type_value]
            type_mask = target_types == token_type_value
            if not type_mask.any():
                continue

            group_targets = valid_targets[type_mask]
            group_logits = valid_logits[type_mask]
            block_len = end - start + 1

            pos_in_block = (group_targets - start).float().unsqueeze(1)
            offsets = torch.arange(block_len, device=logits.device, dtype=torch.float32).unsqueeze(0)

            abs_dist = torch.abs(pos_in_block - offsets)
            if token_type_value == ETokenTypes.PHI.value:
                distances = torch.minimum(abs_dist, block_len - abs_dist)
            else:
                distances = abs_dist

            soft_targets = torch.exp(-0.5 * (distances / sigma).square())
            soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp(min=1e-9)

            block_logits = group_logits[:, start : end + 1]
            log_z = torch.logsumexp(group_logits, dim=1)
            weighted_logit = (soft_targets * block_logits).sum(dim=1)
            per_sample_loss = -weighted_logit + log_z

            total_loss = total_loss + per_sample_loss.sum()
            count += int(group_targets.numel())
            handled[type_mask] = True

        hard_mask = ~handled
        if hard_mask.any():
            total_loss = total_loss + F.cross_entropy(
                valid_logits[hard_mask],
                valid_targets[hard_mask],
                reduction="sum",
            )
            count += int(hard_mask.sum().item())

        if count == 0:
            return logits.sum() * 0.0
        return total_loss / count

    # ----------------------------------------------------------------------------------
    # Embedding path
    # ----------------------------------------------------------------------------------

    def _particle_index_ids(
        self,
        idx: torch.Tensor,
        type_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Return particle-index embedding ids.

        Cached decoding only sees one token at a time, so computing particle ids via
        `cumsum` over the current mini-sequence would reset every step. When absolute
        `position_ids` are available, use the fixed event layout instead:

            position 0: EVENT_START
            positions 1..4: particle 0 features
            positions 5..8: particle 1 features
            ...

        Non-physics tokens map to id 0.
        """
        _require_particle_dictionary("particle index embeddings")
        physics_mask = (type_ids != ETokenTypes.PADDING.value) & (type_ids != ETokenTypes.SPECIAL.value)

        if position_ids is not None:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            if position_ids.size(0) == 1 and idx.size(0) > 1:
                position_ids = position_ids.expand(idx.size(0), -1)
            if position_ids.shape != idx.shape:
                raise ValueError(f"position_ids shape {tuple(position_ids.shape)} must match idx shape {tuple(idx.shape)}")

            physics_position = (position_ids - 1).clamp(min=0)
            particle_index = physics_position // self.config.num_features_per_particle
        else:
            physics_counter = torch.cumsum(physics_mask.long(), dim=1) - 1
            particle_index = physics_counter // self.config.num_features_per_particle

        particle_index = particle_index.clamp(min=0, max=self.config.max_particles_per_event - 1)
        return torch.where(physics_mask, particle_index + 1, torch.zeros_like(particle_index))

    def _embed_inputs(self, idx: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Shared embedding path for training, prefill, and cached decode.
        Keeping all call paths here prevents training/generation representation drift.
        
        In generic mode this is just token embeddings.
        In particle mode it can additionally use token-type, bin-value, and particle-index embeddings.
        """
        x = self.transformer.wte(idx)

        type_ids = None

        if self.config.use_token_type_embeddings:
            type_ids = self.get_token_type_ids(idx)
            x = x + self.transformer.type_emb(type_ids)

        if self.config.use_bin_value_embeddings:
            bin_features = self.bin_feature_lut[idx].to(dtype=x.dtype)
            bin_mask = self.bin_feature_mask[idx].to(dtype=x.dtype)
            bin_emb = self.transformer.bin_value_mlp(bin_features)
            x = x + self.bin_emb_scale.to(dtype=x.dtype) * bin_mask * bin_emb

        if self.config.use_particle_index_embeddings:
            if type_ids is None:
                type_ids = self.get_token_type_ids(idx)
            particle_index_ids = self._particle_index_ids(idx, type_ids, position_ids)
            x = x + self.transformer.particle_index_emb(particle_index_ids)

        x = self.transformer.emb_norm(x)
        return self.transformer.drop(x)

    # ----------------------------------------------------------------------------------
    # Sampling helpers
    # ----------------------------------------------------------------------------------

    @staticmethod
    def _sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Sample one token per batch row from `logits` with optional top-k filtering."""
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature

        if top_k is not None:
            if top_k <= 0:
                raise ValueError(f"top_k must be positive or None, got {top_k}")
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < values[:, [-1]], torch.finfo(logits.dtype).min)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    def mask_logits_for_next_token(self, logits: torch.Tensor, prev_token: torch.Tensor) -> torch.Tensor:
        """
        Apply the generation grammar for the current event format.

        Grammar:
            EVENT_START -> PDGID
            PDGID       -> PT
            PT          -> ETA
            ETA         -> PHI
            PHI         -> PDGID or EVENT_END
            EVENT_END   -> PAD
            PAD         -> PAD
        """
        if not self.config.use_event_grammar:
            raise RuntimeError("grammar_mask=True but config.use_event_grammar=False")
        d = _require_particle_dictionary("event grammar mask")
        
        allowed = torch.zeros_like(logits, dtype=torch.bool)
        prev_type = self.get_token_type_ids(prev_token)

        is_start = prev_token == d.event_start_token
        allowed[is_start, d.PDGID_OFFSET : d.PDGID_OFFSET + d.num_particles] = True

        is_pdgid = prev_type == ETokenTypes.PDGID.value
        allowed[is_pdgid, d.PT_OFFSET : d.PT_OFFSET + len(d.pt_bins)] = True

        is_pt = prev_type == ETokenTypes.PT.value
        allowed[is_pt, d.ETA_OFFSET : d.ETA_OFFSET + len(d.eta_bins)] = True

        is_eta = prev_type == ETokenTypes.ETA.value
        allowed[is_eta, d.PHI_OFFSET : d.PHI_OFFSET + len(d.phi_bins)] = True

        is_phi = prev_type == ETokenTypes.PHI.value
        allowed[is_phi, d.PDGID_OFFSET : d.PDGID_OFFSET + d.num_particles] = True
        allowed[is_phi, d.event_end_token] = True

        is_end = prev_token == d.event_end_token
        allowed[is_end, d.padding_token] = True

        is_pad = prev_token == d.padding_token
        allowed[is_pad, d.padding_token] = True

        # Safety fallback: never create a row of all -inf logits.
        empty_rows = ~allowed.any(dim=-1)
        allowed[empty_rows, :] = True

        return logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)

    @contextmanager
    def _temporary_eval_mode(self):
        """Temporarily switch to eval mode, then restore the original training flag."""
        was_training = self.training
        self.eval()
        try:
            yield
        finally:
            self.train(was_training)
    
    def _padding_token_id(self) -> int:
        """Padding id used for generation utilities."""
        if self.config.data_mode == "particle" and dictionary is not None:
            return int(dictionary.padding_token)
        return int(self.config.padding_token_id)

    def _eos_token_id(self) -> Optional[int]:
        """EOS id used for early stopping in generation. None means never early-stop."""
        if self.config.eos_token_id is not None:
            return int(self.config.eos_token_id)
        if self.config.data_mode == "particle" and dictionary is not None:
            return int(dictionary.event_end_token)
        return None

    def _loss_ignore_index(self) -> int:
        """Ignore index for CE loss. Generic LM defaults to -100, particle mode to PAD."""
        if self.config.loss_ignore_index is not None:
            return int(self.config.loss_ignore_index)
        if self.config.data_mode == "particle" and dictionary is not None:
            return int(dictionary.padding_token)
        return -100

    # ----------------------------------------------------------------------------------
    # Forward paths
    # ----------------------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """
        General forward pass.

        Usage:
            * Training/eval full sequence: `model(idx, targets)`
            * Inference prefill: `model.forward_prefill(prompt)`
            * Cached decode: `model.forward_decode(one_token, kv_cache)`
        """
        if idx.dim() != 2:
            raise ValueError(f"idx must have shape (B, T), got {tuple(idx.shape)}")

        device = idx.device
        batch_size, seq_len = idx.shape

        if kv_cache is not None:
            use_cache = True
            if len(kv_cache) != len(self.transformer.h):
                raise ValueError(f"kv_cache must have one entry per layer; got {len(kv_cache)}")
            
        cache_len = 0
        if kv_cache is not None and kv_cache[0] is not None:
            cache_len = kv_cache[0][0].size(2)

        if cache_len + seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward total sequence length {cache_len + seq_len}; "
                f"block_size is only {self.config.block_size}."
            )
            
        # Only create absolute position ids when we actually need them.
        # For normal training/eval with no cache, leave position_ids=None so attention
        # can use the fast RoPE slice path.
        if position_ids is None and cache_len > 0:
            position_ids = torch.arange(
                cache_len,
                cache_len + seq_len,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(batch_size, -1)

        x = self._embed_inputs(idx, position_ids=position_ids)

        layer_past = kv_cache if kv_cache is not None else [None] * len(self.transformer.h)
        new_kv_cache = [] if use_cache else None

        for block, block_cache in zip(self.transformer.h, layer_past):
            x, new_block_kv_cache = block(
                x,
                kv_cache=block_cache,
                attn_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_cache.append(new_block_kv_cache)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)

            if conf.training.loss_function == "cross_entropy":
                loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=self._loss_ignore_index())
            elif conf.training.loss_function == "distance_sensitive":
                loss = self.distance_sensitive_loss(
                    flat_logits,
                    flat_targets,
                    sigma=getattr(conf.training, "loss_sigma", 1.0),
                )
            else:
                raise ValueError(f"Unknown loss function: {conf.training.loss_function}")
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_kv_cache

    @torch.no_grad()
    def forward_prefill(self, idx: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Process a full prompt once and return last-token logits plus the full KV cache.

        Args:
            idx: Prompt token ids with shape `(B, T_prompt)`.
        """
        if idx.dim() != 2:
            raise ValueError(f"idx must have shape (B, T), got {tuple(idx.shape)}")

        batch_size, seq_len = idx.shape
        position_ids = torch.arange(seq_len, device=idx.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        logits, _, kv_cache = self.forward(
            idx,
            targets=None,
            kv_cache=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        return logits, kv_cache

    @torch.no_grad()
    def forward_decode(self, idx: torch.Tensor, kv_cache: list[tuple[torch.Tensor, torch.Tensor]]):
        """
        Decode one new token using an existing KV cache.

        Args:
            idx: New token ids with shape `(B, 1)`.
            kv_cache: Layer-wise cache returned by `forward_prefill` or this method.
        """
        if idx.dim() != 2 or idx.size(1) != 1:
            raise ValueError(f"forward_decode expects idx with shape (B, 1), got {tuple(idx.shape)}")
        if kv_cache is None or kv_cache[0] is None:
            raise ValueError("forward_decode requires an existing kv_cache from forward_prefill")

        batch_size = idx.size(0)
        cache_len = kv_cache[0][0].size(2)
        position_ids = torch.full((batch_size, 1), cache_len, dtype=torch.long, device=idx.device)

        logits, _, kv_cache = self.forward(
            idx,
            targets=None,
            kv_cache=kv_cache,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=True,
        )
        return logits, kv_cache

    # ----------------------------------------------------------------------------------
    # Maintenance / optimizer / diagnostics
    # ----------------------------------------------------------------------------------

    def crop_block_size(self, block_size: int) -> None:
        """Reduce max context length by cropping all RoPE caches in-place."""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if block_size > self.config.block_size:
            raise ValueError(f"Cannot grow block_size from {self.config.block_size} to {block_size}")

        self.config.block_size = block_size
        for block in self.transformer.h:
            block.attn.cos_cached = block.attn.cos_cached[:, :, :block_size, :].contiguous()
            block.attn.sin_cached = block.attn.sin_cached[:, :, :block_size, :].contiguous()

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas, device_type: str):
        """Create AdamW parameter groups with decay excluded from norms/biases/embeddings."""
        param_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}

        decay_params = []
        nodecay_params = []

        def is_embedding_param(name: str) -> bool:
            return (
                name == "transformer.wte.weight"
                or name == "lm_head.weight"
                or "type_emb" in name
                or "particle_index_emb" in name
            )

        for name, param in param_dict.items():
            if param.dim() >= 2 and not is_embedding_param(name):
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        pLogging.info(
            logger_idx,
            "Optimizer info",
            {
                "num_decay_params": sum(param.numel() for param in decay_params),
                "num_nodecay_params": sum(param.numel() for param in nodecay_params),
            },
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        pLogging.info(logger_idx, "Optimizer info", {"use_fused_adamw": use_fused})
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model FLOPs utilization relative to A100 bf16 peak throughput."""
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        n_params = self.get_num_params()
        cfg = self.config
        n_layers = cfg.n_layer
        n_heads = cfg.n_head
        head_dim = cfg.n_embd // cfg.n_head
        seq_len = cfg.block_size

        flops_per_token = 6 * n_params + 12 * n_layers * n_heads * head_dim * seq_len
        flops_per_fwdbwd = flops_per_token * seq_len
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        a100_bf16_peak_flops = 312e12
        return flops_achieved / a100_bf16_peak_flops

    # ----------------------------------------------------------------------------------
    # Generation
    # ----------------------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        grammar_mask: bool = False,
    ) -> torch.Tensor:
        """Generate continuations for one batch using the cached prefill/decode path."""
        with self._temporary_eval_mode():
            if idx.dim() == 1:
                idx = idx.unsqueeze(0)
            if idx.dim() != 2:
                raise ValueError(f"idx must have shape (B, T), got {tuple(idx.shape)}")

            device = idx.device
            batch_size, prompt_len = idx.shape
            max_new_tokens = min(max_new_tokens, self.config.block_size - prompt_len)
            if max_new_tokens <= 0:
                return idx
            
            pad_token_id = self._padding_token_id()
            eos_token_id = self._eos_token_id()

            output = torch.full(
                (batch_size, prompt_len + max_new_tokens),
                fill_value=pad_token_id,
                dtype=torch.long,
                device=device,
            )
            output[:, :prompt_len] = idx

            logits, kv_cache = self.forward_prefill(idx)
            logits = logits[:, -1, :]

            prev_token = idx[:, -1]
            if grammar_mask:
                logits = self.mask_logits_for_next_token(logits, prev_token)

            next_tokens = self._sample_next_token(logits, temperature=temperature, top_k=top_k)
            output[:, prompt_len] = next_tokens
            unfinished = (
                torch.ones_like(next_tokens, dtype=torch.bool)
                if eos_token_id is None
                else next_tokens != eos_token_id
            )

            for step in range(1, max_new_tokens):
                input_ids = output[:, prompt_len + step - 1].unsqueeze(1)
                logits, kv_cache = self.forward_decode(input_ids, kv_cache)
                logits = logits[:, -1, :]

                if grammar_mask:
                    logits = self.mask_logits_for_next_token(logits, input_ids[:, 0])

                next_tokens = self._sample_next_token(logits, temperature=temperature, top_k=top_k)
                if eos_token_id is not None:
                    next_tokens = torch.where(
                        unfinished,
                        next_tokens,
                        torch.full_like(next_tokens, pad_token_id),
                    )

                output[:, prompt_len + step] = next_tokens
                if eos_token_id is not None:
                    unfinished = unfinished & (next_tokens != eos_token_id)
                    if not unfinished.any():
                        break

            return output

    @torch.no_grad()
    def generate_batched_singleGPU(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
        grammar_mask: bool = False,
    ) -> torch.Tensor:
        """
        Generate many samples on one GPU using fixed-length prompt batches.

        Assumption: every row has the same non-padding prompt length. This works since the
        prompt is always (EVENT_START, PDGID, PT, ETA, PHI), kinematics being for the incident
        particle.
        """
        with self._temporary_eval_mode():
            if idx.dim() == 1:
                idx = idx.unsqueeze(0)
            if idx.dim() != 2:
                raise ValueError(f"idx must have shape (B, T), got {tuple(idx.shape)}")

            device = idx.device
            original_batch_size = idx.size(0)
            batch_size = original_batch_size if batch_size is None else batch_size
            if batch_size <= 0:
                raise ValueError(f"batch_size must be positive, got {batch_size}")

            pad_token_id = self._padding_token_id()
            eos_token_id = self._eos_token_id()

            seq_lens = (idx != pad_token_id).sum(dim=1)
            if not torch.all(seq_lens == seq_lens[0]):
                raise ValueError(
                    "generate_batched_singleGPU expects all prompts to have the same "
                    "non-padding length."
                )

            prompt_len = int(seq_lens[0].item())
            idx = idx[:, :prompt_len]
            max_new_tokens = min(max_new_tokens, self.config.block_size - prompt_len)
            if max_new_tokens <= 0:
                return idx

            output = torch.full(
                (original_batch_size, prompt_len + max_new_tokens),
                fill_value=pad_token_id,
                dtype=torch.long,
                device=device,
            )
            output[:, :prompt_len] = idx

            for start in range(0, original_batch_size, batch_size):
                end = min(start + batch_size, original_batch_size)
                cur_prompt = idx[start:end]
                cur_output = output[start:end]

                logits, kv_cache = self.forward_prefill(cur_prompt)
                logits = logits[:, -1, :]

                prev_token = cur_prompt[:, -1]
                if grammar_mask:
                    logits = self.mask_logits_for_next_token(logits, prev_token)

                next_tokens = self._sample_next_token(logits, temperature=temperature, top_k=top_k)
                cur_output[:, prompt_len] = next_tokens
                unfinished = (
                    torch.ones_like(next_tokens, dtype=torch.bool)
                    if eos_token_id is None
                    else next_tokens != eos_token_id
                )

                for step in range(1, max_new_tokens):
                    input_ids = cur_output[:, prompt_len + step - 1].unsqueeze(1)
                    logits, kv_cache = self.forward_decode(input_ids, kv_cache)
                    logits = logits[:, -1, :]

                    if grammar_mask:
                        logits = self.mask_logits_for_next_token(logits, input_ids[:, 0])

                    next_tokens = self._sample_next_token(logits, temperature=temperature, top_k=top_k)
                    if eos_token_id is not None:
                        next_tokens = torch.where(
                            unfinished,
                            next_tokens,
                            torch.full_like(next_tokens, pad_token_id),
                        )

                    cur_output[:, prompt_len + step] = next_tokens
                    if eos_token_id is not None:
                        unfinished = unfinished & (next_tokens != eos_token_id)
                        if not unfinished.any():
                            break

            return output

    @torch.no_grad()
    def generate_batched_multiGPU(
        self,
        starters: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        batch_size: int = 128,
        grammar_mask: bool = False,
    ) -> torch.Tensor:
        """Generate samples in parallel across all visible CUDA devices."""
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU sampling requires CUDA devices.")
        if starters.dim() != 2:
            raise ValueError(f"starters must have shape (N, T), got {tuple(starters.shape)}")
        if starters.size(0) == 0:
            return starters

        num_devices = torch.cuda.device_count()
        starters_cpu = starters.detach().cpu()
        starter_splits = torch.chunk(starters_cpu, min(num_devices, starters_cpu.size(0)), dim=0)

        ctx = mp.get_context("spawn")
        return_queue = ctx.Queue()

        # Keep the state dict on CPU before spawning.  This avoids pickling live CUDA
        # tensors and reduces parent-process GPU memory pressure.
        model_state_dict = {name: tensor.detach().cpu() for name, tensor in self.state_dict().items()}
        model_config = self.config

        processes = []
        for split_index, starters_chunk in enumerate(starter_splits):
            device_id = split_index % num_devices
            process = ctx.Process(
                target=batched_multiGPU_worker,
                args=(
                    split_index,
                    device_id,
                    model_config,
                    model_state_dict,
                    starters_chunk,
                    max_new_tokens,
                    temperature,
                    top_k,
                    batch_size,
                    grammar_mask,
                    return_queue,
                ),
            )
            process.start()
            processes.append(process)

        results: list[Optional[torch.Tensor]] = [None] * len(processes)
        errors = []
        for _ in range(len(processes)):
            split_index, generated, error = return_queue.get()
            if error is not None:
                errors.append((split_index, error))
            else:
                results[split_index] = generated

        for process in processes:
            process.join()

        if errors:
            raise RuntimeError(f"One or more generation workers failed: {errors}")

        return torch.cat([result for result in results if result is not None], dim=0)