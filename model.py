"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import json
import math
import inspect
from pathlib import Path
from dataclasses import dataclass
import multiprocessing as mp

import torch
import torch.nn as nn
from torch.nn import functional as F

import pLogging
import configurator as conf
from dictionary import Dictionary
from dictionary import ETokenTypes

script_dir = Path(__file__).resolve().parent

# Model will never be used on its own. It will be accessed by the training or generation script.
# Therefore, we inherit the logger from the accessing script.
logger_idx = -1
def set_logger(in_logger_idx):
    global logger_idx
    logger_idx = in_logger_idx

# ===== Load the dictionary =====

try:
    prep_filepath = script_dir / 'preparations' / conf.generic.preparation_name / 'preparation.json'
    with open(prep_filepath, 'r') as f:
        prep_data = json.load(f)
    prop_tokenized_dataset_name = prep_data['train_bin']['tokenized_dataset']
    
    dictionary_filepath = script_dir / 'data' / 'tokenized' / prop_tokenized_dataset_name / 'dictionary.json'
    dictionary = Dictionary(dictionary_filepath)
except Exception as e:
    pLogging.info(logger_idx, f"Error occurred while trying to load dictionary: {e}")
    raise Exception("Error occurred while trying to load dictionary!")

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes over the final dimension:
        x -> x / sqrt(mean(x^2) + eps)

    Uses fp32 internally for numerical stability under bf16/fp16 autocast.

    init_scale:
        Initial value for the learned scale parameter.
        For normal block RMSNorm, use 1.0.
        For embedding norm experiments, you can also test smaller values.
    """

    def __init__(self, ndim, eps=1e-5, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim) * init_scale)
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype

        # Do RMS calculation in fp32 for numerical stability.
        x_float = x.float()
        x_normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Cast back to original dtype.
        return (self.weight * x_normed).to(input_dtype)

def build_embedding_norm(config):
    """
    Build the optional embedding normalization layer.

    This normalization is applied after summing token/type embeddings
    and before dropout.
    """

    norm_type = getattr(config, "embedding_norm_type", "none")

    if norm_type == "none":
        return nn.Identity()

    elif norm_type == "rmsnorm":
        return RMSNorm(
            config.n_embd,
            init_scale=getattr(config, "embedding_norm_init_scale", 1.0),
        )

    else:
        raise ValueError(f"Unknown embedding_norm_type: {norm_type}")
    
class HeadRMSNorm(nn.Module):
    """
    RMSNorm over the per-head dimension.

    Input shape:
        (B, n_heads, T, head_dim)
    """

    def __init__(self, head_dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.float()
        x_normed = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (self.weight * x_normed).to(input_dtype)
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # ===== GQA setup =====
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads else config.n_head
        assert self.n_head % self.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"
        self.n_head_groups = self.n_head // self.n_kv_heads   # how many Q heads share one KV head
        self.head_dim = config.n_embd // self.n_head
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, (self.n_head + 2 * self.n_kv_heads) * self.head_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.q_norm = HeadRMSNorm(self.head_dim)
        self.k_norm = HeadRMSNorm(self.head_dim)

        # flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            raise Exception("Flash attention is not available in this PyTorch version. Please upgrade to PyTorch 2.0 or later.")
        
        # ===== RoPE pre-computation =====
        # Pre‑compute a matrix of cos and sin values for all positions up to block_size and all head dimensions. 
        # The rotation is applied pairwise to half of the head dimensions
        head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim//2,), theta = 10,000 from the original RoPE paper
        # cache for max block_size
        t = torch.arange(config.block_size, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)                                         # (block_size, head_dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)                                  # (block_size, head_dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))  # (1, 1, block_size, head_dim)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))  # (1, 1, block_size, head_dim)

    def forward(self, x, kv_cache=None, attn_mask=None, position_ids=None):
        B, T, C = x.size()

        # Project all Q, K, V together
        qkv = self.c_attn(x)   # (B, T, (n_head + 2 * n_kv_heads) * head_dim)

        # Split into Q, K, V portions
        q_size = self.n_head * self.head_dim
        k_size = self.n_kv_heads * self.head_dim
        v_size = self.n_kv_heads * self.head_dim

        q, k, v = qkv.split([q_size, k_size, v_size], dim=2)
        # Reshape Q: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Reshape K and V: (B, T, n_kv_heads, head_dim) -> (B, n_kv_heads, T, head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ===== RoPE rotation (contiguous positions 0 .. T-1) =====
        cos = self.cos_cached[:, :, :T, :]   # (1, 1, T, head_dim)
        sin = self.sin_cached[:, :, :T, :]

        def rotate_half(t):
            t1 = t[..., : t.shape[-1] // 2]
            t2 = t[..., t.shape[-1] // 2 :]
            return torch.cat((-t2, t1), dim=-1)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        # ===== KV cache handling (works because k,v are already expanded) =====
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat((prev_k, k), dim=2)
            v = torch.cat((prev_v, v), dim=2)
        new_kv_cache = (k, v) if kv_cache is not None else None

        # ===== Attention =====
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False if attn_mask is not None else True,
            enable_gqa=(self.n_head_groups > 1)
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv_cache

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class RELU2MLP(nn.Module):
    """
    ReLU^2 MLP.

    Structure:
        x -> Linear(n_embd, 4*n_embd)
          -> relu(x)^2
          -> Linear(4*n_embd, n_embd)
          -> dropout

    Parameter count is essentially identical to the GELU version.
    """

    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        # ReLU squared activation.
        # This is relu(x)^2, not relu(x^2).
        x = F.relu(x).square()
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        hidden = 256 * ((hidden + 255) // 256)  # tensor-core-friendly
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w2 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w3 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        x = self.dropout(x)
        return x

# Better Block class to support RMS norm
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # RMSNorm replaces LayerNorm.
        # It only needs the embedding dimension; it does not use bias.
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        # @TODO: make this configurable between different MLP types, not just the ReLU^2 one we're testing in upgrade 3
        # self.mlp = MLP(config)
        # self.mlp = SwiGLUMLP(config)
        self.mlp = RELU2MLP(config)

    def forward(self, x, kv_cache=None, attn_mask=None, position_ids=None):
        # Pre-norm attention block:
        # normalize x, run attention, then add residual.
        attn_out, new_kv_cache = self.attn(
            self.ln_1(x),
            kv_cache=kv_cache,
            attn_mask=attn_mask,
            position_ids=position_ids,
        )
        x = x + attn_out
        # Pre-norm MLP block:
        # normalize x, run MLP, then add residual.
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache

# The defaults are not GPT-2, which is irrelevant for us.
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_heads: int = 0          # if 0 or None, defaults to n_head (standard MHA)
    dropout: float = 0.0
    bias: bool = True
    num_token_types: int = len(ETokenTypes)
    
    embedding_norm_type: str = "none"  # options: "none", "rmsnorm"
    embedding_norm_init_scale: float = 0.02
    
    use_particle_index_embeddings: bool = False
    # @TODO: if we plant to keep these, make sure to implement auto-detection of these values.
    # @IMPORTANT: see above todo
    num_features_per_particle: int = 4
    max_particles_per_event: int = 42
    
    # Ordinal embeddings
    use_bin_value_embeddings: bool = True
    bin_embedding_init_scale: float = 0.0

# Helper for batch sampling. Needs to be outside any class due to python forking rules.
def batched_multiGPU_worker(device_id, model_config, model_state_dict, starters_chunk, max_new_tokens, temperature, top_k, batch_size, return_queue):
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')

    # Rebuild model
    model = GPT(model_config).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    starters_chunk = starters_chunk.to(device, non_blocking=True)
    generated = model.generate_batched_singleGPU(
        starters_chunk,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        batch_size=batch_size
    )

    return_queue.put((device_id, generated.cpu()))
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        modules = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Type embeddings tell the model whether the token is PDGID, pT, eta, phi, etc.
            type_emb = nn.Embedding(config.num_token_types, config.n_embd),
            # optional normalization after summing token + type embeddings.
            emb_norm = build_embedding_norm(config),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        )
        
        if config.use_bin_value_embeddings:
            modules["bin_value_mlp"] = nn.Sequential(
                nn.Linear(3, config.n_embd, bias=config.bias),
                nn.SiLU(),
                nn.Linear(config.n_embd, config.n_embd, bias=config.bias),
            )
        if config.use_particle_index_embeddings:
            # index 0 reserved for special/pad.
            # real particles use 1, 2, 3, ...
            # row 0 = no particle / START / END / PAD
            # row 1 = particle 0
            # row 2 = particle 1
            # ...
            modules['particle_index_emb'] = nn.Embedding(config.max_particles_per_event + 1, config.n_embd)

        self.transformer = nn.ModuleDict(modules)
        
        if config.use_bin_value_embeddings:
            bin_feature_lut, bin_feature_mask = self._build_bin_feature_lut(config.vocab_size)
            self.register_buffer("bin_feature_lut", bin_feature_lut, persistent=False)
            self.register_buffer("bin_feature_mask", bin_feature_mask, persistent=False)
            self.bin_emb_scale = nn.Parameter(torch.tensor(float(config.bin_embedding_init_scale)))
            
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                
        # Pre‑compute token type ranges for efficient distance‑sensitive loss
        self._precompute_token_type_ranges()

        # report number of parameters
        pLogging.info(logger_idx, "Model info", {"num_params": str(self.get_num_params() / 1e6) + "M" })
        pLogging.info(logger_idx, "Model info", {"num_params_embedding_incl": str(self.get_num_params(non_embedding=False) / 1e6) + "M" })
    
    def _centers_from_bins(self, bins, expected_len):
        """
        Convert a bin array into bin centers.

        Handles:
        1. empty bin lists
        2. bins already being centers
        3. bins being edges
        """
        values = torch.as_tensor(bins, dtype=torch.float32).flatten()

        if expected_len == 0 or values.numel() == 0:
            return torch.empty(0, dtype=torch.float32)

        if values.numel() == expected_len:
            return values

        if values.numel() == expected_len + 1:
            return 0.5 * (values[:-1] + values[1:])

        raise ValueError(
            f"Cannot infer bin centers. Got {values.numel()} values, "
            f"expected {expected_len} centers or {expected_len + 1} edges."
        )

    def _normalize_to_minus1_plus1(self, values):
        """
        Normalize scalar bin centers to [-1, 1].

        Handles empty tensors safely because some token types may not exist
        for the current tokenization, e.g. ENERGY bins when using pdgid/pt/eta/phi.
        """
        if values.numel() == 0:
            return values

        vmin = values.min()
        vmax = values.max()

        if torch.isclose(vmin, vmax):
            return torch.zeros_like(values)

        return 2.0 * (values - vmin) / (vmax - vmin) - 1.0


    def _build_bin_feature_lut(self, vocab_size):
        """
        Build lookup tables for ordinal/bin embeddings.

        features: (vocab_size, 3)
            [:, 0] = normalized scalar coordinate
            [:, 1] = sin(phi)
            [:, 2] = cos(phi)

        mask: (vocab_size, 1)
            1 for continuous binned tokens
            0 for PDGID, special, padding, absent token types, etc.
        """
        features = torch.zeros(vocab_size, 3, dtype=torch.float32)
        mask = torch.zeros(vocab_size, 1, dtype=torch.float32)

        def valid_range(offset, length):
            if length is None or length <= 0:
                return False
            if offset is None:
                return False
            if offset < 0:
                return False
            if offset >= vocab_size:
                return False
            if offset + length > vocab_size:
                raise ValueError(
                    f"Token range [{offset}, {offset + length}) exceeds vocab_size={vocab_size}"
                )
            return True

        def fill_scalar(name, offset, bins, length):
            if not valid_range(offset, length):
                return

            centers = self._centers_from_bins(bins, length)

            # Still skip if bins are unexpectedly empty.
            if centers.numel() == 0:
                return

            values = self._normalize_to_minus1_plus1(centers)

            features[offset : offset + length, 0] = values
            mask[offset : offset + length, 0] = 1.0

        def fill_phi(name, offset, bins, length):
            if not valid_range(offset, length):
                return

            centers = self._centers_from_bins(bins, length)

            if centers.numel() == 0:
                return

            # If phi bins are actual radians, this is physically meaningful.
            features[offset : offset + length, 1] = torch.sin(centers)
            features[offset : offset + length, 2] = torch.cos(centers)
            mask[offset : offset + length, 0] = 1.0

        fill_scalar("ENERGY", dictionary.ENERGY_OFFSET, dictionary.e_bins, len(dictionary.e_bins))
        fill_scalar("ETA", dictionary.ETA_OFFSET, dictionary.eta_bins, len(dictionary.eta_bins))
        fill_scalar("THETA", dictionary.THETA_OFFSET, dictionary.theta_bins, len(dictionary.theta_bins))
        fill_scalar("PT", dictionary.PT_OFFSET, dictionary.pt_bins, len(dictionary.pt_bins))
        fill_scalar("PX", dictionary.PX_OFFSET, dictionary.px_bins, len(dictionary.px_bins))
        fill_scalar("PY", dictionary.PY_OFFSET, dictionary.py_bins, len(dictionary.py_bins))
        fill_scalar("PZ", dictionary.PZ_OFFSET, dictionary.pz_bins, len(dictionary.pz_bins))

        fill_phi("PHI", dictionary.PHI_OFFSET, dictionary.phi_bins, len(dictionary.phi_bins))

        return features, mask

    def _precompute_token_type_ranges(self):
        """
        Build a dict mapping token type value -> (start, end) inclusive
        for every contiguous vocabulary block, using the global dictionary.
        """
        self.token_type_ranges = {}
        for ttype in ETokenTypes:
            if ttype == ETokenTypes.PADDING:
                continue
            # Map enum member to the correct start + length
            if ttype == ETokenTypes.SPECIAL:
                start = dictionary.SPECIAL_TOKENS_OFFSET + 1
                length = dictionary.num_special_tokens
            elif ttype == ETokenTypes.PDGID:
                start = dictionary.PDGID_OFFSET
                length = dictionary.num_particles
            elif ttype == ETokenTypes.ENERGY:
                start = dictionary.ENERGY_OFFSET
                length = len(dictionary.e_bins)
            elif ttype == ETokenTypes.ETA:
                start = dictionary.ETA_OFFSET
                length = len(dictionary.eta_bins)
            elif ttype == ETokenTypes.THETA:
                start = dictionary.THETA_OFFSET
                length = len(dictionary.theta_bins)
            elif ttype == ETokenTypes.PHI:
                start = dictionary.PHI_OFFSET
                length = len(dictionary.phi_bins)
            elif ttype == ETokenTypes.PT:
                start = dictionary.PT_OFFSET
                length = len(dictionary.pt_bins)
            elif ttype == ETokenTypes.PX:
                start = dictionary.PX_OFFSET
                length = len(dictionary.px_bins)
            elif ttype == ETokenTypes.PY:
                start = dictionary.PY_OFFSET
                length = len(dictionary.py_bins)
            elif ttype == ETokenTypes.PZ:
                start = dictionary.PZ_OFFSET
                length = len(dictionary.pz_bins)
            else:
                continue
            if length > 0:
                self.token_type_ranges[ttype.value] = (start, start + length - 1)
    
    def get_token_type_ids(self, idx):
        # Map token id ranges to type ids
        type_ids = torch.zeros_like(idx)
        type_ids[(idx == 0)]                                                                                                               = ETokenTypes.PADDING.value
        type_ids[(idx >= dictionary.SPECIAL_TOKENS_OFFSET + 1) & (idx < dictionary.SPECIAL_TOKENS_OFFSET + dictionary.num_special_tokens)] = ETokenTypes.SPECIAL.value
        type_ids[(idx >= dictionary.PDGID_OFFSET)              & (idx < dictionary.PDGID_OFFSET + dictionary.num_particles)]               = ETokenTypes.PDGID.value
        type_ids[(idx >= dictionary.ENERGY_OFFSET)             & (idx < dictionary.ENERGY_OFFSET + len(dictionary.e_bins))]                = ETokenTypes.ENERGY.value
        type_ids[(idx >= dictionary.ETA_OFFSET)                & (idx < dictionary.ETA_OFFSET + len(dictionary.eta_bins))]                 = ETokenTypes.ETA.value
        type_ids[(idx >= dictionary.THETA_OFFSET)              & (idx < dictionary.THETA_OFFSET + len(dictionary.theta_bins))]             = ETokenTypes.THETA.value
        type_ids[(idx >= dictionary.PHI_OFFSET)                & (idx < dictionary.PHI_OFFSET + len(dictionary.phi_bins))]                 = ETokenTypes.PHI.value
        type_ids[(idx >= dictionary.PT_OFFSET)                 & (idx < dictionary.PT_OFFSET + len(dictionary.pt_bins))]                   = ETokenTypes.PT.value
        type_ids[(idx >= dictionary.PX_OFFSET)                 & (idx < dictionary.PX_OFFSET + len(dictionary.px_bins))]                   = ETokenTypes.PX.value
        type_ids[(idx >= dictionary.PY_OFFSET)                 & (idx < dictionary.PY_OFFSET + len(dictionary.py_bins))]                   = ETokenTypes.PY.value
        type_ids[(idx >= dictionary.PZ_OFFSET)                 & (idx < dictionary.PZ_OFFSET + len(dictionary.pz_bins))]                   = ETokenTypes.PZ.value
        return type_ids

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters.
        For non-embedding count, we only subtract positions if they existed.
        Since RoPE removes wpe, we simply return sum of all params.
        (@NOTE: perhaps I would want to exclude token embeddings later)
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # no position embeddings to subtract
            pass
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   
    """
    Custom loss function
    
    This function considers distance from correct bin when deciding a penalty for tokens.
    e.g. say the correct token is 81 and we predict 80 or 82.
    
    This custom loss function aims to consider the physical reasonability of the predicted token.
    This function does this by penalizing the bins based on (1) being in the right range for the
    type of token this is (i.e. for a pt token it following the bounds for pt), and (2) by
    defining the penalty as a function of distance from the correct token (i.e. 80 is still ok,
    whereas 700 is unacceptable).
    """
    def distance_sensitive_loss(self, logits, target_idxs, sigma=1.0):
        """
        Efficient distance‑sensitive loss that never materialises a full
        (batch, vocab_size) soft‑target tensor.

        logits:      (N, vocab_size)   – already flattened (B*T, V)
        target_idxs: (N,)              – ground truth token indices
        sigma:       Gaussian width    (configurable, e.g. from conf.training.loss_sigma)
        """
        ignore_index = dictionary.padding_token
        mask = (target_idxs != ignore_index)
        if mask.sum() == 0:
            # No valid tokens: return a zero loss that still carries requires_grad
            return logits.sum() * 0.0

        valid_targets = target_idxs[mask]          # (K,)
        valid_logits = logits[mask]                # (K, V)

        # token types for every valid target
        target_types = self.get_token_type_ids(valid_targets)   # (K,)

        total_loss = 0.0
        count = 0

        # Process one token type at a time. The number of types is tiny (≤ 10).
        for ttype_val, (start, end) in self.token_type_ranges.items():
            type_hit = (target_types == ttype_val)
            if not type_hit.any():
                continue

            # Gather all tokens of this type
            group_targets = valid_targets[type_hit]                # (M,)
            group_logits = valid_logits[type_hit]                  # (M, V)

            M = group_targets.size(0)
            block_len = end - start + 1

            # ---- Gaussian weights inside the block ----
            pos_in_block = (group_targets - start).float()   # (M,)
            offsets = torch.arange(block_len, device=logits.device).float().unsqueeze(0)     # (1, L)
            distances = pos_in_block.unsqueeze(1) - offsets                                  # (M, L)
            gauss = torch.exp(-0.5 * (distances / sigma) ** 2)
            gauss = gauss / gauss.sum(dim=1, keepdim=True).clamp(min=1e-9)   # normalised soft targets

            # ---- Extract logits of the block only ----
            block_logits = group_logits[:, start:end+1]             # (M, L)

            # ---- logsumexp over the FULL vocabulary ----
            logZ = torch.logsumexp(group_logits, dim=1)             # (M,)

            # ---- Cross-entropy H(p, q) = -∑ p * log q = -∑ p * l_block + logZ ----
            weighted_logsum = (gauss * block_logits).sum(dim=1)     # (M,)
            per_sample_loss = -weighted_logsum + logZ               # (M,)

            total_loss += per_sample_loss.sum()
            count += M

        if count == 0:
            return logits.sum() * 0.0
        return total_loss / count
    
    def forward(self, idx, targets=None, kv_cache=None, attention_mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        
        # Position IDs: start from 0, or extend from cache length if kv_cache exists
        if kv_cache is None or kv_cache[0] is None:
            position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        else:
            # When using KV cache, the input is just the new token.
            # The cache already holds the previous keys/values, and their positions are 0..t-1.
            # The new token is at position = length of cache.
            cache_len = kv_cache[0][0].size(2)   # shape (B, nh, T, hs) -> T
            position_ids = torch.full((b, 1), cache_len, dtype=torch.long, device=device)  # (B, 1)

        # New embedding normalization version
        tok_emb = self.transformer.wte(idx)
        type_ids = self.get_token_type_ids(idx)
        type_emb = self.transformer.type_emb(type_ids)
        # Sum all input embedding components.
        x = tok_emb + type_emb
        
        # Build bin/ordinal embeddings
        if self.config.use_bin_value_embeddings:
            # Shape: (B, T, 3)
            bin_features = self.bin_feature_lut[idx].to(dtype=x.dtype)
            # Shape: (B, T, 1)
            bin_mask = self.bin_feature_mask[idx].to(dtype=x.dtype)
            # Shape: (B, T, n_embd)
            bin_emb = self.transformer.bin_value_mlp(bin_features)
            # Add only for continuous binned variables.
            # bin_emb_scale starts at 0.0, so the model begins as the baseline.
            x = x + self.bin_emb_scale.to(dtype=x.dtype) * bin_mask * bin_emb
        
        # Build particle structure index embeddings if enabled.
        if self.config.use_particle_index_embeddings:
            # Physics tokens are only the repeated:
            #   pdgid pt eta phi pdgid pt eta phi ...
            # EVENT_START, EVENT_END, and PAD are not assigned a real particle index.
            physics_mask = (
                (type_ids != ETokenTypes.PADDING.value) &
                (type_ids != ETokenTypes.SPECIAL.value)
            )

            # Count physics tokens only.
            # Example:
            #   START pdgid pt eta phi pdgid pt eta phi END PAD
            # mask:
            #     0     1   1  1   1    1   1  1   1   0   0
            # counter before masking:
            #    -1     0   1  2   3    4   5  6   7   7   7
            physics_counter = torch.cumsum(physics_mask.long(), dim=1) - 1

            # Convert token counter into particle number:
            #   token counters 0,1,2,3 -> particle 0
            #   token counters 4,5,6,7 -> particle 1
            particle_index = physics_counter // self.config.num_features_per_particle

            # Clamp for safety.
            particle_index = particle_index.clamp(
                min=0,
                max=self.config.max_particles_per_event - 1,
            )

            # Separate namespace:
            #   0 = no particle index for START/END/PAD
            #   1 = particle 0
            #   2 = particle 1
            #   ...
            particle_index_ids = torch.where(
                physics_mask,
                particle_index + 1,
                torch.zeros_like(particle_index),
            )

            x = x + self.transformer.particle_index_emb(particle_index_ids)
        
        # normalize the embedding stream before entering the Transformer blocks.
        x = self.transformer.emb_norm(x)
        # Keep dropout after normalization.
        x = self.transformer.drop(x)

        new_kv_cache = []
        if kv_cache is None:
            kv_cache = [None] * len(self.transformer.h)

        for i, block in enumerate(self.transformer.h):
            x, new_block_kv_cache = block(x, kv_cache=kv_cache[i], attn_mask=attention_mask, position_ids=position_ids)
            new_kv_cache.append(new_block_kv_cache)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            if conf.training.loss_function == 'cross_entropy':
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=dictionary.padding_token)
            elif conf.training.loss_function == 'distance_sensitive':
                loss = self.distance_sensitive_loss(logits.view(-1, logits.size(-1)), targets.view(-1), sigma=getattr(conf.training, 'loss_sigma', 1.0))
            else:
                raise ValueError(f"Unknown loss function: {conf.training.loss_function}")
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, new_kv_cache

    def crop_block_size(self, block_size):
        """Surgically reduce the block size to `block_size`."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        # No wpe to crop (RoPE is used instead).
        # Crop the causal bias if any (for manual attention fallback).
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
            # Crop the RoPE precomputed cos/sin buffers
            if hasattr(block.attn, 'cos_cached'):
                block.attn.cos_cached = block.attn.cos_cached[:, :, :block_size, :]
                block.attn.sin_cached = block.attn.sin_cached[:, :, :block_size, :]
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # collect all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate into decay / no-decay groups.
        # Standard rule: all 2D+ params get weight decay, except biases and layernorms.
        # BUT we want to exclude the type embedding explicitly.
        decay_params = []
        nodecay_params = []

        for name, param in param_dict.items():
            if param.dim() >= 2 and 'type_emb' not in name:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        pLogging.info(logger_idx, "Optimizer info", {
            "num_decay_params": num_decay,
            "num_nodecay_params": num_nodecay
        })

        # AdamW fused if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        pLogging.info(logger_idx, "Optimizer info", {"use_fused_adamw": use_fused})
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        This is a simple and slow implementation. Requires the least memory and works across devices.
        Practically, this should only be used is we're sampling on the CPU.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            
            # If we hit EVENT_END token, we stop generating
            if idx_next[0][0] == dictionary.event_end_token:
                break

        return idx
    
    @torch.no_grad()
    def generate_batched_singleGPU(self, idx, max_new_tokens, temperature=1.0, top_k=None, batch_size=None):
        """
        Proper batched generation with KV‑cache.
        - idx: (batch, prompt_len) with right‑padding (padding_token on the right).
        - batch_size: max sequences to process in parallel on this GPU. If None, use all.
        """
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
            
        device = idx.device
        original_batch_size = idx.size(0)
        
        if batch_size is None:
            batch_size = original_batch_size

        # Output tensor: prompt + max_new_tokens
        output = torch.full(
            (original_batch_size, idx.size(1) + max_new_tokens),
            fill_value=dictionary.padding_token,
            dtype=torch.long,
            device=device,
        )
        output[:, :idx.size(1)] = idx

        # Process in mini‑batches to respect batch_size limit
        for start in range(0, original_batch_size, batch_size):
            end = min(start + batch_size, original_batch_size)
            cur_batch = output[start:end]                   # (bs, total_len)
            cur_prompt = idx[start:end]                     # (bs, prompt_len)

            # ---- initial pass: process the whole prompt ----
            # Compute real sequence lengths (assumes padding_token is on the right)
            seq_lens = (cur_prompt != dictionary.padding_token).sum(dim=1)   # (bs,)
            max_len = seq_lens.max().item()

            # Left‑pad to max_len
            padded_prompt = cur_prompt[:, :max_len]          # truncate to max_len
            # Build attention mask: shape (bs, 1, max_len, max_len)
            # True where token is not padding and position is ≤ current token (causal)
            not_padding = torch.arange(max_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)  # (bs, max_len)
            causal_mask = torch.tril(torch.ones(max_len, max_len, device=device, dtype=torch.bool))
            attn_mask = not_padding.unsqueeze(1).unsqueeze(2) & causal_mask.unsqueeze(0).unsqueeze(0)   # (bs, 1, max_len, max_len)

            # Position IDs: for each token, its real position in the non‑padded sequence
            position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(cur_batch.size(0), -1)

            # Forward the prompt
            logits, _, kv_cache = self.forward(padded_prompt, attention_mask=attn_mask)
            # logits shape: (bs, max_len, vocab)
            # Get logits of the last non‑padding token for each sequence
            last_logits = logits[torch.arange(cur_batch.size(0)), seq_lens - 1]  # (bs, vocab)

            # Sample first new token
            last_logits = last_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(last_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)   # (bs,)

            # Write first generated token to output
            cur_batch[torch.arange(cur_batch.size(0)), seq_lens] = next_tokens
            seq_lens += 1

            # Track finished sequences (event_end_token hit)
            unfinished = torch.ones(cur_batch.size(0), dtype=torch.bool, device=device)
            unfinished[next_tokens == dictionary.event_end_token] = False

            # ---- incremental loop ----
            for step in range(1, max_new_tokens):
                if not unfinished.any():
                    break
                # Only process unfinished sequences
                active = unfinished.nonzero(as_tuple=False).squeeze(1)

                # Current last token for each active sequence
                input_ids = cur_batch[active, seq_lens[active] - 1].unsqueeze(1)   # (num_active, 1)

                # Forward with the previous KV cache (only for active sequences)
                # We need to slice the KV cache to keep only active rows
                active_kv_cache = []
                for layer_kv in kv_cache:
                    k, v = layer_kv
                    # k,v shapes: (bs, nh, total_len, hs)
                    k_active = k[active]   # keep along batch dim
                    v_active = v[active]
                    active_kv_cache.append((k_active, v_active))

                logits, _, active_kv_cache = self.forward(
                    input_ids,
                    kv_cache=active_kv_cache,
                    attention_mask=None   # incremental: no mask needed (seq len 1)
                )
                kv_cache = active_kv_cache   # update for next step

                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                # Write tokens
                cur_batch[active, seq_lens[active]] = next_tokens
                seq_lens[active] += 1

                # Mark finished
                is_end = next_tokens == dictionary.event_end_token
                unfinished[active[is_end]] = False

        return output
        
    @torch.no_grad()
    def generate_batched_multiGPU(self, starters, max_new_tokens, temperature=1.0, top_k=None, batch_size=128):
        """
        Uses all available GPUs to generate samples in parallel.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU sampling requires CUDA devices.")
        
        num_devices = torch.cuda.device_count()
        devices = [f'cuda:{i}' for i in range(num_devices)]

        starter_splits = torch.chunk(starters, num_devices, dim=0)

        ctx = mp.get_context('spawn')
        return_queue = ctx.Queue()

        model_state_dict = self.state_dict()
        model_config = self.config

        processes = []
        for device_id, starters_chunk in enumerate(starter_splits):
            p = ctx.Process(
                target=batched_multiGPU_worker,
                args=(device_id, model_config, model_state_dict, starters_chunk, max_new_tokens, temperature, top_k, batch_size, return_queue)
            )
            p.start()
            processes.append(p)

        results = [None] * num_devices
        for _ in range(num_devices):
            device_id, generated = return_queue.get()
            results[device_id] = generated

        for p in processes:
            p.join()

        return torch.cat(results, dim=0)