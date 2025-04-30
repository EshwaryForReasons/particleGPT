"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

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
dictionary = Dictionary(script_dir / 'data' / conf.generic.preparation_name / 'dictionary.json')

logger_idx = -1
# Model will never be used on its own. It will be accessed by the training or generation script.
# Therefore, we inherit the logger from the accessing script.
def set_logger(in_logger_idx):
    global logger_idx
    logger_idx = in_logger_idx

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, kv_cache=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # if kv_cache is provided, concatenate with current k and v
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat((prev_k, k), dim=2)
            v = torch.cat((prev_v, v), dim=2)
        
        # save the current k and v for next step
        new_kv_cache = (k, v) if kv_cache is not None else None

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True if kv_cache is None else False
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if kv_cache is None:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
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

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None):
        attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    num_token_types: int = 7 # len(ETokenTypes)

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

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            type_emb = nn.Embedding(config.num_token_types, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
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

        # report number of parameters
        pLogging.info(logger_idx, "Model info", {"num_params": str(self.get_num_params() / 1e6) + "M" })
    
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
        return type_ids

    def get_token_type_ids_old(self, idx):
        # Map token id ranges to type ids
        type_ids = torch.zeros_like(idx)
        type_ids[(idx == 0)]                                                                                                               = 0 # ETokenTypes.PADDING.value
        type_ids[(idx >= dictionary.SPECIAL_TOKENS_OFFSET + 1) & (idx < dictionary.SPECIAL_TOKENS_OFFSET + dictionary.num_special_tokens)] = 1 # ETokenTypes.SPECIAL.value
        type_ids[(idx >= dictionary.PDGID_OFFSET)              & (idx < dictionary.PDGID_OFFSET + dictionary.num_particles)]               = 2 # ETokenTypes.PDGID.value
        type_ids[(idx >= dictionary.ENERGY_OFFSET)             & (idx < dictionary.ENERGY_OFFSET + len(dictionary.e_bins))]                = 3 # ETokenTypes.ENERGY.value
        type_ids[(idx >= dictionary.ETA_OFFSET)                & (idx < dictionary.ETA_OFFSET + len(dictionary.eta_bins))]                 = 4 # ETokenTypes.ETA.value
        type_ids[(idx >= dictionary.THETA_OFFSET)              & (idx < dictionary.THETA_OFFSET + len(dictionary.theta_bins))]             = 5 # ETokenTypes.THETA.value
        type_ids[(idx >= dictionary.PHI_OFFSET)                & (idx < dictionary.PHI_OFFSET + len(dictionary.phi_bins))]                 = 6 # ETokenTypes.PHI.value
        return type_ids

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, kv_cache=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        
        # Type embedding because it might help in our case
        type_ids = self.get_token_type_ids_old(idx)
        type_emb = self.transformer.type_emb(type_ids)

        x = self.transformer.drop(tok_emb + pos_emb + type_emb)
        
        new_kv_cache = []
        if kv_cache is None:
            kv_cache = [None] * len(self.transformer.h)
        
        for i, block in enumerate(self.transformer.h):
            x, new_block_kv_cache = block(x, kv_cache=kv_cache[i])
            new_kv_cache.append(new_block_kv_cache)
        
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=dictionary.padding_token)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, new_kv_cache

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        pLogging.info(logger_idx, "Optimizer info", {
            "num_decay_parameter_tensors": len(decay_params),
            "num_decay_params": num_decay_params,
            "num_non_decay_parameter_tensors": len(nodecay_params),
            "num_nodecay_params": num_nodecay_params
        })
        # Create AdamW optimizer and use the fused version if it is available
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
    def generate_batched_singleGPU(self, idx, max_new_tokens, temperature=1.0, top_k=None, batch_size=128):
        """
        Can be used standalone to sample on a single GPU. Also used by multiGPU implementation to sample on each GPU.
        """
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)

        device = idx.device
        original_batch_size = idx.size(0)

        padded_outputs = torch.full(
            (original_batch_size, idx.size(1) + max_new_tokens),
            fill_value=dictionary.padding_token,
            dtype=torch.long,
            device=device
        )
        padded_outputs[:, :idx.size(1)] = idx

        for batch_start in range(0, original_batch_size, batch_size):
            batch_end = min(batch_start + batch_size, original_batch_size)
            current_batch = padded_outputs[batch_start:batch_end]
            cur_batch_size = current_batch.size(0)

            unfinished = torch.ones(cur_batch_size, dtype=torch.bool, device=device)
            seq_lens = torch.full((cur_batch_size,), idx.size(1), dtype=torch.long, device=device)

            kv_cache = None

            for _ in range(max_new_tokens):
                if not unfinished.any():
                    break

                active_idx = unfinished.nonzero(as_tuple=False).squeeze(1)
                active_seq_lens = seq_lens[active_idx]

                # Get last max_len tokens
                active_sequences = torch.stack([current_batch[i, :l] for i, l in zip(active_idx, active_seq_lens)], dim=0)

                idx_cond = active_sequences if active_sequences.size(1) <= self.config.block_size else active_sequences[:, -self.config.block_size:]

                logits, _, kv_cache = self(idx_cond, kv_cache=kv_cache)

                # Sample next token
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                # Update
                current_batch[active_idx, seq_lens[active_idx]] = next_tokens
                seq_lens[active_idx] += 1

                # Check for end tokens
                is_end = next_tokens == dictionary.event_end_token
                unfinished[active_idx[is_end]] = False

        return padded_outputs
    
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