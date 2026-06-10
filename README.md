# particleGPT

![particleGPT logo](docs/logo.jpg)

A GPT-based generative model for particle collision events, built on [nanoGPT](https://github.com/karpathy/nanoGPT) and adapted for high-energy physics simulation data.

---

## What Is This?

particleGPT trains a decoder-only Transformer to learn the joint distribution of particle properties inside collision events. Given a partial event (a few "starter" tokens), the model completes the event autoregressively — generating particle identities, momenta, angles, and energies token by token.

The pipeline covers the full workflow:

```
Raw Geant4 CSV  →  Tokenize  →  Train GPT  →  Sample  →  Untokenize  →  Analyze
```

The primary use case is **fast generative simulation**: replacing or augmenting slow Monte Carlo simulators (e.g. Geant4) with a learned surrogate that produces statistically similar events orders of magnitude faster.

---

## Repository Layout

```
particleGPT/
├── train.py                  # Training entrypoint (single-GPU, DDP, multi-node)
├── sample.py                 # Generation entrypoint (multi-GPU parallel sampling)
├── configurator.py           # Config dataclasses; loaded from JSON at startup
├── dictionary.py             # Vocabulary: bins, offsets, token↔value conversion
├── preparation.py            # Data-split config and tokenized-metadata parsing
├── data_manager.py           # Raw dataset loading utilities
├── paths.py                  # Centralised path resolution
├── pLogging.py               # Structured logging wrapper
├── pUtil.py                  # Miscellaneous utilities
│
├── particleGPT/
│   ├── model.py              # GPT model (nanoGPT core + particle-physics extensions)
│   ├── tokenizer.py          # Raw event → integer token sequence
│   ├── untokenizer.py        # Token sequence → reconstructed event
│   └── dataset.py            # Dataset helpers for the particleGPT module
│
├── analysis/
│   ├── analyzer.py           # Analysis orchestrator (loads samples, calls metrics/plots/tables)
│   ├── metrics.py            # Physics metrics: W1, FPD, KPD, multiplicity, etc.
│   ├── plotting.py           # Histogram / distribution plots (generated vs. real)
│   └── tables.py             # LaTeX / Markdown results tables
│
├── config/
│   ├── models/               # Per-model JSON configs (architecture + training hyperparams)
│   └── preparations/         # Data-split JSON configs (train/val/test sequence ranges)
│
├── data/
│   └── tokenized/            # Binary `.bin` files and `tokenized_metadata.json`
│
├── generated_samples/        # Output of sample.py; one CSV per sampling run
│   └── <model_name>/
│       └── sampling_<N>/
│           ├── samples.csv
│           └── sampling_metadata.json
│
├── trained_models/           # Checkpoints written by train.py
│   └── <model_name>/
│       ├── ckpt.pt           # Best checkpoint (saved when val loss improves)
│       ├── ckpt_running_N.pt # Periodic running checkpoints
│       └── ckpt_epoch_N.pt   # End-of-epoch checkpoints
│
└── docs/                     # ← you are here
```

---

## What Kind of Data Does This Expect?

particleGPT is designed for **event-based particle physics datasets**, specifically output from detectors/simulators like **Geant4**. Each event is an unordered set of secondary particles produced by a collision or interaction.

A raw event looks roughly like:

```
<event_start>
  <particle_start> PDGID  material  E  eta  phi  pt <particle_end>
  <particle_start> PDGID  material  E  eta  phi  pt <particle_end>
  ...
<event_end>
```

Each continuous-valued feature (E, η, φ, pT, pX, pY, pZ, θ) is **discretized into bins** before being converted to a single integer token. PDGID and material tokens are assigned directly via lookup. All token types are packed into one flat vocabulary with non-overlapping integer ranges — no special embedding tricks are needed.

The tokenized dataset is stored as a memory-mapped binary (`.bin`) flat array of integers, alongside a `tokenized_metadata.json` that records dtype, vocab size, sequence length, and total token counts.

---

## Quickstart

### 1. Prepare data

```bash
# Tokenize your raw dataset and write the binary + metadata files
python -m particleGPT.tokenizer  config/models/my_model.json
```

### 2. Train

```bash
# Single GPU
python train.py config/models/my_model.json

# Single-node 4-GPU DDP
torchrun --standalone --nproc_per_node=4 train.py config/models/my_model.json

# Multi-node (see README Notes section for full SLURM commands)
```

### 3. Sample

```bash
python sample.py config/models/my_model.json
```

Generated events are written to `generated_samples/<model_name>/sampling_<N>/samples.csv`.

### 4. Analyze

```bash
python analysis/analyzer.py config/models/my_model.json
```

---

## Configuration

Every script loads its settings from a single JSON file passed as the first CLI argument. The JSON has three top-level namespaces:

```json
{
  "model_name": "my_model",
  "dataset": "my_dataset.csv",
  "preparation_config_file": "config/preparations/my_prep.json",
  "data_mode": "particle",
  "mlp_type": "swiglu",
  "training_config": {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "batch_size": 32,
    "block_size": 512,
    "learning_rate": 6e-4,
    "max_iters": 100000,
    ...
  },
  "sampling_config": {
    "batch_size": 128,
    "temperature": 0.8,
    "top_k": 200,
    ...
  }
}
```

See [`configurator.py`](configurator.md) for every available field and its default value.

---

## Model Architecture

The model is a standard decoder-only Transformer (GPT) with several optional particle-physics-specific extensions controlled by the config:

| Feature | Config key | Default |
|---|---|---|
| MLP variant | `mlp_type` | `"swiglu"` |
| MLP hidden ratio | `mlp_ratio` | `4.0` |
| QK normalisation | `qk_norm` | `true` |
| Embedding layer norm | `embd_norm_type` | `"none"` |
| Particle index embeddings | `use_particle_index_embd` | `false` |
| Bin-value embeddings | `use_bin_value_embd` | `false` |
| Grouped-query attention | `n_kv_heads` | `0` (= standard MHA) |

---

## Training Details

- **Optimiser**: AdamW with gradient clipping
- **LR schedule**: Cosine annealing with linear warmup (also supports warm restarts)
- **Mixed precision**: bfloat16 (where hardware supports it) or float16
- **Compilation**: `torch.compile` enabled by default
- **Checkpointing**: best-val-loss checkpoint + periodic running/epoch checkpoints; fully resumable
- **DDP**: torchrun-based; `gradient_accumulation_steps` is automatically divided by `world_size`
- **Epoch mode**: blocks can be constrained to not cross event boundaries (`use_self_contained_blocks`)

---

## Sampling Details

- Starters come from the test split (first `N` tokens of each test sequence, default 5)
- Generation is batched across all available GPUs using `torch.multiprocessing.spawn`
- Each GPU writes a shard CSV; shards are concatenated in rank order into a final `samples.csv`
- Each row in `samples.csv` is a space-separated list of integer token IDs (including starters)

---

## Analysis & Metrics

After sampling, `analyzer.py` untokenizes the CSV, reconstructs particle four-vectors, and computes:

- **Wasserstein-1 distances** per feature (energy, η, φ, pT, multiplicity, …)
- **FPD** (Fréchet Physics Distance) — physics analogue of FID
- **KPD** (Kernel Physics Distance)
- Per-feature **histogram overlays** (generated vs. real)
- Summary **results tables** (LaTeX-ready)

---

## Example Distributions

<!-- Add your example plots here -->

| Feature | Distribution Plot |
|---|---|
| Energy | *(insert plot)* |
| η | *(insert plot)* |
| φ | *(insert plot)* |
| pT | *(insert plot)* |
| Multiplicity | *(insert plot)* |

---

## Dependencies

| Library | Purpose |
|---|---|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Base Transformer architecture |
| [scikit-hep/particle](https://github.com/scikit-hep/particle) | PDGID ↔ particle name conversion |
| [scikit-hep/vector](https://github.com/scikit-hep/vector) | Four-vector arithmetic |
| [JetNet](https://github.com/jet-net/JetNet) | FPD, KPD, W1 physics metrics |
| PyTorch ≥ 2.0 | Model, training, DDP |
| NumPy, SciPy | Numerical utilities, bin spacing |

---

## Running on SLURM / HPC

```bash
# Interactive single-node 4-GPU job
srun -C "gpu" -q interactive -N 1 -G 4 -c 32 -t 4:00:00 -A <account> --pty /bin/bash -l
torchrun --standalone --nproc_per_node=4 train.py config/models/my_model.json

# Multi-node (4 nodes × 4 GPUs = 16 GPUs)
salloc -C gpu -q interactive -N 4 -G 16 -c 32 -t 4:00:00 -A <account>
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
srun -N4 --ntasks-per-node=1 -c32 --gpus-per-task=4 --gpu-bind=none --cpu-bind=none \
  bash -c 'torchrun --nnodes=4 --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train.py config/models/my_model.json'
```

---

## Documentation Index

| File | Doc |
|---|---|
| `train.py` | [train.md](train.md) |
| `sample.py` | [sample.md](sample.md) |
| `configurator.py` | [configurator.md](configurator.md) |
| `dictionary.py` | [dictionary.md](dictionary.md) |
| `particleGPT/model.py` | [model.md](model.md) |
| `particleGPT/tokenizer.py` | [tokenizer.md](tokenizer.md) |
| `particleGPT/untokenizer.py` | [untokenizer.md](untokenizer.md) |
| `particleGPT/dataset.py` | [dataset.md](dataset.md) |
| `analysis/analyzer.py` | [analyzer.md](analyzer.md) |
| `analysis/metrics.py` | [metrics.md](metrics.md) |
| `analysis/plotting.py` | [plotting.md](plotting.md) |
| `analysis/tables.py` | [tables.md](tables.md) |
