"""
particleGPT sampling entrypoint.

This script samples from a trained checkpoint using the current particleGPT data
pipeline:
  - one flat tokenized binary dataset
  - one tokenized metadata JSON file
  - one preparation JSON file that selects train/validation/test ranges

Sampling behavior
-----------------
- Starters come from test_bin.
- The starter for each generated event is the first `sampling.starter_tokens`
  tokens of each selected test sequence. The default is 5.
- By default, sampling uses the full configured test_bin. Set
  `sampling.max_test_sequences` to cap this for quick testing, for example 10_000.
- Generation is batched. Each model call receives a 2D tensor with shape
  (batch_size, starter_tokens).
- All visible CUDA GPUs are used by default. Single-GPU sampling can be forced
  with sampling.single_gpu=True, sampling.num_gpus=1, sampling.use_all_gpus=False,
  sampling.device="cuda:0", or PARTICLEGPT_SAMPLE_NUM_GPUS=1.
- Each worker writes one temporary CSV shard. Rank-order concatenation produces
  one final CSV:
      paths.PROJECT_DIR/generated_samples/<model_name>/sampling_<sample_idx>/samples.csv

The final CSV contains one generated event per row. Each row is a comma-separated
list of token ids and includes the starter tokens at the beginning of the row.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path
from pydantic.dataclasses import dataclass
import warnings

import numpy as np
import torch
import torch.multiprocessing as mp

import paths
import pUtil
import particleGPT.configurator as conf
from particleGPT.model import GPT, GPTConfig
from train import clean_state_dict_keys
from particleGPT.preparation import ESplitTypes, DataloaderSplitConfig, TokenizedMetadataConfig

FINAL_CSV_NAME = "samples.csv"
SHARD_SUFFIX = ".csv.part"

@dataclass(frozen=True, slots=True)
class SamplingJob:
    """Used for passing data to different processes for multi-GPU sampling"""
    checkpoint_filepath: Path
    device_names: tuple[str, ...]
    batch_size: int
    num_starter_tokens: int
    max_new_tokens: int
    temperature: float
    top_k: int
    dtype_name: str
    seed: int
    compile: bool
    log_interval: int
    require_batch_generate: bool
    tokenized_data_filepath: Path
    tokenized_dtype: str
    sequence_length: int
    split_start_token_idx: int
    num_sample_sequences: int
    event_start_token: int


def resolve_device_names(num_sample_sequences: int) -> tuple[str, ...]:
    """
    Resolve which devices sampling should use.

    All visible CUDA GPUs are used by default. Single-GPU mode can be requested
    through config or PARTICLEGPT_SAMPLE_NUM_GPUS=1.
    """
    requested_device = str(conf.sampling.device)
    if requested_device == "cpu":
        return ("cpu",)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for sampling, but torch.cuda.is_available() is False.")
    if requested_device.startswith("cuda:"):
        return (requested_device,)
    if not torch.cuda.is_available():
        return ("cpu",)

    cuda_count = torch.cuda.device_count()
    gpu_ids = list(range(cuda_count))
    
    env_num_gpus = os.environ.get("PARTICLEGPT_SAMPLE_NUM_GPUS", None)
    if env_num_gpus is None:
        requested_num_gpus = None
    elif isinstance(env_num_gpus, str) and env_num_gpus.strip() == "":
        requested_num_gpus = None
    else:
        requested_num_gpus = int(env_num_gpus)
    
    if conf.sampling.force_single_gpu:
        requested_num_gpus = 1
    elif requested_num_gpus is None:
        requested_num_gpus = len(gpu_ids)

    requested_num_gpus = max(1, min(int(requested_num_gpus), len(gpu_ids), num_sample_sequences))
    return tuple(f"cuda:{idx}" for idx in gpu_ids[:requested_num_gpus])

def resolve_sample_idx(generated_samples_dir: Path) -> int:
    """
    Resolve the numeric sample index used in sampling_<sample_idx>.

    If sampling.sampling_idx_override is set, that value is used.
    Otherwise, the next numeric index after existing sampling_N directories is used.
    """
    configured_idx = conf.sampling.sampling_idx_override
    if configured_idx is not None:
        return int(configured_idx)

    max_idx = -1
    if generated_samples_dir.exists():
        for path in generated_samples_dir.iterdir():
            if not path.is_dir():
                continue
            match = re.fullmatch(r"sampling_(\d+)", path.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
    
    return max_idx + 1

def autocast_context(device_type: str, dtype: torch.dtype):
    """Return the correct inference autocast context for the selected device/dtype."""
    if device_type == "cuda" and dtype == torch.float16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device_type == "cuda" and dtype == torch.bfloat16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

def shard_path(output_dir: Path, rank: int) -> Path:
    """Return the temporary CSV shard path for one worker."""
    return output_dir / f"worker_{rank:03d}{SHARD_SUFFIX}"

def generate_batch(model: torch.nn.Module, idx: torch.Tensor, max_new_tokens: int, temperature: float, top_k: int | None, require_batch_generate: bool) -> torch.Tensor:
    """
    Run batched generation through the model's batch-capable generation path.

    idx must have shape (batch_size, starter_tokens). If the model exposes an
    explicit batch-generation method, this uses it. Otherwise, it falls back to
    model.generate(idx, ...), which is batch-capable for nanoGPT-style models
    when idx is 2D.
    """
    if idx.ndim != 2:
        raise ValueError(f"Expected batched idx with shape (B, T), got shape={tuple(idx.shape)}.")

    for method_name in ("generate_batch", "batch_generate", "generate_batched"):
        method = getattr(model, method_name, None)
        if callable(method):
            return method(idx, max_new_tokens, temperature=temperature, top_k=top_k)

    if require_batch_generate:
        raise AttributeError(
            "sampling.require_batch_generate=True, but the model does not expose "
            "generate_batch, batch_generate, or generate_batched."
        )
    
    if not hasattr(model, "generate"):
        raise AttributeError("Model does not expose generate(...), so sampling cannot continue.")

    return model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)

def extract_starters(job: SamplingJob) -> np.ndarray:
    data = np.memmap(job.tokenized_data_filepath, dtype=np.dtype(job.tokenized_dtype), mode="r")
    num_split_tokens = job.num_sample_sequences * job.sequence_length
    test_sequences = data[job.split_start_token_idx:job.split_start_token_idx + num_split_tokens]
    # Useful for packed event streams, extracts all EVENT_START token idxs
    event_starts = np.flatnonzero(test_sequences == job.event_start_token)
    # Keep only starts that have enough proceeding tokens for a full prompt.
    event_starts = event_starts[event_starts + job.num_starter_tokens <= len(test_sequences)]
    # Build a 2D array of shape (num_events_found, job.num_starter_tokens)
    prompt_indices = event_starts[:, None] + np.arange(job.num_starter_tokens)
    prompts = test_sequences[prompt_indices]
    return prompts

def sampling_worker(rank: int, job: SamplingJob, output_dir: Path) -> None:
    """
    Generate one contiguous shard of extracted starter prompts on one worker/device.
    Each worker owns a contiguous slice of the starter prompts extracted from this job's test-token window.
    """
    device = job.device_names[rank]
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))

    worker_seed = int(job.seed) + rank
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    world_size = len(job.device_names)
    
    # Get prompts for this worker's assigned slice of test sequences.
    prompts = extract_starters(job)
    prompts = prompts.astype(np.int64, copy=False)
    start_prompt_idx = (len(prompts) * rank) // world_size
    end_prompt_idx = (len(prompts) * (rank + 1)) // world_size
    prompts = prompts[start_prompt_idx:end_prompt_idx]

    # ===== Load model for sampling =====
    # Load the checkpoint model onto one sampling worker device
    
    checkpoint = torch.load(job.checkpoint_filepath, map_location="cpu", weights_only=False)
    model_args = dict(checkpoint["model_args"])
    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(clean_state_dict_keys(checkpoint["model"]))
    del checkpoint

    model.to(device)
    model.eval()
    if job.compile:
        model = torch.compile(model, mode="default", dynamic=False, fullgraph=False)
        
    # ===== Get datatype from config =====    
    # Convert a config dtype string into the torch dtype used for autocast
    
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if job.dtype_name not in dtype_map:
        raise ValueError(f"Unknown dtype {job.dtype_name!r}; expected one of {sorted(dtype_map)}")
    dtype = dtype_map[job.dtype_name]

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    output_path = shard_path(output_dir, rank)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    t0 = time.time()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        with torch.no_grad():
            for batch_start in range(0, len(prompts), int(job.batch_size)):
                batch_end = min(batch_start + int(job.batch_size), len(prompts))
                starter_batch = prompts[batch_start:batch_end]
                idx = torch.from_numpy(starter_batch).to(device, non_blocking=True)

                with autocast_context(device_type, dtype):
                    generated = generate_batch(
                        model,
                        idx,
                        max_new_tokens=int(job.max_new_tokens),
                        temperature=float(job.temperature),
                        top_k=job.top_k,
                        require_batch_generate=bool(job.require_batch_generate),
                    )

                generated_np = generated.detach().cpu().numpy().astype(np.int64, copy=False)
                for row in generated_np:
                    csv_row = " ".join(str(int(token)) for token in row) + "\n"
                    handle.write(csv_row)
                written += int(generated_np.shape[0])

                if job.log_interval > 0 and written % job.log_interval == 0:
                    elapsed = max(time.time() - t0, 1e-12)
                    print(
                        f"worker {rank}: wrote {written:,} / {end_prompt_idx - start_prompt_idx:,} events "
                        f"({written / elapsed:.2f} events/s)",
                        flush=True,
                    )

    print(
        f"worker {rank}: finished {written:,} events from test sequences "
        f"[{start_prompt_idx:,}, {end_prompt_idx:,}) -> {output_path}",
        flush=True,
    )
    
def write_metadata(
    job: SamplingJob,
    output_dir: Path,
    output_filepath: Path,
    elapsed_time,
    sample_idx: int,
    dls_conf: DataloaderSplitConfig
):
    num_gpus_used = sum(1 for device_name in job.device_names if device_name.startswith("cuda"))
    output_metadata_filepath = output_dir / "sampling_metadata.json"
    metadata = {
        "checkpoint_filepath": paths.project_relative_path(job.checkpoint_filepath),
        "output_dir": paths.project_relative_path(output_dir),
        "output_filepath":  paths.project_relative_path(output_filepath),
        "output_metadata_filepath": paths.project_relative_path(output_metadata_filepath),
        "sampling_elapsed_seconds": elapsed_time,
        "model_name": conf.generic.model_name,
        "config_filepath": conf.generic.config_file_path,
        "sample_idx": sample_idx,
        "num_gpus_used": num_gpus_used,
        "num_devices_used": len(job.device_names),
        "device_names": tuple(job.device_names),
        "batch_size": job.batch_size,
        "num_starter_tokens": job.num_starter_tokens,
        "max_new_tokens": job.max_new_tokens,
        "temperature": job.temperature,
        "top_k": job.top_k,
        "seed": job.seed,
        "compile": job.compile,
        "require_batch_generate": job.require_batch_generate,
        "tokenized_data_filepath": paths.project_relative_path(job.tokenized_data_filepath),
        "tokenized_metadata_filepath": paths.project_relative_path(dls_conf.tokenized_metadata_filepath),
        "config_filepath": paths.project_relative_path(conf.generic.config_file_path),
        "tokenized_dtype": job.tokenized_dtype,
        "split": "test_bin",
        "sequence_length": job.sequence_length,
        "split_start_token_idx": job.split_start_token_idx,
        "num_sample_sequences": job.num_sample_sequences,
    }
    with output_metadata_filepath.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    return metadata

def main() -> None:
    """Sample generated events from the first configured subset of test_bin sequences."""
    if conf.generic.preparation_config_file is None:
        raise ValueError("preparation_config_file in configuration cannot be None!")

    model_output_dir = paths.PROJECT_DIR / "trained_models" / conf.generic.model_name
    b = model_output_dir / "ckpt.pt"
    
    preparation_config_file = Path(conf.generic.preparation_config_file)
    dls_conf = DataloaderSplitConfig(ESplitTypes.TEST, preparation_config_file)
    
    checkpoint = torch.load(b, map_location="cpu", weights_only=False)
    model_args = dict(checkpoint["model_args"])
    del checkpoint
    
    # ===== Validate all values derived from model and provided in config =====

    # @TODO: this should be configured somehow. Maybe the tokenizer class can be used to calculate this?
    num_starter_tokens = 5
    if num_starter_tokens <= 0:
        raise ValueError(f"starter_tokens must be positive, got {num_starter_tokens}.")
    if num_starter_tokens > dls_conf.tmd_conf.sequence_length:
        raise ValueError(f"starter_tokens={num_starter_tokens}, but sequence_length={dls_conf.tmd_conf.sequence_length}.")

    if int(model_args["vocab_size"]) != int(dls_conf.tmd_conf.vocab_size):
        raise ValueError(f"Checkpoint vocab_size={model_args['vocab_size']}, but tokenized metadata vocab_size={dls_conf.tmd_conf.vocab_size}.")
    if num_starter_tokens > int(model_args["block_size"]):
        raise ValueError(f"starter_tokens={num_starter_tokens}, but checkpoint block_size={model_args['block_size']}.")

    # For each sequence, Number of tokens to generate after the starter tokens
    max_new_tokens = conf.sampling.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = int(dls_conf.tmd_conf.sequence_length) - num_starter_tokens
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}.")
    if max_new_tokens > model_args["block_size"]:
        max_new_tokens = min(max_new_tokens, model_args["block_size"])
        warnings.warn(
            f"max_new_tokens={max_new_tokens} is greater than block_size={model_args['block_size']}. "
            f"Clamping max_new_tokens to block_size, which is {model_args['block_size']}.",
            RuntimeWarning
        )

    # Number of sequences to generate
    num_sample_sequences = conf.sampling.num_sample_sequences
    if num_sample_sequences is None:
        num_sample_sequences = dls_conf.num_sequences
    if num_sample_sequences <= 0:
        raise ValueError(f"max_test_sequences must be positive when configured, got {num_sample_sequences}.")
    if num_sample_sequences > dls_conf.num_sequences:
        num_sample_sequences = min(dls_conf.num_sequences, num_sample_sequences)
        warnings.warn(
            f"num_sample_sequences={num_sample_sequences} is greater than sequences avaliable in the dataset TEST split "
            f"sequences available={dls_conf.num_sequences}. Clamping num_sample_sequences to sequences available, "
            f"which is {dls_conf.num_sequences}.",
            RuntimeWarning
        )
        
    if conf.sampling.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {conf.sampling.batch_size}.")
    
    # ===== Prepare directory =====
    
    generated_samples_dir = paths.PROJECT_DIR / "generated_samples" / conf.generic.model_name
    sample_idx = resolve_sample_idx(generated_samples_dir)
    output_dir = generated_samples_dir / f"sampling_{sample_idx}"
    output_csv_name = FINAL_CSV_NAME

    # Clean temp
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob(f"*{SHARD_SUFFIX}"):
        path.unlink()
    for path in output_dir.glob("*.tmp"):
        path.unlink()
        
    # ===== Do sampling =====
    
    device_names = resolve_device_names(num_sample_sequences)

    top_k = None if conf.sampling.top_k is None or conf.sampling.top_k <= 0 else conf.sampling.top_k
    
    # Create job, required for multi-GPU sampling
    dictionary = pUtil.get_dictionary(conf.generic.preparation_config_file)
    raw_start, raw_end = dls_conf.get_raw_tokens_range()
    job = SamplingJob(
        checkpoint_filepath         = b,
        device_names                = tuple(device_names),
        batch_size                  = conf.sampling.batch_size,
        num_starter_tokens          = num_starter_tokens,
        max_new_tokens              = max_new_tokens,
        temperature                 = float(conf.sampling.temperature),
        top_k                       = top_k,
        dtype_name                  = conf.sampling.dtype,
        seed                        = conf.sampling.seed,
        compile                     = conf.sampling.compile,
        log_interval                = conf.sampling.log_interval,
        output_csv_name             = output_csv_name,
        require_batch_generate      = False, # @TODO: make configurable, figure out what it does first
        tokenized_data_filepath     = str(dls_conf.tmd_conf.tokenized_data_filepath),
        tokenized_dtype             = np.dtype(dls_conf.tmd_conf.dtype).name,
        sequence_length             = int(dls_conf.tmd_conf.sequence_length),
        split_start_token_idx       = int(raw_start),
        split_end_token_idx         = int(raw_end),
        num_sample_sequences        = num_sample_sequences,
        event_start_token           = int(dictionary.event_start_token)
    )

    print(f"Sampling from checkpoint: {b}")
    print(f"Sampling output directory: {output_dir}")
    print(f"Sampling sequences: {num_sample_sequences:,} / {dls_conf.num_sequences:,}")
    print(f"Starter tokens per event: {num_starter_tokens}")
    print(f"New tokens per event: {max_new_tokens}")
    print(f"Batch size per worker: {conf.sampling.batch_size}")
    print(f"Worker devices: {', '.join(device_names)}")

    # Time should only measure sampling time, no point of measuring IO time.
    # @NOTE: this technically does have IO time since we write the shards to disk.
    # @TODO: maybe find a more sophisticated way to measure only the sampling time.
    sample_t0 = time.time()

    # Main sampling
    world_size = len(device_names)
    if world_size == 1:
        sampling_worker(0, job, output_dir)
    else:
        mp.spawn(sampling_worker, args=(job, output_dir), nprocs=world_size, join=True)

    sampling_elapsed_seconds = time.time() - sample_t0
    
    # Concatenate shards
    final_path = output_dir / output_csv_name
    tmp_final_path = output_dir / f"{output_csv_name}.tmp"
    with tmp_final_path.open("w", encoding="utf-8", newline="") as out_handle:
        for rank in range(world_size):
            part_path = shard_path(output_dir, rank)
            if not part_path.exists():
                raise FileNotFoundError(f"Expected worker shard does not exist: {part_path}")
            with part_path.open("r", encoding="utf-8") as in_handle:
                while True:
                    chunk = in_handle.read(1024 * 1024)
                    if not chunk:
                        break
                    out_handle.write(chunk)
    os.replace(tmp_final_path, final_path)

    if not conf.sampling.keep_shards:
        for rank in range(world_size):
            part_path = shard_path(output_dir, rank)
            if part_path.exists():
                part_path.unlink()
    
    # ===== Finalize =====
    
    metadata = write_metadata(
        job=job,
        output_dir=output_dir,
        output_filepath=final_path,
        elapsed_time=sampling_elapsed_seconds,
        sample_idx=sample_idx,
        dls_conf=dls_conf
    )

    print(f"Finished. Final generated samples CSV: {final_path}")
    print(f"Sampling took {sampling_elapsed_seconds:.2f} seconds using {metadata['num_gpus_used']} GPU(s).")

if __name__ == "__main__":
    main()
