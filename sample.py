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
from typing import Any, Optional

import numpy as np
import torch
import torch.multiprocessing as mp

import paths
import particleGPT.configurator as conf
from particleGPT.model import GPT, GPTConfig
from train import clean_state_dict_keys
from particleGPT.preparation import ESplitTypes, DataloaderSplitConfig, TokenizedMetadataConfig

FINAL_CSV_NAME = "samples.csv"
SHARD_SUFFIX = ".csv.part"

def resolve_device_names(total_starters: int) -> tuple[str, ...]:
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

    requested_num_gpus = max(1, min(int(requested_num_gpus), len(gpu_ids), total_starters))
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

def dtype_from_config(dtype_name: str) -> torch.dtype:
    """Convert a config dtype string into the torch dtype used for autocast."""
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    
    if dtype_name not in dtype_map:
        raise ValueError(f"Unknown dtype {dtype_name!r}; expected one of {sorted(dtype_map)}")
    
    return dtype_map[dtype_name]

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

def generate_batch(model: torch.nn.Module, idx: torch.Tensor, max_new_tokens: int, temperature: float, top_k: Optional[int], require_batch_generate: bool) -> torch.Tensor:
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

def load_model_for_sampling(checkpoint_path: str, device: str, compile_model: bool) -> torch.nn.Module:
    """Load the checkpoint model onto one sampling worker device."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = dict(checkpoint["model_args"])
    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(clean_state_dict_keys(checkpoint["model"]))
    del checkpoint

    model.to(device)
    model.eval()
    if compile_model:
        model = torch.compile(model, mode="default", dynamic=False, fullgraph=False)
    return model

def sampling_worker(rank: int, job: dict[str, Any]) -> None:
    """
    Generate one contiguous shard of test starters on one worker/device.

    This must remain a top-level function because torch.multiprocessing.spawn
    needs a pickleable entrypoint. Each worker owns a contiguous slice of the
    first `job["total_starters"]` test sequences.
    """
    device = job["device_names"][rank]
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))

    worker_seed = int(job["seed"]) + rank
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_starters = int(job["total_starters"])
    world_size = len(job["device_names"])
    start_sequence_idx = (total_starters * rank) // world_size
    end_sequence_idx = (total_starters * (rank + 1)) // world_size

    data = np.memmap(job["tokenized_data_filepath"], dtype=np.dtype(job["tokenized_dtype"]), mode="r")
    split_token_start = int(job["split_start_token_idx"])
    sequence_length = int(job["sequence_length"])
    starter_tokens = int(job["starter_tokens"])
    split_tokens = total_starters * sequence_length
    test_sequences = data[split_token_start:split_token_start + split_tokens].reshape(total_starters, sequence_length)

    model = load_model_for_sampling(job["checkpoint_path"], device, bool(job["compile"]))
    dtype = dtype_from_config(job["dtype_name"])
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    output_path = shard_path(Path(job["output_dir"]), rank)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    t0 = time.time()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        with torch.no_grad():
            for batch_start in range(start_sequence_idx, end_sequence_idx, int(job["batch_size"])):
                batch_end = min(batch_start + int(job["batch_size"]), end_sequence_idx)
                starter_batch = test_sequences[batch_start:batch_end, :starter_tokens].astype(np.int64, copy=True)
                idx = torch.from_numpy(starter_batch).to(device, non_blocking=True)

                with autocast_context(device_type, dtype):
                    generated = generate_batch(
                        model,
                        idx,
                        max_new_tokens=int(job["max_new_tokens"]),
                        temperature=float(job["temperature"]),
                        top_k=job["top_k"],
                        require_batch_generate=bool(job["require_batch_generate"]),
                    )

                generated_np = generated.detach().cpu().numpy().astype(np.int64, copy=False)
                for row in generated_np:
                    csv_row = " ".join(str(int(token)) for token in row) + "\n"
                    handle.write(csv_row)
                written += int(generated_np.shape[0])

                if int(job["log_interval"]) > 0 and written % int(job["log_interval"]) == 0:
                    elapsed = max(time.time() - t0, 1e-12)
                    print(
                        f"worker {rank}: wrote {written:,} / {end_sequence_idx - start_sequence_idx:,} events "
                        f"({written / elapsed:.2f} events/s)",
                        flush=True,
                    )

    print(
        f"worker {rank}: finished {written:,} events from test sequences "
        f"[{start_sequence_idx:,}, {end_sequence_idx:,}) -> {output_path}",
        flush=True,
    )

def main() -> None:
    """Sample generated events from the first configured subset of test_bin sequences."""
    if conf.generic.preparation_config_file is None:
        raise ValueError("preparation_config_file in configuration cannot be None!")

    model_output_dir = paths.PROJECT_DIR / "trained_models" / conf.generic.model_name
    checkpoint_path = model_output_dir / "ckpt.pt"
    
    preparation_config_file = Path(conf.generic.preparation_config_file)
    dls_conf = DataloaderSplitConfig(ESplitTypes.TEST, preparation_config_file)
    if not dls_conf.verify():
        raise RuntimeError("Failure when verifying test split config. Ensure all required arguments exist.")

    tmd_conf = TokenizedMetadataConfig(Path(dls_conf.tokenized_metadata_filepath))
    if not tmd_conf.verify():
        raise RuntimeError("Failure when verifying tokenized metadata config. Ensure all required arguments exist.")

    file_bytes = tmd_conf.tokenized_data_filepath.stat().st_size
    dtype_bytes = np.dtype(tmd_conf.dtype).itemsize
    if file_bytes % dtype_bytes != 0:
        raise ValueError(
            f"Size of tokenized data file ({file_bytes} bytes) is not divisible by dtype size {dtype_bytes}."
        )

    data_total_tokens = file_bytes // dtype_bytes
    if data_total_tokens != tmd_conf.total_tokens:
        raise ValueError(
            f"Tokenized data contains {data_total_tokens:,} tokens, but metadata expected {tmd_conf.total_tokens:,}."
        )

    raw_split_tokens = dls_conf.num_sequences * tmd_conf.sequence_length
    if dls_conf.from_end:
        raw_end = (tmd_conf.num_full_sequences - dls_conf.skip_sequences) * tmd_conf.sequence_length
        raw_start = raw_end - raw_split_tokens
    else:
        raw_start = dls_conf.skip_sequences * tmd_conf.sequence_length
        raw_end = raw_start + raw_split_tokens

    if raw_start < 0 or raw_end > tmd_conf.total_tokens or raw_start >= raw_end:
        raise ValueError(
            f"Invalid test split range: raw_start={raw_start}, raw_end={raw_end}, total_tokens={tmd_conf.total_tokens}."
        )

    # @TODO: this should be configured somehow. Maybe the tokenizer class can be used to calculate this?
    starter_tokens = 5
    if starter_tokens <= 0:
        raise ValueError(f"starter_tokens must be positive, got {starter_tokens}.")
    if starter_tokens > tmd_conf.sequence_length:
        raise ValueError(f"starter_tokens={starter_tokens}, but sequence_length={tmd_conf.sequence_length}.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = dict(checkpoint["model_args"])
    del checkpoint

    if int(model_args["vocab_size"]) != int(tmd_conf.vocab_size):
        raise ValueError(
            f"Checkpoint vocab_size={model_args['vocab_size']}, but tokenized metadata vocab_size={tmd_conf.vocab_size}."
        )
    if starter_tokens > int(model_args["block_size"]):
        raise ValueError(
            f"starter_tokens={starter_tokens}, but checkpoint block_size={model_args['block_size']}."
        )

    max_new_tokens = conf.sampling.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = int(tmd_conf.sequence_length) - starter_tokens
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}.")

    max_test_sequences = conf.sampling.max_test_sequences
    if max_test_sequences is None:
        total_starters = int(dls_conf.num_sequences)
        max_test_sequences_metadata = None
    else:
        if max_test_sequences <= 0:
            raise ValueError(f"max_test_sequences must be positive when configured, got {max_test_sequences}.")
        total_starters = min(int(dls_conf.num_sequences), max_test_sequences)
        max_test_sequences_metadata = max_test_sequences

    device_names = resolve_device_names(total_starters)
    batch_size = conf.sampling.batch_size
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    top_k_value = conf.sampling.top_k
    top_k = None if top_k_value is None or int(top_k_value) <= 0 else int(top_k_value)
    generated_samples_dir = paths.PROJECT_DIR / "generated_samples" / conf.generic.model_name
    sample_idx = resolve_sample_idx(generated_samples_dir)
    output_dir = generated_samples_dir / f"sampling_{sample_idx}"
    output_csv_name = FINAL_CSV_NAME

    sample_t0 = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob(f"*{SHARD_SUFFIX}"):
        path.unlink()
    for path in output_dir.glob("*.tmp"):
        path.unlink()

    require_batch_generate = False # @TODO: make configurable
    job = {
        "checkpoint_path": paths.project_relative_path(checkpoint_path),
        "output_dir": paths.project_relative_path(output_dir),
        "device_names": tuple(device_names),
        "batch_size": batch_size,
        "starter_tokens": starter_tokens,
        "max_new_tokens": max_new_tokens,
        "temperature": float(conf.sampling.temperature),
        "top_k": top_k,
        "dtype_name": conf.sampling.dtype,
        "seed": conf.sampling.seed,
        "compile": conf.sampling.compile,
        "total_starters": total_starters,
        "keep_shards": False,
        "log_interval": 1000, # @TODO: maybe make configurable?
        "output_csv_name": output_csv_name,
        "require_batch_generate": require_batch_generate,
        "tokenized_data_filepath": str(tmd_conf.tokenized_data_filepath),
        "tokenized_dtype": np.dtype(tmd_conf.dtype).name,
        "sequence_length": int(tmd_conf.sequence_length),
        "split_start_token_idx": int(raw_start),
        "max_test_sequences": max_test_sequences_metadata,
    }

    print(f"Sampling from checkpoint: {checkpoint_path}")
    print(f"Sampling output directory: {output_dir}")
    print(f"Sampling test starters: {total_starters:,} / {dls_conf.num_sequences:,}")
    print(f"Starter tokens per event: {starter_tokens}")
    print(f"New tokens per event: {max_new_tokens}")
    print(f"Batch size per worker: {batch_size}")
    print(f"Worker devices: {', '.join(device_names)}")

    world_size = len(device_names)
    if world_size == 1:
        sampling_worker(0, job)
    else:
        mp.spawn(sampling_worker, args=(job,), nprocs=world_size, join=True)

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

    sampling_elapsed_seconds = time.time() - sample_t0
    num_gpus_used = sum(1 for device_name in device_names if device_name.startswith("cuda"))

    metadata = dict(job)
    metadata["sample_idx"] = sample_idx
    metadata["sampling_dir_name"] = f"sampling_{sample_idx}"
    metadata["model_name"] = conf.generic.model_name
    metadata["final_csv_path"] = str(final_path)
    metadata["sampling_metadata_path"] = str(output_dir / "sampling_metadata.json")
    metadata["split"] = "test_bin"
    metadata["starter_source"] = "first tokens of the first configured test_bin sequences"
    metadata["sampling_elapsed_seconds"] = sampling_elapsed_seconds
    metadata["num_gpus_used"] = num_gpus_used
    metadata["num_devices_used"] = len(device_names)

    for path_key in ("output_dir", "final_csv_path", "sampling_metadata_path"):
        metadata[path_key] = os.path.relpath(metadata[path_key], paths.PROJECT_DIR)

    with (output_dir / "sampling_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Finished. Final generated samples CSV: {final_path}")
    print(f"Sampling took {sampling_elapsed_seconds:.2f} seconds using {num_gpus_used} GPU(s).")

if __name__ == "__main__":
    main()
