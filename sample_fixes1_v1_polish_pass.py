"""
Sample sequences from a trained particleGPT model.

This script is intentionally executable as a script, not as an import-time program:
    python sample.py

Main design goals:
    1. Keep all side effects inside main(), so multiprocessing workers are safe.
    2. Validate required files before loading large checkpoints/models.
    3. Use one clear sampling path for CPU/single-GPU and a separate explicit path
       for multi-GPU sampling.
    4. Write samples atomically so interrupted jobs do not leave a partially named
       final output file.

Output format:
    One generated event per line, with token ids separated by spaces. The filename
    keeps the historical .csv suffix for compatibility with existing utilities.
"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
from timeit import default_timer as timer
from typing import Iterator

import numpy as np
import torch
import torch.multiprocessing as mp

import pLogging
import pUtil
import configurator as conf
import model_fixes1_v1 as model_module
from model_fixes1_v1 import GPTConfig, GPT
from dictionary import Dictionary

script_dir = Path(__file__).resolve().parent

# =====================
# Small data containers
# =====================

@dataclass(frozen=True)
class SamplingPaths:
    """All filesystem paths used by one sampling run."""

    prep_info: Path
    prep_data: Path
    test_bin: Path
    checkpoint: Path
    samples_output: Path

@dataclass(frozen=True)
class SamplingRuntime:
    """Torch runtime settings used for sampling."""

    device: torch.device
    device_type: str
    dtype: torch.dtype
    seed: int

    def autocast_context(self):
        """Return the right autocast context for this runtime."""
        if self.device_type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            return torch.amp.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

# =====================
# Configuration/path helpers
# =====================

def build_sampling_paths(sampling_id: int) -> SamplingPaths:
    """Resolve every path needed by this sampling run."""
    prep_dir = script_dir / "preparations" / conf.generic.preparation_name
    sampling_dir = pUtil.get_sampling_dir(conf.generic.model_name) / f"sampling_{sampling_id}"

    return SamplingPaths(
        prep_info=prep_dir / "preparation_info.json",
        prep_data=prep_dir / "preparation.json",
        test_bin=prep_dir / "test.bin",
        checkpoint=pUtil.get_training_dir(conf.generic.model_name) / "ckpt.pt",
        samples_output=sampling_dir / "generated_samples.csv",
    )

def validate_required_files(paths: SamplingPaths) -> None:
    """Fail early if any required input file is missing."""
    missing = [
        path for path in (paths.prep_info, paths.prep_data, paths.test_bin, paths.checkpoint)
        if not path.exists()
    ]
    if missing:
        missing_str = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required sampling input file(s):\n{missing_str}")

def load_json_file(path: Path) -> dict:
    """Read a JSON file with a useful error message."""
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Could not read JSON file {path}: {exc}") from exc

def load_dictionary(paths: SamplingPaths) -> Dictionary:
    """Load the dictionary associated with the prepared tokenized dataset."""
    prep_data = load_json_file(paths.prep_data)
    tokenized_dataset_name = prep_data["train_bin"]["tokenized_dataset"]
    dictionary_path = script_dir / "data" / "tokenized" / tokenized_dataset_name / "dictionary.json"
    return Dictionary(dictionary_path)

def resolve_starter_len(max_sequence_len: int) -> int:
    """
    Determine prompt length.

    Default is 5 for the current event format:
        EVENT_START, incident PDGID, incident PT, incident ETA, incident PHI
    """
    starter_len = int(getattr(conf.sampling, "starter_len", 5))
    if starter_len <= 0:
        raise ValueError(f"conf.sampling.starter_len must be positive, got {starter_len}")
    if starter_len > max_sequence_len:
        raise ValueError(
            f"starter_len={starter_len} exceeds max_sequence_length={max_sequence_len}"
        )
    return starter_len

def resolve_max_new_tokens(max_sequence_len: int, starter_len: int) -> int:
    """Use the configured max_new_tokens without exceeding the prepared sequence length."""
    max_possible = max_sequence_len - starter_len
    configured = getattr(conf.sampling, "max_new_tokens", -1)

    if configured is None or int(configured) < 0:
        return max_possible

    return min(int(configured), max_possible)

def resolve_num_samples(total_events: int) -> int:
    """Allow optional subsampling through conf.sampling.num_samples; default samples all."""
    configured = int(getattr(conf.sampling, "num_samples", -1))
    if configured <= 0:
        return total_events
    return min(configured, total_events)

def resolve_batch_size() -> int:
    """Return the per-device generation batch size."""
    batch_size = int(getattr(conf.sampling, "batch_size", 128))
    if batch_size <= 0:
        raise ValueError(f"conf.sampling.batch_size must be positive, got {batch_size}")
    return batch_size

def resolve_temperature() -> float:
    """Validate and return sampling temperature."""
    temperature = float(getattr(conf.sampling, "temperature", 1.0))
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    return temperature

def resolve_top_k():
    """Normalize top_k so non-positive values mean no top-k filtering."""
    top_k = getattr(conf.sampling, "top_k", None)
    if top_k is None:
        return None
    top_k = int(top_k)
    return top_k if top_k > 0 else None

def resolve_log_every_chunks() -> int:
    """Return progress-log interval in chunks; non-positive disables chunk logging."""
    return int(getattr(conf.sampling, "log_every_chunks", 10))

def resolve_chunk_size(num_samples: int, batch_size: int, use_multi_gpu: bool) -> int:
    """
    Determine how many starters to process before writing to disk.

    Multi-GPU default: process all requested samples at once to avoid repeatedly
    respawning workers. For very large test sets, set conf.sampling.chunk_size.

    Single-device default: process one generation batch at a time to keep memory bounded.
    """
    default_chunk_size = num_samples if use_multi_gpu else batch_size
    chunk_size = int(getattr(conf.sampling, "chunk_size", default_chunk_size))
    if chunk_size <= 0:
        chunk_size = default_chunk_size
    return max(1, chunk_size)

# =====================
# Torch/model helpers
# =====================

def dtype_from_config(dtype_name: str) -> torch.dtype:
    """Map config dtype strings to torch dtypes."""
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unknown sampling dtype {dtype_name!r}; expected one of {sorted(dtype_map)}")
    return dtype_map[dtype_name]

def setup_torch_runtime() -> SamplingRuntime:
    """Set seeds/backends and return the runtime object used by generation."""
    seed = int(getattr(conf.sampling, "seed", 1337))
    device = torch.device(conf.sampling.device)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("conf.sampling.device requests CUDA, but CUDA is not available.")
        device_index = device.index if device.index is not None else 0
        torch.cuda.set_device(device_index)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(seed)

    return SamplingRuntime(
        device=device,
        device_type=device_type,
        dtype=dtype_from_config(conf.sampling.dtype),
        seed=seed,
    )

def torch_load_checkpoint(path: Path, map_location):
    """Load checkpoints across PyTorch versions with explicit pickle behavior when supported."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def strip_known_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Clean state dict keys saved from DDP and/or torch.compile wrappers.

    Handles old checkpoints containing prefixes such as:
        module.transformer...
        _orig_mod.transformer...
        module._orig_mod.transformer...
    """
    prefixes = ("module.", "_orig_mod.")
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value

    return cleaned

def load_checkpoint_payload(checkpoint_path: Path) -> tuple[GPTConfig, dict[str, torch.Tensor]]:
    """Load model config and a CPU state dict from a checkpoint."""
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")

    if "model_args" not in checkpoint or "model" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} must contain 'model_args' and 'model' entries."
        )

    gpt_config = GPTConfig(**checkpoint["model_args"])
    state_dict = strip_known_state_dict_prefixes(checkpoint["model"])
    state_dict = {key: value.detach().cpu() for key, value in state_dict.items()}
    return gpt_config, state_dict

def build_model(
    gpt_config: GPTConfig,
    state_dict: dict[str, torch.Tensor],
    runtime: SamplingRuntime,
    *,
    compile_model: bool,
    logger_idx: int,
):
    """Construct, load, move, optionally compile, and return a model for single-device sampling."""
    sampling_model = GPT(gpt_config)
    sampling_model.load_state_dict(state_dict)
    sampling_model.to(runtime.device)
    sampling_model.eval()

    if not compile_model:
        return sampling_model

    try:
        compiled = torch.compile(sampling_model, dynamic=False, fullgraph=False)
        # Some PyTorch versions expose custom nn.Module methods through the compiled wrapper;
        # some do not. Sampling relies on generate*/forward_prefill/forward_decode methods.
        if hasattr(compiled, "generate"):
            pLogging.info(logger_idx, "Sampling info", {"compiled_model": True})
            return compiled
    except Exception as exc:
        pLogging.info(logger_idx, "Sampling info", {"compiled_model": False, "compile_error": str(exc)})
        return sampling_model

    pLogging.info(
        logger_idx,
        "Sampling info",
        {"compiled_model": False, "reason": "compiled wrapper did not expose generate()"},
    )
    return sampling_model

def should_use_multi_gpu(runtime: SamplingRuntime) -> bool:
    """Decide whether to use all visible CUDA devices for sampling."""
    if runtime.device_type != "cuda":
        return False

    num_cuda = torch.cuda.device_count()
    if num_cuda <= 1:
        return False

    configured = getattr(conf.sampling, "multi_gpu", None)
    if configured is not None:
        return bool(configured)

    # If the user explicitly selected cuda:N, treat that as a single-device request
    # unless conf.sampling.multi_gpu=True is set.
    if runtime.device.index is not None:
        return False

    return True

# =====================
# Data iteration/output helpers
# =====================

def open_test_memmap(test_bin_path: Path, max_sequence_len: int) -> np.memmap:
    """Open test.bin as a 2D event matrix of uint16 token ids."""
    raw = np.memmap(test_bin_path, dtype=np.uint16, mode="r")
    if raw.size % max_sequence_len != 0:
        raise ValueError(
            f"test.bin has {raw.size} tokens, which is not divisible by "
            f"max_sequence_length={max_sequence_len}."
        )
    return raw.reshape(-1, max_sequence_len)

def iter_starter_batches(
    test_data: np.memmap,
    *,
    starter_len: int,
    num_samples: int,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    """Yield copied int64 starter-token batches from the test memmap."""
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        yield np.asarray(test_data[start:end, :starter_len], dtype=np.int64)

def trim_generated_sequence(
    seq: list[int],
    dictionary: Dictionary,
    *,
    trim_after_event_end: bool,
) -> list[int]:
    """Trim tokens after EVENT_END, preserving EVENT_END itself."""
    if not trim_after_event_end:
        return seq

    event_end_token = int(dictionary.event_end_token)
    if event_end_token in seq:
        return seq[: seq.index(event_end_token) + 1]
    return seq

def write_generated_sequences(
    out_file,
    generated: torch.Tensor,
    dictionary: Dictionary,
    *,
    trim_after_event_end: bool,
) -> None:
    """Write generated token sequences to the output file."""
    for seq in generated.cpu():
        seq_list = [int(tok) for tok in seq.tolist()]
        seq_list = trim_generated_sequence(
            seq_list,
            dictionary,
            trim_after_event_end=trim_after_event_end,
        )
        out_file.write(" ".join(map(str, seq_list)) + "\n")

# =====================
# Generation paths
# =====================

def generate_single_device(
    sampling_model,
    starters_np: np.ndarray,
    runtime: SamplingRuntime,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k,
    batch_size: int,
    grammar_mask: bool,
) -> torch.Tensor:
    """Generate a batch on CPU or one CUDA device."""
    starters = torch.as_tensor(starters_np, dtype=torch.long, device=runtime.device)

    with torch.inference_mode(), runtime.autocast_context():
        if hasattr(sampling_model, "generate_batched_singleGPU"):
            generated = sampling_model.generate_batched_singleGPU(
                starters,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                batch_size=batch_size,
                grammar_mask=grammar_mask,
            )
        else:
            # Backward-compatible fallback for older model files.
            generated = sampling_model.generate(
                starters,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                grammar_mask=grammar_mask,
            )

    return generated.cpu()

def _multi_gpu_worker(
    split_index: int,
    device_id: int,
    gpt_config: GPTConfig,
    state_dict_cpu: dict[str, torch.Tensor],
    starters_cpu: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k,
    batch_size: int,
    grammar_mask: bool,
    dtype_name: str,
    seed: int,
    compile_worker_model: bool,
    return_queue,
) -> None:
    """Worker process used by generate_multi_gpu()."""
    try:
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        torch.manual_seed(seed + split_index)
        torch.cuda.manual_seed_all(seed + split_index)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        dtype = dtype_from_config(dtype_name)
        runtime = SamplingRuntime(
            device=device,
            device_type="cuda",
            dtype=dtype,
            seed=seed + split_index,
        )

        worker_model = GPT(gpt_config)
        worker_model.load_state_dict(state_dict_cpu)
        worker_model.to(device)
        worker_model.eval()

        if compile_worker_model:
            worker_model = torch.compile(worker_model, dynamic=False, fullgraph=False)

        starters = starters_cpu.to(device, non_blocking=True)

        with torch.inference_mode(), runtime.autocast_context():
            if hasattr(worker_model, "generate_batched_singleGPU"):
                generated = worker_model.generate_batched_singleGPU(
                    starters,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    batch_size=batch_size,
                    grammar_mask=grammar_mask,
                )
            else:
                generated = worker_model.generate(
                    starters,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    grammar_mask=grammar_mask,
                )

        return_queue.put((split_index, "ok", generated.cpu()))
    except Exception:
        return_queue.put((split_index, "error", traceback.format_exc()))

def generate_multi_gpu(
    gpt_config: GPTConfig,
    state_dict_cpu: dict[str, torch.Tensor],
    starters_np: np.ndarray,
    runtime: SamplingRuntime,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k,
    batch_size: int,
    grammar_mask: bool,
    compile_worker_model: bool,
) -> torch.Tensor:
    """Generate a batch by splitting it across all visible CUDA devices."""
    if runtime.device_type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU sampling requires CUDA.")

    num_devices = torch.cuda.device_count()
    if num_devices <= 1:
        raise RuntimeError("Multi-GPU sampling requested, but only one CUDA device is visible.")

    starters_cpu = torch.as_tensor(starters_np, dtype=torch.long, device="cpu")
    starter_splits = torch.chunk(starters_cpu, num_devices, dim=0)

    ctx = mp.get_context("spawn")
    return_queue = ctx.Queue()
    processes = []

    for split_index, starters_chunk in enumerate(starter_splits):
        if starters_chunk.numel() == 0:
            continue

        process = ctx.Process(
            target=_multi_gpu_worker,
            args=(
                split_index,
                split_index,  # device id in the visible CUDA namespace
                gpt_config,
                state_dict_cpu,
                starters_chunk,
                max_new_tokens,
                temperature,
                top_k,
                batch_size,
                grammar_mask,
                conf.sampling.dtype,
                runtime.seed,
                compile_worker_model,
                return_queue,
            ),
        )
        process.start()
        processes.append((split_index, process))

    results: dict[int, torch.Tensor] = {}
    errors: list[str] = []

    for _ in processes:
        split_index, status, payload = return_queue.get()
        if status == "ok":
            results[split_index] = payload
        else:
            errors.append(f"Worker split {split_index} failed:\n{payload}")

    for split_index, process in processes:
        process.join()
        if process.exitcode != 0:
            errors.append(f"Worker split {split_index} exited with code {process.exitcode}")

    if errors:
        raise RuntimeError("\n".join(errors))

    ordered = [results[idx] for idx, _ in processes]
    return torch.cat(ordered, dim=0)

# =====================
# Main script
# =====================

def main() -> None:
    """Run one sampling job."""
    sampling_id = pUtil.get_latest_sampling_id(conf.generic.model_name) + 1
    paths = build_sampling_paths(sampling_id)
    paths.samples_output.parent.mkdir(parents=True, exist_ok=True)

    logger_idx = pLogging.create_sampling_logger(conf.generic.model_name, sampling_id)
    model_module.set_logger(logger_idx)

    start_time = timer()

    try:
        validate_required_files(paths)

        runtime = setup_torch_runtime()
        dictionary = load_dictionary(paths)
        prep_info = load_json_file(paths.prep_info)
        max_sequence_len = int(prep_info["max_sequence_length"])

        starter_len = resolve_starter_len(max_sequence_len)
        max_new_tokens = resolve_max_new_tokens(max_sequence_len, starter_len)
        batch_size = resolve_batch_size()
        temperature = resolve_temperature()
        top_k = resolve_top_k()
        log_every_chunks = resolve_log_every_chunks()

        test_data = open_test_memmap(paths.test_bin, max_sequence_len)
        num_samples = resolve_num_samples(total_events=len(test_data))

        gpt_config, state_dict_cpu = load_checkpoint_payload(paths.checkpoint)
        use_multi_gpu = should_use_multi_gpu(runtime)
        chunk_size = resolve_chunk_size(num_samples, batch_size, use_multi_gpu)

        grammar_mask = bool(getattr(conf.sampling, "grammar_mask", False))
        trim_after_event_end = bool(getattr(conf.sampling, "trim_after_event_end", True))
        compile_single_device = bool(getattr(conf.sampling, "compile", False)) and not use_multi_gpu
        compile_worker_model = bool(getattr(conf.sampling, "compile_workers", False)) and use_multi_gpu

        sampling_model = None
        if not use_multi_gpu:
            sampling_model = build_model(
                gpt_config, state_dict_cpu, runtime,
                compile_model=compile_single_device,
                logger_idx=logger_idx,
            )

        pLogging.info(logger_idx, "Sample generation started.")
        pLogging.info(logger_idx, "Sampling info", {
            "preparation": conf.generic.preparation_name,
            "model_path": str(paths.checkpoint),
            "samples_output_filename": str(paths.samples_output),
            "num_samples": num_samples,
            "total_test_events": len(test_data),
            "starter_len": starter_len,
            "max_sequence_length": max_sequence_len,
            "configured_max_new_tokens": getattr(conf.sampling, "max_new_tokens", -1),
            "effective_max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "grammar_mask": grammar_mask,
            "trim_after_event_end": trim_after_event_end,
            "seed": runtime.seed,
            "device": str(runtime.device),
            "device_type": runtime.device_type,
            "dtype": conf.sampling.dtype,
            "compile_single_device": compile_single_device,
            "use_multi_gpu": use_multi_gpu,
            "num_cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "compile_worker_model": compile_worker_model,
        })

        temp_output = paths.samples_output.with_suffix(paths.samples_output.suffix + ".tmp")

        with temp_output.open("w") as out_file:
            for batch_idx, starters_np in enumerate(iter_starter_batches(test_data, starter_len=starter_len, num_samples=num_samples, chunk_size=chunk_size)):
                if use_multi_gpu:
                    generated = generate_multi_gpu(
                        gpt_config, state_dict_cpu, starters_np, runtime,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        batch_size=batch_size,
                        grammar_mask=grammar_mask,
                        compile_worker_model=compile_worker_model,
                    )
                else:
                    generated = generate_single_device(
                        sampling_model, starters_np, runtime,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        batch_size=batch_size,
                        grammar_mask=grammar_mask,
                    )

                write_generated_sequences(out_file, generated, dictionary, trim_after_event_end=trim_after_event_end)

                if log_every_chunks > 0 and (batch_idx + 1) % log_every_chunks == 0:
                    pLogging.info(logger_idx, "Sampling progress", {
                        "chunks_completed": batch_idx + 1,
                        "samples_completed_upper_bound": min((batch_idx + 1) * chunk_size, num_samples),
                    })

        os.replace(temp_output, paths.samples_output)

        elapsed = timer() - start_time
        pLogging.info(
            logger_idx,
            f"Sample generation finished in {elapsed} seconds.",
            {"time_seconds": elapsed, "samples_written": num_samples},
        )

    except Exception as exc:
        pLogging.info(logger_idx, "Sample generation failed.", {"error": str(exc)})
        raise

if __name__ == "__main__":
    main()